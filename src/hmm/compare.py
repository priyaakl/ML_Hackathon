"""Tune the decoder on training trips and compare routes on the student-test split.

From the repository root, set PATHSNAP_DATA to the extracted Student folder,
then run python -m src.hmm.compare --tune --map-trip 00000065.
"""

import argparse
import json
from copy import copy
from hashlib import sha256
from itertools import groupby, product
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.api.pipeline import classifier_only_arcs
from src.common.config import ROOT, load_config
from src.common.io import load_trip
from src.data.candidates import build_candidates_for_trip
from src.hmm.graph import build_graph
from src.hmm.viterbi import decode_trip
from src.models.model_zoo import build_model
from src.models.scorer import CandidateScorer


DEFAULTS = {
    "transition_scale_m": 30.0,
    "max_network_distance_m": 250.0,
    "disconnected_penalty": 8.0,
    "gap_reset_s": 30.0,
}


def sequence_similarity(prediction, reference):
    """Return 1 minus normalized edit distance after collapsing repeated arcs."""
    a = [arc for arc, _ in groupby(prediction)]
    b = [arc for arc, _ in groupby(reference)]
    longest = max(len(a), len(b))
    if not longest:
        return float("nan")
    if len(b) > len(a):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[-1] + 1,
                previous[j - 1] + (left != right),
            ))
        previous = current
    return 1 - previous[-1] / longest


def load_inputs(cfg, scorer, partition, trip_id, labels):
    trip_dir = cfg["paths"]["data_root"] / partition / trip_id
    cands, obs = build_candidates_for_trip(
        trip_dir,
        radius_m=cfg["candidates"]["radius_m"],
        max_candidates=cfg["candidates"]["max_candidates"],
    )
    cands = cands.sort_values(["obs_idx", "cand_rank"], kind="stable")
    probs = scorer.predict_proba(cands) if len(cands) else np.empty(0)
    graph = build_graph(trip_dir / f"{trip_id}.nodes", trip_dir / f"{trip_id}.arcs")

    aligned = labels.get_group(trip_id).sort_values("obs_idx")
    if not np.array_equal(aligned["obs_idx"].to_numpy(), np.arange(len(obs))):
        raise ValueError(f"Labels do not align with the raw GPS track for {trip_id}.")
    reference = aligned["ref_arc_id"].to_numpy()
    valid = aligned["label_valid"].to_numpy(dtype=bool)
    return trip_dir, cands, probs, graph, len(obs), reference, valid


def training_rows(cfg, trip_ids, labels, features):
    """Rebuild the classifier's labelled training rows with current candidate settings."""
    frames = []
    for trip_id in tqdm(trip_ids, desc="Building training candidates for fold models"):
        reference = labels.get_group(trip_id)
        reference = reference.loc[reference["label_valid"], ["obs_idx", "ref_arc_id"]]
        if reference.empty:
            continue
        candidates, _ = build_candidates_for_trip(
            cfg["paths"]["data_root"] / "train" / trip_id,
            radius_m=cfg["candidates"]["radius_m"],
            max_candidates=cfg["candidates"]["max_candidates"],
        )
        rows = candidates[["trip_id", "obs_idx", "cand_arc_id", *features]].merge(
            reference, on="obs_idx", validate="many_to_one",
        )
        rows["label"] = (rows["cand_arc_id"] == rows["ref_arc_id"]).astype(np.int8)
        frames.append(rows[["trip_id", *features, "label"]])
    if not frames:
        raise ValueError("No training trips have valid reference labels.")
    return pd.concat(frames, ignore_index=True)


def tune_decoder(cfg, scorer, trip_ids, labels, model_meta):
    """Grid search on out-of-fold classifier scores from up to eight training trips."""
    if model_meta["feature_columns"] != scorer.feature_columns:
        raise ValueError("Saved model metadata and scorer feature columns disagree.")
    training = labels.size().reindex(trip_ids)
    if not len(training) or training.isna().any():
        raise ValueError("Every training trip must have aligned labels.")
    ordered = training.sort_values(kind="stable").index.to_numpy()
    size = min(8, len(ordered))
    # Mid-quantile samples cover trip lengths while avoiding the most extreme trips.
    selected = ordered[((np.arange(size) + 0.5) * len(ordered) / size).astype(int)]
    settings = [
        {**DEFAULTS, "transition_scale_m": scale, "disconnected_penalty": penalty}
        for scale, penalty in product((15.0, 30.0, 60.0), (4.0, 8.0))
    ]
    trip_ids = np.asarray(trip_ids)
    n_splits = min(5, len(trip_ids))
    if n_splits < 2:
        raise ValueError("Out-of-fold tuning needs at least two training trips.")
    features = scorer.feature_columns
    train = training_rows(cfg, trip_ids, labels, features)
    rows = []
    folds = GroupKFold(n_splits=n_splits).split(trip_ids, groups=trip_ids)
    with tqdm(total=len(selected) * len(settings), desc="Decoder tuning (out-of-fold)") as progress:
        for fold, (fit_idx, holdout_idx) in enumerate(folds, start=1):
            holdout_ids = trip_ids[holdout_idx]
            to_score = selected[np.isin(selected, holdout_ids)]
            if not len(to_score):
                continue
            fit = train.loc[train["trip_id"].isin(trip_ids[fit_idx])]
            fold_scorer = copy(scorer)
            fold_scorer.model = build_model(
                model_meta["model"], model_meta["seed"], model_meta["config"],
            )
            fold_scorer.model.fit(fit[features], fit["label"])
            for trip_id in to_score:
                _, cands, probs, graph, n_obs, reference, valid = load_inputs(
                    cfg, fold_scorer, "train", trip_id, labels
                )
                for params in settings:
                    start = perf_counter()
                    prediction = decode_trip(cands, probs, graph, {**params, "n_observations": n_obs})
                    elapsed = perf_counter() - start
                    rows.append({
                        "trip_id": trip_id,
                        "holdout_fold": fold,
                        "fit_trip_count": len(fit_idx),
                        **params,
                        "n_valid": int(valid.sum()),
                        "correct": int(np.count_nonzero(np.asarray(prediction)[valid] == reference[valid])),
                        "continuity": graph.route_continuity(prediction, max_hops=2),
                        "decode_s": elapsed,
                    })
                    progress.update()
    trials = pd.DataFrame(rows)
    ranked = trials.groupby(list(DEFAULTS), as_index=False).agg(
        correct=("correct", "sum"), n_valid=("n_valid", "sum"),
        mean_continuity=("continuity", "mean"),
    )
    if not ranked["n_valid"].gt(0).all():
        raise ValueError("Decoder tuning needs at least one valid reference observation.")
    ranked["top1_accuracy"] = ranked["correct"] / ranked["n_valid"]
    ranked = ranked.sort_values(["top1_accuracy", "mean_continuity"], ascending=False)
    best = {name: float(ranked.iloc[0][name]) for name in DEFAULTS}
    return best, trials, ranked, selected.tolist()


def compare(cfg, scorer, trip_ids, labels, params, map_trip=None):
    rows = []
    for trip_id in tqdm(trip_ids, desc="Student-test comparison"):
        directory, cands, probs, graph, n_obs, reference, valid = load_inputs(
            cfg, scorer, "student_test", trip_id, labels
        )
        for method, settings in (
            ("classifier_only", None), ("viterbi_default", DEFAULTS), ("viterbi_tuned", params)
        ):
            start = perf_counter()
            if settings is None:
                prediction = classifier_only_arcs(cands, probs, n_obs)
            else:
                prediction = decode_trip(cands, probs, graph, {**settings, "n_observations": n_obs})
            elapsed = perf_counter() - start
            prediction = np.asarray(prediction)
            predicted_valid, reference_valid = prediction[valid], reference[valid]
            correct = int(np.count_nonzero(predicted_valid == reference_valid))
            n_valid = len(reference_valid)
            macro_f1 = np.nan
            if n_valid:
                macro_f1 = f1_score(reference_valid, predicted_valid, average="macro", zero_division=0)
            rows.append({
                "trip_id": trip_id,
                "method": method,
                "n_obs": n_obs,
                "n_valid": n_valid,
                "correct": correct,
                "top1_accuracy": correct / n_valid if n_valid else np.nan,
                "arc_macro_f1": macro_f1,
                "sequence_similarity": sequence_similarity(
                    predicted_valid.tolist(), reference_valid.tolist()
                ),
                "continuity": graph.route_continuity(prediction, max_hops=2),
                "decode_s": elapsed,
            })
            if trip_id == map_trip and method == "viterbi_tuned":
                save_map(cfg, directory, graph, prediction)
    return pd.DataFrame(rows)


def save_map(cfg, trip_dir, graph, prediction):
    from src.viz.folium_maps import _export_demo_image, render_trip_map

    trip = load_trip(trip_dir, with_route=True)
    track = trip["track"][["lon", "lat", "t"]].to_numpy()
    caption = "Student-test trip · Trained classifier + Viterbi"
    stem = cfg["paths"]["results"] / "maps" / f"hmm_student_test_{trip['trip_id']}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    leaflet_dir = ROOT / "src" / "ui" / "web" / "vendor" / "leaflet"
    render_trip_map(
        track, graph, prediction, reference_arcs=trip["route"],
        trip_id=trip["trip_id"], caption=caption,
        leaflet_dir=leaflet_dir if leaflet_dir.is_dir() else None,
    ).save(stem.with_suffix(".html"))
    _export_demo_image(
        ROOT / "report" / "figures" / stem.with_suffix(".png").name, track, graph, trip["route"], prediction, trip["trip_id"], caption,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true", help="Run the training-only decoder grid search.")
    parser.add_argument("--map-trip", help="Export a map for one student-test trip.")
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg["paths"]["processed"]
    results = cfg["paths"]["results"]
    results.mkdir(parents=True, exist_ok=True)
    splits = json.loads((processed / "splits.json").read_text())
    if args.map_trip and args.map_trip not in splits["student_test"]:
        parser.error("--map-trip must be listed in the student-test split.")
    model = ROOT / "models" / "final_model.joblib"
    features = ROOT / "models" / "feature_columns.json"
    scorer = CandidateScorer(str(model), str(features))
    config_path = ROOT / "models" / "hmm_config.json"
    if args.tune:
        labels = pd.read_parquet(processed / "labels_train.parquet").groupby("trip_id")
        model_meta = json.loads((ROOT / "models" / "final_model_meta.json").read_text())
        params, trials, ranked, selected = tune_decoder(
            cfg, scorer, splits["train"], labels, model_meta,
        )
        config_path.write_text(json.dumps(params, indent=2) + "\n")
        trials.to_csv(results / "hmm_tuning_by_trip.csv", index=False)
        ranked.to_csv(results / "hmm_tuning_summary.csv", index=False)
        (results / "hmm_tuning_metadata.json").write_text(json.dumps({
            "model_sha256": sha256(model.read_bytes()).hexdigest(),
            "feature_columns_sha256": sha256(features.read_bytes()).hexdigest(),
            "candidate_settings": cfg["candidates"],
            "partition": "train",
            "scorer": model_meta["model"],
            "probability_source": "out-of-fold, grouped by trip",
            "classifier_folds": min(5, len(splits["train"])),
            "trip_ids": selected,
            "selection_metric": "pointwise accuracy, then mean two-hop continuity",
        }, indent=2) + "\n")
    else:
        params = json.loads(config_path.read_text())

    labels = pd.read_parquet(processed / "labels_student_test.parquet").groupby("trip_id")
    per_trip = compare(cfg, scorer, splits["student_test"], labels, params, args.map_trip)
    per_trip.to_csv(results / "hmm_student_test_by_trip.csv", index=False)
    summary = per_trip.groupby("method").agg(
        correct=("correct", "sum"), n_valid=("n_valid", "sum"),
        mean_arc_macro_f1=("arc_macro_f1", "mean"),
        mean_sequence_similarity=("sequence_similarity", "mean"),
        mean_continuity=("continuity", "mean"),
        median_decode_s=("decode_s", "median"),
    )
    summary["top1_accuracy"] = summary["correct"] / summary["n_valid"]
    summary.to_csv(results / "hmm_student_test_summary.csv")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
