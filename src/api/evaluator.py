"""Server-side evaluation (Person 4).

The prediction pipeline never sees ground truth. AFTER a route is predicted, this module:
  1. reads the trip's reference route (.route) from disk, on the server only;
  2. turns it into one reference arc per GPS point using Person 1's label_trip()
     (the same labelling the models were trained with, so the numbers are comparable);
  3. compares predictions with it and returns ONLY summary numbers.
Raw reference arcs are never returned to the UI.

Metrics (computed on GPS points whose reference label is reliable, label_valid == True):
  top1_accuracy    share of GPS points whose predicted arc equals the reference arc
  macro_f1         F1 per arc id (each arc = one class), averaged with equal weight per arc
  route_continuity share of consecutive predicted arcs that are connected in the road network
                   (Person 3's RoadGraph.route_continuity; does not use ground truth)
Reported for both the HMM route and the classifier-only route.

Batch run over every trip in the API's trips folder (saves a CSV for the report):
    python -m src.api.evaluator
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.common.config import ROOT
from src.common.io import load_trip
from src.data.labels import label_trip


def _scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    if len(ref) == 0:
        return {"top1_accuracy": None, "macro_f1": None}
    return {
        "top1_accuracy": round(float((pred == ref).mean()), 4),
        "macro_f1": round(float(f1_score(ref, pred, average="macro", zero_division=0)), 4),
    }


class Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline     # used only to find the trip folder and its road graph
        self._labels = {}            # trip_id -> reference labels (cached, server memory only)

    def _reference(self, trip_id: str):
        if trip_id not in self._labels:
            trip_dir = self.pipeline.trip_dir(trip_id)
            if not (trip_dir / f"{trip_id}.route").exists():
                self._labels[trip_id] = None
            else:
                self._labels[trip_id] = label_trip(load_trip(trip_dir, with_route=True))
        return self._labels[trip_id]

    def evaluate(self, trip_id: str, predicted_arcs: list, classifier_only_arcs: list) -> dict:
        t0 = time.perf_counter()
        labels = self._reference(trip_id)
        if labels is None:
            return {"available": False, "reason": "no reference route (.route) for this trip on the server"}

        n_obs = len(labels)
        if len(predicted_arcs) != n_obs or len(classifier_only_arcs) != n_obs:
            raise ValueError(f"Prediction length does not match the number of GPS points ({n_obs})")

        valid = labels["label_valid"].to_numpy()
        ref = labels["ref_arc_id"].to_numpy()[valid]
        hmm = np.asarray(predicted_arcs, dtype=np.int64)
        clf = np.asarray(classifier_only_arcs, dtype=np.int64)
        graph = self.pipeline.graph_for(trip_id)

        hmm_m = {**_scores(hmm[valid], ref), "route_continuity": round(float(graph.route_continuity(hmm.tolist())), 4)}
        clf_m = {**_scores(clf[valid], ref), "route_continuity": round(float(graph.route_continuity(clf.tolist())), 4)}
        return {
            "available": True,
            "n_observations": n_obs,
            "n_scored_observations": int(valid.sum()),
            "n_excluded_unreliable_labels": int((~valid).sum()),
            "hmm": hmm_m,
            "classifier_only": clf_m,
            "hmm_minus_classifier_top1": (
                None if hmm_m["top1_accuracy"] is None
                else round(hmm_m["top1_accuracy"] - clf_m["top1_accuracy"], 4)
            ),
            "evaluation_s": round(time.perf_counter() - t0, 3),
        }


def evaluate_all(save: bool = True) -> pd.DataFrame:
    """Run predict + evaluate for every trip in the API's trips folder. Used for testing and the report."""
    from tqdm import tqdm
    from src.api.pipeline import get_pipeline

    pipe = get_pipeline()
    ev = Evaluator(pipe)
    rows = []
    for tid in tqdm(pipe.list_trips(), desc="evaluate"):
        res = pipe.predict_trip(tid)
        m = ev.evaluate(tid, res["predicted_arcs"], res["classifier_only_arcs"])
        row = {"trip_id": tid, "n_obs": res["n_observations"], "predict_total_s": res["timing"]["total_s"]}
        if m["available"]:
            row.update({
                "scored_obs": m["n_scored_observations"], "excluded_obs": m["n_excluded_unreliable_labels"],
                "hmm_top1": m["hmm"]["top1_accuracy"], "hmm_macro_f1": m["hmm"]["macro_f1"],
                "hmm_continuity": m["hmm"]["route_continuity"],
                "clf_top1": m["classifier_only"]["top1_accuracy"], "clf_macro_f1": m["classifier_only"]["macro_f1"],
                "clf_continuity": m["classifier_only"]["route_continuity"],
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    mode = pipe.mode()
    print("\nscorer :", mode["scorer"])
    print("decoder:", mode["decoder"], "|", mode["decoder_params"])
    print(df.to_string(index=False))
    if "hmm_top1" in df:
        print("\nMEAN over trips:", df.drop(columns=["trip_id"]).mean(numeric_only=True).round(4).to_dict())
        # Pooled = every scored GPS point counts equally (comparable with Person 2's harness top1_all).
        w = df["scored_obs"]
        print("POOLED over GPS points: hmm_top1 =", round(float((df.hmm_top1 * w).sum() / w.sum()), 4),
              "| clf_top1 =", round(float((df.clf_top1 * w).sum() / w.sum()), 4))
    if save:
        tag = "MOCK_" if mode["is_mock"] else ""
        out = ROOT / "results" / f"api_eval_{tag}{pipe.trips_dir.name}.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print("\nSaved:", out.relative_to(ROOT))
    return df


if __name__ == "__main__":
    evaluate_all(save="--no-save" not in sys.argv)
