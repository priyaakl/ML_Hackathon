"""Person 2: evaluation harness (the "test rig" every model goes through).

What it does
  1. load_partition()      reads candidates_<part>.parquet, keeps ONLY rows with label_valid == True
  2. evaluate()            turns per-row probabilities into the metrics the report needs
  3. cross_validate()      5-fold CV where whole trips stay together (GroupKFold on trip_id)
  4. log_experiment()      appends one row per experiment to results/experiments.csv

Metric definitions (write these in the report):
  top1_all        for each valid GPS point, take the highest-scored candidate; correct if it is the reference arc.
                  Points whose true arc is not a candidate, and points with no candidates, count as WRONG.
                  So top1_all can never exceed 1 - miss_rate.
  top1_in_cands   same, but only over points whose true arc IS in the candidate set (isolates model errors).
  miss_rate       share of valid points whose true arc is not a candidate (same definition as Person 1's stats).
  macro_f1        candidate-level, threshold 0.5, average of F1(class 0) and F1(class 1).
  log_loss        candidate-level binary log loss.
  route_continuity_1hop   share of consecutive predicted arcs that are the same arc or directly connected
                  (end node of arc A == start node of arc B). Simple version; Person 3 has a fuller one.
"""
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.common.config import ROOT, load_config

META_COLS = ["trip_id", "obs_idx", "cand_arc_id", "cand_rank", "label"]
THRESHOLD = 0.5


# --------------------------------------------------------------------------- loading
def load_feature_columns(path=None):
    path = Path(path) if path else ROOT / "models" / "feature_columns.json"
    with open(path) as f:
        return json.load(f)


def load_partition(part, processed_dir=None, sample_trips=None, seed=42):
    """Returns (df, n_valid_by_trip).
    df: one row per (observation, candidate), valid-label trips/points only, sorted by (trip, obs, rank),
        with an extra integer column `gid` that identifies one observation.
    n_valid_by_trip: Series trip_id -> number of valid observations (includes points with zero candidates,
        so that top1_all and miss_rate match Person 1's definition).
    """
    cfg = load_config()
    pdir = Path(processed_dir) if processed_dir else cfg["paths"]["processed"]
    feats = load_feature_columns()
    cand_path = pdir / f"candidates_{part}.parquet"
    if not cand_path.exists():
        raise FileNotFoundError(
            f"{cand_path} not found. These files are gitignored: run `python -m src.data.candidates` "
            "(needs the raw dataset) or get the parquet files from Person 1.")
    df = pd.read_parquet(cand_path, columns=META_COLS + ["label_valid"] + feats)
    df = df[df["label_valid"].eq(True)].drop(columns="label_valid")

    obs = pd.read_parquet(pdir / f"observations_{part}.parquet", columns=["trip_id", "obs_idx", "label_valid"])
    obs = obs[obs["label_valid"].eq(True)]
    n_valid = obs.groupby("trip_id").size()

    if sample_trips:  # quick smoke runs: use only N trips
        trips = sorted(df["trip_id"].unique())
        keep = np.random.default_rng(seed).choice(trips, size=min(sample_trips, len(trips)), replace=False)
        df = df[df["trip_id"].isin(keep)]
        n_valid = n_valid[n_valid.index.isin(keep)]

    df = df.sort_values(["trip_id", "obs_idx", "cand_rank"]).reset_index(drop=True)
    codes, _ = pd.factorize(df["trip_id"])
    df["gid"] = codes.astype(np.int64) * 10_000_000 + df["obs_idx"].to_numpy().astype(np.int64)
    df["label"] = df["label"].astype(np.int8)
    return df, n_valid


def n_valid_for(n_valid_by_trip, trips):
    return int(n_valid_by_trip.reindex(list(trips)).fillna(0).sum())


# --------------------------------------------------------------------------- metrics
def top1_pick(df, p):
    """Row positions of the best-scored candidate of every observation (ties -> nearer candidate)."""
    gid = df["gid"].to_numpy()
    if len(gid) == 0:
        return np.array([], dtype=np.int64)
    order = np.lexsort((df["cand_rank"].to_numpy(), -np.asarray(p), gid))
    g = gid[order]
    first = np.r_[True, g[1:] != g[:-1]]
    return order[first]


def evaluate(df, p, n_valid_obs):
    """df must have a default RangeIndex (use reset_index(drop=True))."""
    y = df["label"].to_numpy()
    p = np.asarray(p, dtype=np.float64)
    pick = top1_pick(df, p)
    correct = int(y[pick].sum())
    n_true = int(df.groupby("gid", sort=False)["label"].max().sum())   # points whose true arc is a candidate
    yhat = (p >= THRESHOLD).astype(int)
    return dict(
        top1_all=correct / n_valid_obs,
        top1_in_cands=correct / max(n_true, 1),
        miss_rate=1.0 - n_true / n_valid_obs,
        macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
        n_obs=n_valid_obs, n_rows=len(df),
    )


def route_continuity_1hop(df, p, data_root, partition="student_test"):
    """Share of consecutive predicted arcs that are identical or directly connected.
    Reads each trip's .arcs file (the road network is per trip). Returns NaN if files are missing.
    Points dropped for invalid labels create small gaps, which slightly lowers this number."""
    pick = top1_pick(df, p)
    arcs_pred = df["cand_arc_id"].to_numpy()[pick]
    trips = df["trip_id"].to_numpy()[pick]
    ok_total = total = 0
    for tid in np.unique(trips):
        f = Path(data_root) / partition / tid / f"{tid}.arcs"
        if not f.exists():
            return float("nan")
        arcs = pd.read_csv(f, sep=r"\s+", header=None, names=["u", "v"])
        a = arcs_pred[trips == tid]
        if len(a) < 2:
            continue
        a0, a1 = a[:-1], a[1:]
        ok = (a0 == a1) | (arcs["v"].to_numpy()[a0] == arcs["u"].to_numpy()[a1])
        ok_total += int(ok.sum())
        total += len(ok)
    return ok_total / total if total else float("nan")


# --------------------------------------------------------------------------- cross-validation
def cross_validate(build_fn, df, n_valid_by_trip, feats, n_splits=5, gap_trips=10, seed=42, desc="CV"):
    """Group-aware CV: every trip is entirely in train or entirely in validation.
    build_fn() must return a NEW untrained model each call (so scalers are refit inside every fold).
    gap_trips: also score this many training-fold trips, to measure the train-vs-validation gap (0 = skip)."""
    rng = np.random.default_rng(seed)
    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for k, (tr, va) in enumerate(tqdm(list(splitter.split(df, groups=df["trip_id"])), desc=desc)):
        dtr, dva = df.iloc[tr], df.iloc[va].reset_index(drop=True)
        model = build_fn()
        t0 = time.perf_counter()
        model.fit(dtr[feats], dtr["label"])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pv = model.predict_proba(dva[feats])[:, 1]
        pred_s = time.perf_counter() - t0
        row = {"fold": k, "fit_s": fit_s, "pred_s": pred_s,
               **evaluate(dva, pv, n_valid_for(n_valid_by_trip, dva["trip_id"].unique()))}
        if gap_trips:
            tr_trips = dtr["trip_id"].unique()
            pick = rng.choice(tr_trips, size=min(gap_trips, len(tr_trips)), replace=False)
            dg = dtr[dtr["trip_id"].isin(pick)].reset_index(drop=True)
            m = evaluate(dg, model.predict_proba(dg[feats])[:, 1], n_valid_for(n_valid_by_trip, pick))
            row.update({f"train_{a}": b for a, b in m.items() if a in ("top1_all", "macro_f1", "log_loss")})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- experiment log
def log_experiment(row, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "experiments.csv"
    row = {**row, "date": datetime.datetime.now().isoformat(timespec="seconds")}
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)
