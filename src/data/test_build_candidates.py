"""Task 1.6 check: build_candidates_for_trip() must work from raw files only and match the batch table.

Put this file at  src/data/test_build_candidates.py  and run from the repo root:

    python -m src.data.test_build_candidates                 # 3 student_test trips
    python -m src.data.test_build_candidates --all           # all 10 student_test trips
    python -m src.data.test_build_candidates --trips 00000065 00000007

What it proves, per trip:
  1. NO-LABEL PATH   the trip is copied to a temp folder holding ONLY .track/.nodes/.arcs (no .route),
                     so the function cannot read a label file even by accident.
  2. COLUMNS         every column in models/feature_columns.json is present (and the contract columns too).
  3. MATCHES BATCH   same (obs_idx, cand_arc_id) pairs as data/processed/candidates_student_test.parquet,
                     and the same feature values (float32 tolerance).
  4. DETERMINISTIC   two calls give identical output.
  5. SANE            no NaN/inf in features, one row group per observation, obs count == track rows.
Exit code is 1 if any check fails, so it can be wired into a pre-demo checklist.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.data.candidates import build_candidates_for_trip

CONTRACT_COLS = ["trip_id", "obs_idx", "t", "lon", "lat", "x", "y", "cand_arc_id", "cand_rank"]


def compare(api, batch, feats):
    """Return a list of problem strings (empty = match). Pure function, easy to unit test."""
    problems = []
    key = ["obs_idx", "cand_arc_id"]
    a = api.sort_values(key).reset_index(drop=True)
    b = batch.sort_values(key).reset_index(drop=True)
    if len(a) != len(b):
        problems.append(f"row count differs: API {len(a)} vs batch {len(b)}")
    m = a[key].merge(b[key], on=key, how="outer", indicator=True)
    n_only_api, n_only_batch = int((m["_merge"] == "left_only").sum()), int((m["_merge"] == "right_only").sum())
    if n_only_api or n_only_batch:
        problems.append(f"candidate pairs differ: {n_only_api} only in API, {n_only_batch} only in batch")
        return problems
    # same pairs -> compare feature values row by row
    for c in feats:
        x, y = a[c].to_numpy(dtype=np.float64), b[c].to_numpy(dtype=np.float64)
        if not np.allclose(x, y, rtol=1e-5, atol=1e-4, equal_nan=True):
            bad = int((~np.isclose(x, y, rtol=1e-5, atol=1e-4, equal_nan=True)).sum())
            # dist_rank / cand_rank can legitimately differ on exact-distance ties (twin arcs, see data notes)
            tag = " (tie-break on twin arcs?)" if c in ("dist_rank", "cand_rank") else ""
            problems.append(f"feature {c}: {bad} rows differ{tag}")
    return problems


def check_trip(trip_dir, batch_all, feats, radius, cap):
    tid = trip_dir.name
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / tid
        clean.mkdir()
        for ext in ("track", "nodes", "arcs"):           # deliberately NOT .route
            shutil.copy(trip_dir / f"{tid}.{ext}", clean / f"{tid}.{ext}")
        assert not any(clean.glob("*.route"))
        cands, obs = build_candidates_for_trip(clean, radius_m=radius, max_candidates=cap)
        cands2, _ = build_candidates_for_trip(clean, radius_m=radius, max_candidates=cap)

    missing = [c for c in feats + CONTRACT_COLS if c not in cands.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if "label" in cands.columns:
        problems.append("a 'label' column is present in the no-label output")
    if not cands.equals(cands2):
        problems.append("two calls gave different output (not deterministic)")
    n_track = len(pd.read_csv(trip_dir / f"{tid}.track", sep=r"\s+", header=None))
    if len(obs) != n_track:
        problems.append(f"obs rows {len(obs)} != track rows {n_track}")
    f = cands[feats].to_numpy(dtype=np.float64)
    if not np.isfinite(f).all():
        problems.append("NaN/inf in feature columns")
    if cands["cand_rank"].min() != 1:
        problems.append("cand_rank does not start at 1")
    batch = batch_all[batch_all["trip_id"] == tid]
    if batch.empty:
        problems.append("trip not found in candidates_student_test.parquet (cannot compare with batch)")
    elif not missing:
        problems += compare(cands, batch, feats)
    return problems, len(cands), len(obs), int((obs["n_cands"] == 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    root = cfg["paths"]["data_root"] / "student_test"
    proc = cfg["paths"]["processed"]
    radius, cap = cfg["candidates"]["radius_m"], cfg["candidates"]["max_candidates"]
    feats = json.loads((proc.parents[1] / "models" / "feature_columns.json").read_text())
    if isinstance(feats, dict):
        feats = feats.get("feature_columns", feats.get("columns"))
    batch_all = pd.read_parquet(proc / "candidates_student_test.parquet")

    all_ids = sorted(p.name for p in root.iterdir() if p.is_dir())
    ids = args.trips or (all_ids if args.all else all_ids[:3])
    failed = 0
    print(f"radius={radius} m, cap={cap}, features={len(feats)}, trips={len(ids)}\n")
    for tid in ids:
        probs, n_rows, n_obs, n_nocand = check_trip(root / tid, batch_all, feats, radius, cap)
        status = "PASS" if not probs else "FAIL"
        print(f"[{status}] {tid}: {n_obs} obs, {n_rows} candidate rows, {n_nocand} obs without candidates")
        for p in probs:
            print("        -", p)
        failed += bool(probs)
    print(f"\n{len(ids) - failed}/{len(ids)} trips passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
