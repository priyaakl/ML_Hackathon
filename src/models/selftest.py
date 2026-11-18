"""Person 2: self-test on FAKE data. Run this first to prove the harness works before real data is involved.

  python -m src.models.selftest

It writes small fake candidates_/observations_ parquet files into a temp folder (not your real data/processed),
runs two models through the full pipeline, saves a model, and reloads it through CandidateScorer.
Expect: "SELFTEST PASSED". The accuracy numbers are meaningless (fake data).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.harness import load_feature_columns


def fake_partition(n_trips, seed, feats, start=1):
    rng = np.random.default_rng(seed)
    cands, obs = [], []
    for ti in range(n_trips):
        tid = f"{start + ti:08d}"
        valid = ti != 2                                     # one trip with invalid labels, like the real data
        for o in range(80):
            k = int(rng.integers(0, 7))                     # some points get zero candidates
            truth = int(rng.integers(0, max(k, 1)))
            obs.append(dict(trip_id=tid, obs_idx=o, n_cands=k, label_valid=valid))
            for r in range(k):
                row = {f: float(rng.normal()) for f in feats}
                is_true = r == truth
                row["dist_to_arc_m"] = abs(rng.normal(3 if is_true else 15, 4))   # signal: true arc is nearer
                row.update(trip_id=tid, obs_idx=o, cand_arc_id=1000 * ti + r, cand_rank=r + 1,
                           label=int(is_true), label_valid=valid)
                cands.append(row)
    return pd.DataFrame(cands), pd.DataFrame(obs)


def main():
    feats = load_feature_columns()
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for part, n, seed, start in [("train", 12, 0, 1), ("student_test", 4, 1, 100)]:
            c, o = fake_partition(n, seed, feats, start)
            c.to_parquet(tmp / f"candidates_{part}.parquet", index=False)
            o.to_parquet(tmp / f"observations_{part}.parquet", index=False)
        cmd = [sys.executable, "-m", "src.models.run_experiments", "--models", "lr", "lgbm", "mlp",
               "--folds", "3", "--processed", str(tmp), "--results", str(tmp / "res")]
        subprocess.run(cmd, check=True, cwd=root)
        log = pd.read_csv(tmp / "res" / "experiments.csv")
        assert len(log) == 3 and log["st_top1_all"].between(0, 1).all(), "experiment log looks wrong"
        assert (log["st_top1_all"] <= 1 - log["st_miss_rate"] + 1e-9).all(), "top1 exceeds the candidate ceiling"
        subprocess.run([sys.executable, "-m", "src.models.train_final", "--model", "lr", "--processed", str(tmp),
                        "--out", str(tmp / "m")], check=True, cwd=root)
        from src.models.scorer import CandidateScorer
        sc = CandidateScorer(str(tmp / "m" / "final_model.joblib"), str(root / "models" / "feature_columns.json"))
        df = pd.read_parquet(tmp / "candidates_student_test.parquet")
        p = sc.predict_proba(df)
        assert p.shape == (len(df),) and ((p >= 0) & (p <= 1)).all()
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
