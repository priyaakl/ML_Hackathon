"""Task 1.5 evidence: numbers that back the feature rationale (results/feature_rationale.md).

Run from the repo root (after python -m src.data.candidates):
    python -m src.data.feature_stats

Writes:
    results/feature_stats.csv       per partition and feature: single-feature AUC, mean for true arc vs other candidates
    results/feature_twin_stats.csv  per partition: how well direction features separate a road from its opposite twin

Only rows with label_valid == True are used. These are one-feature-at-a-time views, NOT model feature importance.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.common.config import load_config

PARTS = ["train", "student_test"]


def single_feature_auc(df, feats):
    """AUC of each feature alone for 'true arc vs other candidate'. Reported as max(auc, 1-auc), plus the sign."""
    y = df["label"].to_numpy() == 1
    n1, n0 = int(y.sum()), int((~y).sum())
    rows = []
    for f in feats:
        x = df[f].to_numpy(dtype=np.float64)
        r = rankdata(x)                                   # average ranks handle ties
        auc = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        rows.append(dict(feature=f, auc_raw=auc, auc_strength=max(auc, 1 - auc),
                         higher_for_true_arc=bool(auc >= 0.5),
                         mean_true=float(x[y].mean()), mean_other=float(x[~y].mean())))
    return pd.DataFrame(rows)


def twin_separation(df):
    """Among valid points where the true arc AND its opposite-direction twin are both candidates:
    how often does the true arc look 'more aligned' than the twin? Uses heading-valid points only."""
    pos = df[df["label"] == 1]
    cols = ["trip_id", "obs_idx", "cand_arc_id", "dist_to_arc_m", "cos_heading", "progress_ratio", "heading_valid"]
    pos = pos[cols + ["arc_reverse_id"]]
    twin = df[cols].rename(columns={"cand_arc_id": "arc_reverse_id", "dist_to_arc_m": "tw_dist",
                                    "cos_heading": "tw_cos", "progress_ratio": "tw_ratio"})[
        ["trip_id", "obs_idx", "arc_reverse_id", "tw_dist", "tw_cos", "tw_ratio"]]
    m = pos.merge(twin, on=["trip_id", "obs_idx", "arc_reverse_id"], how="inner")
    h = m[m["heading_valid"] == 1]
    return dict(points_true_and_twin_both_candidates=len(m),
                median_abs_dist_diff_m=float((m.dist_to_arc_m - m.tw_dist).abs().median()),
                points_with_valid_heading=len(h),
                pct_true_higher_cos_heading=round(float((h.cos_heading > h.tw_cos).mean()) * 100, 2),
                pct_true_higher_progress_ratio=round(float((h.progress_ratio > h.tw_ratio).mean()) * 100, 2),
                pct_true_higher_on_both=round(float(((h.cos_heading > h.tw_cos) & (h.progress_ratio > h.tw_ratio)).mean()) * 100, 2))


def main():
    cfg = load_config()
    proc, res = cfg["paths"]["processed"], cfg["paths"]["results"]
    feats = json.loads((proc.parents[1] / "models" / "feature_columns.json").read_text())
    auc_rows, twin_rows = [], []
    for p in PARTS:
        df = pd.read_parquet(proc / f"candidates_{p}.parquet")
        df = df[df["label_valid"].fillna(False).astype(bool)]
        a = single_feature_auc(df, feats)
        a.insert(0, "partition", p)
        a["heading_valid_share_pct"] = round(float(df["heading_valid"].mean()) * 100, 2)
        auc_rows.append(a)
        twin_rows.append(dict(partition=p, **twin_separation(df)))
        print(p, len(df), "valid rows")
    auc = pd.concat(auc_rows, ignore_index=True)
    twin = pd.DataFrame(twin_rows)
    auc.to_csv(res / "feature_stats.csv", index=False)
    twin.to_csv(res / "feature_twin_stats.csv", index=False)
    print("\nSingle-feature AUC (strength = max(AUC, 1-AUC)), train:")
    print(auc[auc.partition == "train"].sort_values("auc_strength", ascending=False)[["feature", "auc_strength"]].round(3).to_string(index=False))
    print("\nTwin separation:\n", twin.T.to_string(header=False))


if __name__ == "__main__":
    main()
