"""Step 5: how do the candidate radius and the cap change the table size and the miss rate?

Person 2 needs a measured answer to "how big is the table / is KNN feasible", and the report needs a
defensible choice of radius and cap. This script builds candidates ONCE with a generous setting
(radius 60 m, cap 40) and derives every smaller setting from it. That is exact, because candidates are
ranked by distance: the candidates for (radius r, cap c) are the rows with dist <= r and rank <= c.

Run from the repo root (needs labels_*.parquet and splits.json):
    python -m src.data.cap_study

Writes (results/):
    candidate_cap_study.csv       one row per (partition, radius, cap): table size, miss rate, memory
    candidate_miss_by_trip.csv    per trip at the current setting (40 m, cap 20), with the CAUSE of each miss
    figures/fig_cap_study.png     miss rate vs table size
Nothing is written to data/processed, so nothing here changes the shared candidate table.

"Miss" = a point with label_valid whose reference arc is not among its candidates. (100 - miss) is the
ceiling for Top-1 accuracy of ANY classifier on that table.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.common.config import load_config
from src.common.io import load_trip
from src.data.candidates import build_candidates, FEATURE_COLUMNS

BUILD_RADIUS, BUILD_CAP = 60, 40              # generous build; smaller settings are filtered from it
RADII = [20, 30, 40, 50, 60]
CAPS = [3, 5, 8, 10, 15, 20, 40]
CUR_R, CUR_C = 40, 20                         # the setting currently in config.yaml
PARTS = ["train", "student_test"]
N_FEATURES = len(FEATURE_COLUMNS)


def per_obs_table(trip, lab):
    """One row per GPS point: label info, distance and rank of its true arc, and the number of
    candidates within each radius. All candidate rows are discarded afterwards (memory)."""
    cands, obs = build_candidates(trip, BUILD_RADIUS, BUILD_CAP)
    n = len(obs)
    lab = lab.sort_values("obs_idx")
    assert len(lab) == n, "labels and track differ in length: re-run python -m src.data.labels"
    ref = lab["ref_arc_id"].to_numpy()
    qi = cands["obs_idx"].to_numpy()
    dist = cands["dist_to_arc_m"].to_numpy()
    hit = cands["cand_arc_id"].to_numpy() == ref[qi]
    true_dist = np.full(n, np.inf)
    true_rank = np.full(n, np.inf)
    true_dist[qi[hit]] = dist[hit]
    true_rank[qi[hit]] = cands["cand_rank"].to_numpy()[hit]
    out = pd.DataFrame({
        "trip_id": trip["trip_id"], "obs_idx": np.arange(n),
        "valid": lab["label_valid"].to_numpy(dtype=bool),
        "label_dist_m": lab["label_dist_m"].to_numpy(),
        "true_dist": true_dist, "true_rank": true_rank,
    })
    for r in RADII:
        out[f"cnt_{r}"] = np.bincount(qi[dist <= r], minlength=n)
    return out


def summarise(po, part):
    """Table-size and miss-rate numbers for every (radius, cap)."""
    rows = []
    valid = po["valid"].to_numpy()
    for r in RADII:
        cnt = po[f"cnt_{r}"].to_numpy()
        for c in CAPS:
            kept = np.minimum(cnt, c)                                   # candidates per point
            found = (po["true_dist"].to_numpy() <= r) & (po["true_rank"].to_numpy() <= c)
            miss = float((valid & ~found).sum() / valid.sum() * 100)
            rows_all = int(kept.sum())
            rows_valid = int(kept[valid].sum())                         # the rows a model trains/evaluates on
            rows.append(dict(
                partition=part, radius_m=r, cap=c, rows_all=rows_all, rows_valid_only=rows_valid,
                avg_cands_per_point=round(rows_all / len(po), 2),
                points_without_candidates=int((kept == 0).sum()),
                miss_pct=round(miss, 3), top1_ceiling_pct=round(100 - miss, 3),
                valid_table_float32_MB=round(rows_valid * N_FEATURES * 4 / 1e6, 1),
            ))
    return rows


def miss_by_trip(po):
    """Per trip at (CUR_R, CUR_C): miss rate and WHY the true arc was missed."""
    po = po[po["valid"]]
    out = []
    for tid, g in po.groupby("trip_id"):
        td, tr = g["true_dist"].to_numpy(), g["true_rank"].to_numpy()
        found = (td <= CUR_R) & (tr <= CUR_C)
        too_far = ~found & (td > CUR_R)                                  # true arc is farther than the radius
        beyond_cap = ~found & (td <= CUR_R) & (tr > CUR_C)               # within radius but ranked below the cap
        out.append(dict(
            trip_id=tid, valid_points=len(g), missed=int((~found).sum()),
            miss_pct=round(float((~found).mean() * 100), 2),
            missed_true_arc_beyond_radius=int(too_far.sum()),
            missed_beyond_cap=int(beyond_cap.sum()),
            median_label_dist_of_misses_m=round(float(g.loc[~found, "label_dist_m"].median()), 1) if (~found).any() else np.nan,
        ))
    return pd.DataFrame(out).sort_values("miss_pct", ascending=False)


def figure(study, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for a, part in zip(ax, PARTS):
        s = study[study.partition == part]
        for r in RADII:
            q = s[s.radius_m == r].sort_values("cap")
            a.plot(q.rows_valid_only / 1e6, np.maximum(q.miss_pct, 0.005), marker="o", label=f"radius {r} m")
        cur = s[(s.radius_m == CUR_R) & (s.cap == CUR_C)].iloc[0]
        a.scatter([cur.rows_valid_only / 1e6], [cur.miss_pct], s=160, facecolors="none", edgecolors="k",
                  linewidths=2, zorder=5, label="current (40 m, cap 20)")
        a.set_yscale("log")
        a.set(xlabel="rows in the valid-label table (millions)", title=f"{part}: points of each line = caps {CAPS}")
    ax[0].set_ylabel("candidate miss rate, % of valid points (log; 0 drawn as 0.005)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    root, proc, res = cfg["paths"]["data_root"], cfg["paths"]["processed"], cfg["paths"]["results"]
    (res / "figures").mkdir(parents=True, exist_ok=True)
    with open(proc / "splits.json") as f:
        splits = json.load(f)

    study, by_trip = [], []
    for part in PARTS:
        labels = pd.read_parquet(proc / f"labels_{part}.parquet")
        frames = []
        for tid in tqdm(splits[part], desc=f"cap study [{part}]"):
            trip = load_trip(root / part / tid, with_route=False)
            frames.append(per_obs_table(trip, labels[labels.trip_id == tid]))
        po = pd.concat(frames, ignore_index=True)
        study += summarise(po, part)
        bt = miss_by_trip(po)
        bt.insert(0, "partition", part)
        by_trip.append(bt)

    study = pd.DataFrame(study)
    bt = pd.concat(by_trip, ignore_index=True)
    study.to_csv(res / "candidate_cap_study.csv", index=False)
    bt.to_csv(res / "candidate_miss_by_trip.csv", index=False)
    figure(study, res / "figures" / "fig_cap_study.png")

    show = ["radius_m", "cap", "rows_valid_only", "avg_cands_per_point", "points_without_candidates",
            "miss_pct", "valid_table_float32_MB"]
    for part in PARTS:
        s = study[study.partition == part]
        print(f"\n=== {part}: miss rate / table size (rows_valid_only = rows a model trains on) ===")
        print(s[s.radius_m.isin([30, 40, 50]) & s.cap.isin([5, 8, 10, 20])][show].to_string(index=False))
    cur = study[(study.partition == "train") & (study.radius_m == CUR_R) & (study.cap == CUR_C)].iloc[0]
    print(f"\nCurrent setting check (train, {CUR_R} m, cap {CUR_C}): rows_all={cur.rows_all}, "
          f"avg={cur.avg_cands_per_point}, miss={cur.miss_pct}%  (candidate_stats.csv says 1359888 rows, 0.64%)")
    print("\nTrips with the most misses at the current setting:")
    print(bt.head(8).to_string(index=False))
    print("\nSaved results/candidate_cap_study.csv, candidate_miss_by_trip.csv, figures/fig_cap_study.png")


if __name__ == "__main__":
    main()
