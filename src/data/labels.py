"""Step 1.3: per-GPS-point reference labels.

The .route file lists the arcs the vehicle travelled, in order, but it does NOT say which
GPS point is on which arc. We recover that by monotone alignment:

  * Compute the distance from every GPS point to every route arc (in metres).
  * Choose one route position per GPS point so that the position never goes backwards as time
    advances, and the total point-to-arc distance is as small as possible (dynamic programming).

Because the supplied reference route is not perfect (the guidelines warn about this), each label also gets
`label_valid` = the point lies within VALID_DIST_M of its assigned reference arc. Points farther away are kept
in the table (the decoder still needs every GPS point) but must NOT be used as training/evaluation labels.

Run from the repo root:   python -m src.data.labels
Writes: data/processed/labels_<partition>.parquet, results/label_quality.csv
"""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.config import load_config
from src.common.io import load_trip
from src.common.geo import trip_origin, project

VALID_DIST_M = 50.0        # label counts as reliable only if the GPS point is within 50 m of its reference arc
BAD_TRIP_FRAC = 0.25       # trip flagged "unreliable reference" if more than 25% of its points are invalid


def point_segment_dist(P, A, B):
    """Distance (m) from each point in P (n,2) to ONE segment A-B, vectorised over points."""
    AB = B - A
    L2 = float(AB @ AB)
    if L2 == 0.0:
        return np.linalg.norm(P - A, axis=1)
    t = np.clip(((P - A) @ AB) / L2, 0.0, 1.0)
    C = A + t[:, None] * AB
    return np.linalg.norm(P - C, axis=1)


def monotone_align(D):
    """D: (n_obs, m_route) distance matrix. Returns route position per observation (non-decreasing)
    minimising sum of D[i, pos[i]].  f[i, j] = D[i, j] + min_{k <= j} f[i-1, k]."""
    n, m = D.shape
    back = np.zeros((n, m), dtype=np.int16 if m < 32000 else np.int32)
    ar = np.arange(m)
    f = D[0].astype(np.float64)
    for i in range(1, n):
        cm = np.minimum.accumulate(f)                         # running minimum of previous row
        back[i] = np.maximum.accumulate(np.where(f <= cm, ar, 0))  # where that running minimum came from
        f = D[i] + cm
    pos = np.empty(n, dtype=np.int64)
    j = int(np.argmin(f))
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = int(back[i, j])
    return pos


def label_trip(trip):
    """Returns DataFrame: trip_id, obs_idx, route_pos, ref_arc_id, label_dist_m."""
    tr, nd, ar, rt = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    lon0, lat0 = trip_origin(tr)
    P = project(tr["lon"], tr["lat"], lon0, lat0)
    N = project(nd["lon"], nd["lat"], lon0, lat0)
    A = N[ar["u"].to_numpy()[rt]]          # start point of each route arc
    B = N[ar["v"].to_numpy()[rt]]          # end point of each route arc
    D = np.empty((len(P), len(rt)), dtype=np.float32)
    for j in range(len(rt)):
        D[:, j] = point_segment_dist(P, A[j], B[j])
    pos = monotone_align(D)
    return pd.DataFrame({
        "trip_id": trip["trip_id"],
        "obs_idx": np.arange(len(P), dtype=np.int32),
        "route_pos": pos.astype(np.int32),
        "ref_arc_id": rt[pos].astype(np.int64),
        "label_dist_m": D[np.arange(len(P)), pos],
    }).assign(label_valid=lambda df: df["label_dist_m"] <= VALID_DIST_M)


def main():
    cfg = load_config()
    root, out = cfg["paths"]["data_root"], cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["results"].mkdir(parents=True, exist_ok=True)
    with open(out / "splits.json") as f:
        splits = json.load(f)

    quality = []
    for part in ["train", "student_test"]:
        frames = []
        for tid in tqdm(splits[part], desc=f"labels [{part}]"):
            trip = load_trip(root / part / tid)
            lab = label_trip(trip)
            frames.append(lab)
            d = lab["label_dist_m"].to_numpy()
            quality.append(dict(
                partition=part, trip_id=tid, n_obs=len(lab), n_route_arcs=len(trip["route"]),
                route_arcs_used=int(lab["route_pos"].nunique()),
                dist_median_m=round(float(np.median(d)), 2), dist_p99_m=round(float(np.percentile(d, 99)), 2),
                dist_max_m=round(float(d.max()), 2), n_over_30m=int((d > 30).sum()),
                n_invalid=int((~lab["label_valid"]).sum()),
                pct_invalid=round(float((~lab["label_valid"]).mean()) * 100, 2),
            ))
        pd.concat(frames, ignore_index=True).to_parquet(out / f"labels_{part}.parquet", index=False)

    q = pd.DataFrame(quality)
    q.to_csv(cfg["paths"]["results"] / "label_quality.csv", index=False)
    print("\nLabel distance (GPS point to its assigned reference arc), metres:")
    print(q.groupby("partition")[["dist_median_m", "dist_p99_m", "dist_max_m"]].agg(["median", "max"]).round(1))
    print("Observations farther than 30 m from their assigned arc:",
          int(q.n_over_30m.sum()), "of", int(q.n_obs.sum()))
    bad = q[q.pct_invalid > BAD_TRIP_FRAC * 100]
    print(f"\nValid labels (<= {VALID_DIST_M:.0f} m): {int(q.n_obs.sum() - q.n_invalid.sum())} of {int(q.n_obs.sum())} "
          f"({100 - q.n_invalid.sum() / q.n_obs.sum() * 100:.2f}%)")
    print(f"Trips flagged unreliable reference (> {BAD_TRIP_FRAC*100:.0f}% invalid):",
          bad[["partition", "trip_id", "n_obs", "pct_invalid"]].to_dict("records"))
    print("student_test: max % invalid in any trip =", q[q.partition == "student_test"].pct_invalid.max())
    print("Saved: data/processed/labels_train.parquet, labels_student_test.parquet, results/label_quality.csv")


if __name__ == "__main__":
    main()
