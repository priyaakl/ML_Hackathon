"""Steps 1.4 + 1.5: candidate generation and feature engineering.

For every GPS observation we keep ALL road arcs within `radius_m` metres (nearest first, at most `max_candidates`).
Each (observation, candidate arc) pair is one row, described by features. Persons 2/3/4 consume this table.

Run from the repo root:   python -m src.data.candidates
Writes: data/processed/candidates_<partition>.parquet   (one row per observation x candidate arc)
        data/processed/observations_<partition>.parquet (one row per observation, incl. those with no candidate)
        models/feature_columns.json, results/candidate_stats.csv, data/processed/no_candidate_obs.csv

The function build_candidates_for_trip() works from raw files only (no labels), so the API can call it live.
"""
import json
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from tqdm import tqdm

from src.common.config import load_config
from src.common.io import load_trip
from src.common.geo import trip_origin, project

WINDOW_S = 5.0          # motion window: GPS points within +-5 s of the observation define heading/speed
MIN_MOVE_M = 8.0        # below this displacement across the window the vehicle is "not moving": heading unreliable

# Order matters: this list is what models/feature_columns.json stores and what the classifier is trained on.
FEATURE_COLUMNS = [
    "dist_to_arc_m",      # distance from the GPS point to the arc (metres)
    "dist_minus_best_m",  # this distance minus the nearest candidate's distance (0 for the nearest)
    "dist_rank",          # 1 = nearest candidate
    "n_cands",            # how many candidates this observation has
    "arc_len_m",          # arc length
    "offset_along_m",     # distance from arc start to the point's projection on the arc
    "dist_to_arc_end_m",  # distance from the projection to the arc end
    "arc_has_reverse",    # 1 if the opposite-direction twin arc exists (two-way street)
    "arc_in_deg",         # arcs entering the start node
    "arc_out_deg",        # arcs leaving the end node
    "heading_valid",      # 1 if the vehicle was moving enough to trust its heading
    "heading_diff_deg",   # angle between vehicle heading and arc direction (0 = same, 180 = opposite); 90 if invalid
    "cos_heading",        # cosine of that angle (+1 aligned, -1 opposite); 0 if invalid
    "speed_mps",          # vehicle speed across the +-5 s window
    "progress_m",         # movement along the arc direction across the window (negative = against the arc)
    "progress_ratio",     # progress_m / straight-line movement (about +1 along the arc, -1 against it)
    "dist_prev_m",        # distance of the window's first point to this arc
    "dist_next_m",        # distance of the window's last point to this arc
    "dt_prev_s",          # time since previous GPS point
    "dt_next_s",          # time until next GPS point
]


def _bearing_deg(dx, dy):
    """Compass-style bearing: 0 = north, 90 = east."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def _reverse_arc_ids(u, v, n_nodes):
    key = u.astype(np.int64) * n_nodes + v
    rkey = v.astype(np.int64) * n_nodes + u
    order = np.argsort(key, kind="stable")
    sk = key[order]
    pos = np.clip(np.searchsorted(sk, rkey), 0, len(sk) - 1)
    return np.where(sk[pos] == rkey, order[pos], -1)


def build_candidates(trip, radius_m, max_candidates):
    """Returns (cands, obs) DataFrames for one trip. No labels are used."""
    tr, nd, ar = trip["track"], trip["nodes"], trip["arcs"]
    n = len(tr)
    lon0, lat0 = trip_origin(tr)
    P = project(tr["lon"], tr["lat"], lon0, lat0)
    N = project(nd["lon"], nd["lat"], lon0, lat0)
    t = tr["t"].to_numpy(dtype=np.float64)
    u, v = ar["u"].to_numpy(), ar["v"].to_numpy()
    A, B = N[u], N[v]
    AB = B - A
    L2 = (AB ** 2).sum(1)
    L = np.sqrt(L2)
    L2s = np.where(L2 == 0, 1.0, L2)
    rev = _reverse_arc_ids(u, v, len(nd))
    in_deg = np.bincount(v, minlength=len(nd))      # arcs ending at each node
    out_deg = np.bincount(u, minlength=len(nd))     # arcs starting at each node

    # ---- motion window per observation (heading, speed, window end points)
    i = np.arange(n)
    i0 = np.searchsorted(t, t - WINDOW_S, side="left")
    i1 = np.searchsorted(t, t + WINDOW_S, side="right") - 1
    disp = P[i1] - P[i0]
    move = np.linalg.norm(disp, axis=1)
    span = t[i1] - t[i0]
    heading_valid = (move >= MIN_MOVE_M) & (span > 0)
    heading = np.where(heading_valid, _bearing_deg(disp[:, 0], disp[:, 1]), np.nan)
    speed = np.where(span > 0, move / np.where(span > 0, span, 1), 0.0)
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]

    # ---- candidates: all arcs within radius of each point
    lines = shapely.linestrings(np.stack([A, B], axis=1))
    qi, ai = STRtree(lines).query(shapely.points(P), predicate="dwithin", distance=radius_m)

    def proj(Q):  # unclamped projection parameter of points Q onto the candidate arcs
        return ((Q - A[ai]) * AB[ai]).sum(1) / L2s[ai]

    def seg_dist(Q, tu):
        tc = np.clip(tu, 0.0, 1.0)
        return np.linalg.norm(Q - (A[ai] + tc[:, None] * AB[ai]), axis=1)

    Pq = P[qi]
    tu = proj(Pq)
    tc = np.clip(tu, 0.0, 1.0)
    dist = seg_dist(Pq, tu)

    order = np.lexsort((ai, dist, qi))                  # by observation, then nearest first
    qi, ai, tu, tc, dist = qi[order], ai[order], tu[order], tc[order], dist[order]
    start = np.r_[0, np.flatnonzero(np.diff(qi)) + 1]
    group_size = np.diff(np.r_[start, len(qi)])
    rank0 = np.arange(len(qi)) - np.repeat(start, group_size)
    keep = rank0 < max_candidates
    qi, ai, tu, tc, dist, rank0 = qi[keep], ai[keep], tu[keep], tc[keep], dist[keep], rank0[keep]
    start = np.r_[0, np.flatnonzero(np.diff(qi)) + 1]
    group_size = np.diff(np.r_[start, len(qi)])
    best = np.repeat(dist[start], group_size)
    n_cands = np.repeat(group_size, group_size)

    # window end points relative to each candidate arc
    P0, P1 = P[i0[qi]], P[i1[qi]]
    tu0, tu1 = proj(P0), proj(P1)
    d0, d1 = seg_dist(P0, tu0), seg_dist(P1, tu1)
    progress = (tu1 - tu0) * L[ai]
    hv = heading_valid[qi]
    arc_bearing = _bearing_deg(AB[ai, 0], AB[ai, 1])
    diff = np.abs((heading[qi] - arc_bearing + 180.0) % 360.0 - 180.0)
    diff = np.where(hv, diff, 90.0)
    cos_h = np.where(hv, np.cos(np.radians(diff)), 0.0)
    ratio = np.where(hv, progress / np.maximum(move[qi], 1.0), 0.0)

    f32 = np.float32
    cands = pd.DataFrame({
        "trip_id": trip["trip_id"],
        "obs_idx": qi.astype(np.int32),
        "t": t[qi].astype(f32),
        "lon": tr["lon"].to_numpy()[qi], "lat": tr["lat"].to_numpy()[qi],
        "x": P[qi, 0].astype(f32), "y": P[qi, 1].astype(f32),
        "cand_arc_id": ai.astype(np.int64),
        "cand_rank": (rank0 + 1).astype(np.int16),
        "arc_reverse_id": rev[ai].astype(np.int64),
        "dist_to_arc_m": dist.astype(f32),
        "dist_minus_best_m": (dist - best).astype(f32),
        "dist_rank": (rank0 + 1).astype(np.int16),
        "n_cands": n_cands.astype(np.int16),
        "arc_len_m": L[ai].astype(f32),
        "offset_along_m": (tc * L[ai]).astype(f32),
        "dist_to_arc_end_m": ((1 - tc) * L[ai]).astype(f32),
        "arc_has_reverse": (rev[ai] >= 0).astype(np.int8),
        "arc_in_deg": in_deg[u[ai]].astype(np.int16),
        "arc_out_deg": out_deg[v[ai]].astype(np.int16),
        "heading_valid": hv.astype(np.int8),
        "heading_diff_deg": diff.astype(f32),
        "cos_heading": cos_h.astype(f32),
        "speed_mps": speed[qi].astype(f32),
        "progress_m": progress.astype(f32),
        "progress_ratio": ratio.astype(f32),
        "dist_prev_m": d0.astype(f32),
        "dist_next_m": d1.astype(f32),
        "dt_prev_s": dt_prev[qi].astype(f32),
        "dt_next_s": dt_next[qi].astype(f32),
    })
    cnt = np.bincount(qi, minlength=n)
    obs = pd.DataFrame({
        "trip_id": trip["trip_id"], "obs_idx": i.astype(np.int32), "t": t.astype(f32),
        "lon": tr["lon"].to_numpy(), "lat": tr["lat"].to_numpy(),
        "x": P[:, 0].astype(f32), "y": P[:, 1].astype(f32),
        "n_cands": cnt.astype(np.int16), "speed_mps": speed.astype(f32),
        "heading_deg": heading.astype(f32),
    })
    return cands, obs


def build_candidates_for_trip(trip_dir, radius_m=40, max_candidates=20):
    """Raw files in, (cands, obs) out. No label file is read, so the API can use this for unseen trips."""
    return build_candidates(load_trip(trip_dir, with_route=False), radius_m, max_candidates)


def main():
    cfg = load_config()
    root, out = cfg["paths"]["data_root"], cfg["paths"]["processed"]
    rad, cap = cfg["candidates"]["radius_m"], cfg["candidates"]["max_candidates"]
    (cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)
    models_dir = out.parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    with open(out / "splits.json") as f:
        splits = json.load(f)

    stats, no_cand = [], []
    for part in ["train", "student_test"]:
        labels = pd.read_parquet(out / f"labels_{part}.parquet")
        cl, ol = [], []
        for tid in tqdm(splits[part], desc=f"candidates [{part}]"):
            c, o = build_candidates(load_trip(root / part / tid, with_route=False), rad, cap)
            cl.append(c)
            ol.append(o)
        cands, obs = pd.concat(cl, ignore_index=True), pd.concat(ol, ignore_index=True)

        # attach labels (training / development data only)
        obs = obs.merge(labels[["trip_id", "obs_idx", "ref_arc_id", "label_valid"]], on=["trip_id", "obs_idx"], how="left")
        cands = cands.merge(obs[["trip_id", "obs_idx", "ref_arc_id", "label_valid"]], on=["trip_id", "obs_idx"], how="left")
        cands["label"] = (cands["cand_arc_id"] == cands["ref_arc_id"]).astype(np.int8)
        hit = cands.groupby(["trip_id", "obs_idx"])["label"].max().rename("label_in_cands").reset_index()
        obs = obs.merge(hit, on=["trip_id", "obs_idx"], how="left")
        obs["label_in_cands"] = obs["label_in_cands"].fillna(0).astype(bool)
        cands = cands.drop(columns=["ref_arc_id"])

        cands.to_parquet(out / f"candidates_{part}.parquet", index=False)
        obs.to_parquet(out / f"observations_{part}.parquet", index=False)
        nc = obs[obs.n_cands == 0][["trip_id", "obs_idx"]].assign(partition=part)
        no_cand.append(nc)

        v = obs[obs.label_valid.fillna(False)]
        nearest = cands[cands.cand_rank == 1].merge(obs[["trip_id", "obs_idx", "ref_arc_id"]], on=["trip_id", "obs_idx"])
        nearest = nearest[nearest.label_valid.fillna(False)]
        stats.append(dict(
            partition=part, trips=len(splits[part]), observations=len(obs), candidate_rows=len(cands),
            avg_candidates_per_obs=round(len(cands) / len(obs), 2), obs_without_candidates=int((obs.n_cands == 0).sum()),
            valid_label_obs=len(v),
            candidate_miss_rate_pct=round(float((~v.label_in_cands).mean()) * 100, 3),
            nearest_arc_top1_pct=round(float((nearest.cand_arc_id == nearest.ref_arc_id).mean()) * 100, 2),
            positive_rate_pct=round(float(cands[cands.label_valid.fillna(False)].label.mean()) * 100, 2),
        ))
    pd.concat(no_cand).to_csv(out / "no_candidate_obs.csv", index=False)
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    st = pd.DataFrame(stats)
    st.to_csv(cfg["paths"]["results"] / "candidate_stats.csv", index=False)
    print("\n", st.T.to_string(header=False))
    print(f"\n(radius {rad} m, cap {cap}). Saved candidates_*.parquet, observations_*.parquet, models/feature_columns.json")


if __name__ == "__main__":
    main()
