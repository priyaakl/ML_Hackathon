"""Step 4a: exploratory data analysis (EDA) figures and tables for the report.

Run from the repo root (after labels.py and candidates.py have been run):
    python -m src.data.eda

Writes into results/figures/ (PNG) and results/ (CSV):
    eda_summary.csv        one row per partition: counts, missing values, gaps, route jumps, network facts
    eda_network.csv        one row per trip: nodes, arcs, two-way share, connected components
    eda_gps_jumps_isolated.csv  one row per trip: GPS speed jumps and isolated road-network nodes
    fig_trip_lengths.png   GPS points per trip
    fig_time_gaps.png      distribution of time between consecutive GPS points + trips with long gaps
    fig_speed.png          vehicle speed distribution (from observations_*.parquet)
    fig_label_distance.png GPS-to-reference-arc distance and % invalid labels per trip
    fig_candidates.png     candidates per GPS point and the candidate miss rate per trip

Figures that need parquet files skip themselves with a message if the files are missing.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from src.common.config import load_config
from src.common.io import load_trip
from src.common.geo import trip_origin, project

PARTS = ["train", "student_test"]
COLORS = {"train": "#215cb5", "student_test": "#e08a1e"}
BAD_TRIP_PCT = 25.0                       # same rule as labels.py
JUMP_SPEED_MPS = 50.0                     # a step implying > 50 m/s (180 km/h) counts as a GPS speed jump (our choice, not tuned)
JUMP_MAX_DT_S = 60.0                      # only steps up to 60 s apart are judged; longer ones are time gaps, handled separately
BIG_STEP_M, BIG_STEP_DT_S = 200.0, 5.0    # "teleport" check: a step over 200 m within 5 s


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  saved", path)


def network_stats(root, splits):
    """One row per trip: size of the road network and how connected it is."""
    rows = []
    jobs = [(p, t) for p in PARTS for t in splits[p]]
    for part, tid in tqdm(jobs, desc="network stats"):
        d = load_trip(root / part / tid, with_route=False)
        nd, ar = d["nodes"], d["arcs"]
        n, u, v = len(nd), ar["u"].to_numpy(), ar["v"].to_numpy()
        key = set((u.astype(np.int64) * n + v).tolist())
        two_way = np.fromiter(((int(b) * n + int(a)) in key for a, b in zip(u, v)), dtype=bool, count=len(u))
        g = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
        n_weak, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab)
        rows.append(dict(
            partition=part, trip_id=tid, n_nodes=n, n_arcs=len(ar),
            pct_arcs_two_way=round(float(two_way.mean()) * 100, 1),
            n_components=int(n_weak), largest_component_pct=round(float(sizes.max() / n) * 100, 1),
            missing_values=int(d["track"].isna().sum().sum() + nd.isna().sum().sum() + ar.isna().sum().sum()),
        ))
    return pd.DataFrame(rows)


def track_quality(root, splits):
    """One row per trip: GPS speed jumps (from the projected track) and isolated road-network nodes
    (nodes that no arc starts or ends at). Isolated nodes are different from disconnected components."""
    rows = []
    jobs = [(p, t) for p in PARTS for t in splits[p]]
    for part, tid in tqdm(jobs, desc="gps jumps / isolated nodes"):
        d = load_trip(root / part / tid, with_route=False)
        tr = d["track"]
        lon0, lat0 = trip_origin(tr)
        P = project(tr["lon"], tr["lat"], lon0, lat0)
        t = tr["t"].to_numpy(dtype=np.float64)
        step, dt = np.linalg.norm(np.diff(P, axis=0), axis=1), np.diff(t)
        ok = (dt > 0) & (dt <= JUMP_MAX_DT_S)
        speed = np.where(ok, step / np.where(dt > 0, dt, 1.0), 0.0)
        n_nodes = len(d["nodes"])
        used = np.zeros(n_nodes, dtype=bool)
        used[d["arcs"]["u"].to_numpy()] = True
        used[d["arcs"]["v"].to_numpy()] = True
        rows.append(dict(
            partition=part, trip_id=tid, n_steps=len(step),
            n_jump_gt50mps=int((speed > JUMP_SPEED_MPS).sum()),
            n_jump_gt200m_1s=int(((step > BIG_STEP_M) & (dt <= BIG_STEP_DT_S)).sum()),
            max_speed_mps=float(speed.max()) if len(speed) else 0.0,
            max_step_m=float(step[ok].max()) if ok.any() else 0.0,
            n_nodes=n_nodes, isolated_nodes=int((~used).sum()),
        ))
    return pd.DataFrame(rows)


def gap_table(root, splits):
    """All time steps (dt) of every trip, for the gap plots."""
    out = []
    for part in PARTS:
        for tid in splits[part]:
            t = load_trip(root / part / tid, with_route=False)["track"]["t"].to_numpy(dtype=np.float64)
            out.append(pd.DataFrame({"partition": part, "trip_id": tid, "dt": np.diff(t)}))
    return pd.concat(out, ignore_index=True)


def fig_trip_lengths(inv, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(np.log10(inv.n_obs.min()), np.log10(inv.n_obs.max()), 25)
    for p in PARTS:
        ax[0].hist(inv[inv.partition == p].n_obs, bins=bins, alpha=0.7, label=p, color=COLORS[p])
    ax[0].set_xscale("log")
    ax[0].set(xlabel="GPS points per trip (log scale)", ylabel="trips", title="Trip length")
    ax[0].legend()
    s = inv.sort_values("n_obs", ascending=False).head(10)
    ax[1].barh(s.trip_id + " (" + s.partition.str[:5] + ")", s.n_obs, color=[COLORS[p] for p in s.partition])
    ax[1].invert_yaxis()
    ax[1].set(xlabel="GPS points", title="10 longest trips")
    _save(fig, path)


def fig_time_gaps(dts, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    clipped = np.clip(dts.dt.to_numpy(), 0, 70)
    ax[0].hist(clipped, bins=np.arange(0, 72, 1), color="#215cb5")
    ax[0].set_yscale("log")
    ax[0].set(xlabel="seconds between consecutive GPS points (clipped at 70)", ylabel="count (log)",
              title="Time step distribution")
    g = dts[dts.dt > 60].groupby("trip_id").agg(n_gaps=("dt", "size"), longest_s=("dt", "max")).reset_index()
    g = g.sort_values("longest_s", ascending=False).head(15)
    ax[1].barh(g.trip_id, g.longest_s / 3600, color="#bf4728")
    ax[1].set_xscale("log")
    ax[1].invert_yaxis()
    ax[1].set(xlabel="longest gap (hours, log scale)", title="Trips with gaps > 60 s (top 15)")
    _save(fig, path)


def fig_speed(obs, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p, o in obs.items():
        ax.hist(np.clip(o.speed_mps * 3.6, 0, 150), bins=60, alpha=0.7, density=True, label=p, color=COLORS[p])
    ax.set(xlabel="speed (km/h, clipped at 150)", ylabel="density", title="Vehicle speed (10 s window)")
    ax.legend()
    _save(fig, path)


def fig_label_distance(labels, quality, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p, lab in labels.items():
        ax[0].hist(np.clip(lab.label_dist_m, 0, 100), bins=np.arange(0, 101, 2), alpha=0.7, density=True,
                   label=p, color=COLORS[p])
    ax[0].axvline(50, color="k", ls="--", lw=1)
    ax[0].text(51, 3e-5, "50 m = label_valid limit", fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set(xlabel="distance GPS point to its reference arc (m, clipped at 100)", ylabel="density (log)",
              title="Reference label quality")
    ax[0].legend(loc="center right")
    q = quality.sort_values("pct_invalid", ascending=False).head(12)
    cols = ["#bf4728" if x > BAD_TRIP_PCT else "#7b8998" for x in q.pct_invalid]
    ax[1].barh(q.trip_id, q.pct_invalid, color=cols)
    ax[1].invert_yaxis()
    ax[1].set(xlabel="% of GPS points with invalid label", title="Trips with most invalid labels (top 12)")
    _save(fig, path)


def fig_candidates(obs, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p, o in obs.items():
        ax[0].hist(np.clip(o.n_cands, 0, 20), bins=np.arange(-0.5, 21.5, 1), alpha=0.7, density=True,
                   label=p, color=COLORS[p])
    ax[0].set(xlabel="candidate arcs per GPS point (cap 20)", ylabel="share of points", title="Candidates per point")
    ax[0].legend()
    o = pd.concat(obs.values())
    v = o[o.label_valid.fillna(False).astype(bool)]
    miss = (1 - v.groupby("trip_id").label_in_cands.mean()) * 100
    miss = miss[miss > 0].sort_values(ascending=False).head(12)
    ax[1].barh(miss.index, miss.values, color="#bf4728")
    ax[1].invert_yaxis()
    ax[1].set(xlabel="% valid points whose true arc is NOT a candidate", title="Candidate miss rate (top 12 trips)")
    _save(fig, path)


def main():
    cfg = load_config()
    root, proc, res = cfg["paths"]["data_root"], cfg["paths"]["processed"], cfg["paths"]["results"]
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    with open(proc / "splits.json") as f:
        splits = json.load(f)

    inv = pd.read_csv(res / "dataset_inventory.csv", dtype={"trip_id": str})
    quality = pd.read_csv(res / "label_quality.csv", dtype={"trip_id": str})

    print("Network statistics ...")
    net = network_stats(root, splits)
    net.to_csv(res / "eda_network.csv", index=False)
    tq = track_quality(root, splits)
    tq.to_csv(res / "eda_gps_jumps_isolated.csv", index=False)
    print("Time gaps ...")
    dts = gap_table(root, splits)

    # ---- summary table (numbers to quote in the report)
    rows = []
    for p in PARTS:
        i, n, d = inv[inv.partition == p], net[net.partition == p], dts[dts.partition == p].dt
        q = tq[tq.partition == p]
        rows.append(dict(
            partition=p, trips=len(i), gps_points=int(i.n_obs.sum()), median_points_per_trip=float(i.n_obs.median()),
            missing_values=int(n.missing_values.sum()), non_monotonic_time_trips=int((~i.time_monotonic).sum()),
            pct_steps_exactly_1s=round(float((d == 1).mean()) * 100, 2), steps_gt_60s=int((d > 60).sum()),
            trips_with_gt_60s=int((i.n_gaps_gt_60s > 0).sum()), longest_gap_s=float(d.max()),
            trips_with_route_jumps=int((i.route_connected_frac < 1).sum()),
            min_route_connected_pct=round(float(i.route_connected_frac.min()) * 100, 2),
            median_pct_arcs_two_way=float(n.pct_arcs_two_way.median()),
            trips_with_disconnected_network=int((n.n_components > 1).sum()),
            median_largest_component_pct=float(n.largest_component_pct.median()),
            gps_speed_jumps_gt50mps=int(q.n_jump_gt50mps.sum()), trips_with_speed_jumps=int((q.n_jump_gt50mps > 0).sum()),
            steps_gt200m_within5s=int(q.n_jump_gt200m_1s.sum()), isolated_nodes=int(q.isolated_nodes.sum()),
        ))
    summ = pd.DataFrame(rows)
    summ.to_csv(res / "eda_summary.csv", index=False)
    print("\n", summ.set_index("partition").T.to_string(), "\n")

    print("Figures ...")
    fig_trip_lengths(inv, figs / "fig_trip_lengths.png")
    fig_time_gaps(dts, figs / "fig_time_gaps.png")

    labels = {p: pd.read_parquet(proc / f"labels_{p}.parquet") for p in PARTS if (proc / f"labels_{p}.parquet").exists()}
    if labels:
        fig_label_distance(labels, quality, figs / "fig_label_distance.png")
    else:
        print("  skipped label figure: run python -m src.data.labels first")

    obs_files = [proc / f"observations_{p}.parquet" for p in PARTS]
    if all(f.exists() for f in obs_files):
        obs = {p: pd.read_parquet(f) for p, f in zip(PARTS, obs_files)}
        fig_speed(obs, figs / "fig_speed.png")
        fig_candidates(obs, figs / "fig_candidates.png")
    else:
        print("  skipped speed/candidate figures: run python -m src.data.candidates first")
    print("\nDone. Figures are in", figs)


if __name__ == "__main__":
    main()
