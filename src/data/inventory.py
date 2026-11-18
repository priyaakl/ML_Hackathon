"""Step 1.1: dataset inventory + splits.json.
Run from the repo root:   python -m src.data.inventory
Writes: data/processed/splits.json, results/dataset_inventory.csv
"""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.config import load_config
from src.common.io import list_trips, load_trip
from src.common.geo import trip_origin, project

PARTITIONS = ["train", "student_test"]   # evaluator trips are NOT in our zip (they come at the demo)


def main():
    cfg = load_config()
    root = cfg["paths"]["data_root"]
    cfg["paths"]["processed"].mkdir(parents=True, exist_ok=True)
    cfg["paths"]["results"].mkdir(parents=True, exist_ok=True)

    splits = {p: list_trips(root, p) for p in PARTITIONS}
    splits["evaluator_test"] = []          # filled at the demo, never used in development
    overlap = set(splits["train"]) & set(splits["student_test"])
    assert not overlap, f"trips in two partitions: {overlap}"
    with open(cfg["paths"]["processed"] / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    rows = []
    jobs = [(p, t) for p in PARTITIONS for t in splits[p]]
    for part, tid in tqdm(jobs, desc="inventory"):
        d = load_trip(root / part / tid)
        tr, nd, ar, rt = d["track"], d["nodes"], d["arcs"], d["route"]
        dt = np.diff(tr["t"].to_numpy())
        lon0, lat0 = trip_origin(tr)
        xy = project(tr["lon"], tr["lat"], lon0, lat0)
        ext = xy.max(0) - xy.min(0)
        connected = float((ar["v"].to_numpy()[rt[:-1]] == ar["u"].to_numpy()[rt[1:]]).mean()) if len(rt) > 1 else 1.0
        rows.append(dict(
            partition=part, trip_id=tid,
            n_obs=len(tr), n_nodes=len(nd), n_arcs=len(ar), n_route_arcs=len(rt),
            missing_track=int(tr.isna().sum().sum()), missing_nodes=int(nd.isna().sum().sum()),
            time_monotonic=bool((dt > 0).all()),
            pct_dt_1s=round(float((dt == 1).mean()) * 100, 2),
            max_dt_s=float(dt.max()), n_gaps_gt_60s=int((dt > 60).sum()),
            extent_x_m=round(float(ext[0])), extent_y_m=round(float(ext[1])),
            lon0=round(lon0, 5), lat0=round(lat0, 5),
            route_connected_frac=round(connected, 4),
            route_arc_oob=int((rt >= len(ar)).sum()),
        ))
    inv = pd.DataFrame(rows)
    inv.to_csv(cfg["paths"]["results"] / "dataset_inventory.csv", index=False)

    print("\nTrips per partition:", {p: len(splits[p]) for p in PARTITIONS})
    print(inv.groupby("partition")[["n_obs", "n_nodes", "n_arcs", "n_route_arcs"]].agg(["sum", "median", "max"]).T)
    print("\nNaNs:", int(inv.missing_track.sum() + inv.missing_nodes.sum()),
          "| non-monotonic time trips:", int((~inv.time_monotonic).sum()),
          "| trips with gaps > 60 s:", int((inv.n_gaps_gt_60s > 0).sum()),
          "| trips with route jumps:", int((inv.route_connected_frac < 1).sum()))
    print("Saved: data/processed/splits.json, results/dataset_inventory.csv")


if __name__ == "__main__":
    main()
