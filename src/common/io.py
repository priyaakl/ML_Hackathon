"""Readers for the raw PathSnap files. Each trip folder holds 4 files with the same stem:
   <id>.track  lon, lat, t(seconds)      (GPS observations)
   <id>.nodes  lon, lat                  (row index = node id; THIS trip's own road network)
   <id>.arcs   from_node, to_node        (row index = arc id)
   <id>.route  arc_id                    (reference route, in travel order; may be absent)
All files are whitespace/tab separated with no header.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def list_trips(data_root, partition):
    """Sorted trip ids (8-digit strings) found in data_root/<partition>/."""
    part_dir = Path(data_root) / partition
    return sorted(p.name for p in part_dir.iterdir() if p.is_dir())


def _read(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def load_trip(trip_dir, with_route=True):
    """Load one trip from its folder. Returns dict with track, nodes, arcs (DataFrames)
    and route (numpy array of arc ids, or None if absent / not requested)."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    out = {
        "trip_id": tid,
        "track": _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"]),
        "nodes": _read(trip_dir / f"{tid}.nodes", ["lon", "lat"]),
        "arcs": _read(trip_dir / f"{tid}.arcs", ["u", "v"]),
        "route": None,
    }
    route_path = trip_dir / f"{tid}.route"
    if with_route and route_path.exists():
        out["route"] = _read(route_path, ["arc_id"])["arc_id"].to_numpy(dtype=np.int64)
    return out
