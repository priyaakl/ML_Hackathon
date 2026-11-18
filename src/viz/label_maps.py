"""Step 4b: Folium maps to CHECK THE LABELS by eye (Person 1).

For one trip the map shows:
  * nearby road network (grey)
  * the reference route (green)
  * the raw GPS track as a thin line (blue)
  * GPS points coloured by label quality: green = label_valid (<= 50 m from its reference arc),
    red = invalid. Hover a point to see obs_idx, time, distance and its assigned reference arc.
  * a thin line from each invalid point to the closest point of its assigned reference arc
  * markers where the time between two GPS points is larger than 60 s (gaps)
Layers can be switched on and off with the control in the top-right corner.

Run from the repo root (needs labels_*.parquet from `python -m src.data.labels`):
    python -m src.viz.label_maps                      # the 4 default example trips
    python -m src.viz.label_maps --trip 00000054      # any trip (train or student_test)

Writes: results/maps/label_check_<trip>.html   (open in a browser; map tiles need internet)
This file does not touch src/viz/folium_maps.py (Person 3's route-map helpers).
"""
import argparse
import json
import numpy as np
import pandas as pd
import folium

from src.common.config import load_config
from src.common.io import load_trip
from src.common.geo import trip_origin, project

# Default example trips: clean | broken reference | partly mismatched | very long time gap.
DEFAULT_TRIPS = ["00000001", "00000054", "00000002", "00000090"]
NEAR_ROAD_M = 120          # draw only road arcs within this distance of some GPS point
MAX_ROAD_ARCS = 6000       # safety cap so the HTML stays small
MAX_VALID_POINTS = 1500    # valid points are subsampled for file size; invalid ones are kept (up to MAX_INVALID)
MAX_INVALID_POINTS = 2500
MAX_LINKS = 400            # lines from invalid points to their reference arc
GAP_S = 60.0

GREEN, RED, BLUE, GREY = "#147a58", "#d0021b", "#215cb5", "#7b8998"


def _find_partition(trip_id, splits):
    for part in ("train", "student_test"):
        if trip_id in splits[part]:
            return part
    raise SystemExit(f"Trip {trip_id} is not in train or student_test.")


def _evenly(idx, k):
    """At most k entries of idx, evenly spread (keeps order)."""
    if len(idx) <= k:
        return idx
    return idx[np.linspace(0, len(idx) - 1, k).astype(int)]


def _nearby_arcs(P, N, ar, radius):
    """Indices of arcs with an endpoint or midpoint within `radius` metres of some GPS point (projected coords)."""
    from scipy.spatial import cKDTree
    A, B = N[ar["u"].to_numpy()], N[ar["v"].to_numpy()]
    tree = cKDTree(P[:: max(1, len(P) // 4000)])          # a subsample of the track is enough for a road filter
    near = np.zeros(len(A), dtype=bool)
    for Q in (A, B, (A + B) / 2):
        near |= tree.query(Q, distance_upper_bound=radius)[0] < np.inf
    idx = np.flatnonzero(near)
    return idx[:: int(np.ceil(len(idx) / MAX_ROAD_ARCS))] if len(idx) > MAX_ROAD_ARCS else idx


def _closest_point_on_segment(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 == 0 else float(np.clip(((p - a) @ ab) / L2, 0.0, 1.0))
    return a + t * ab


def build_label_map(trip, labels):
    """trip: dict from load_trip (with route). labels: the rows of labels_*.parquet for this trip.
    Returns a folium.Map. All geometry is drawn from lon/lat directly (only the distance work is in metres)."""
    tr, nd, ar, rt = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    tid = trip["trip_id"]
    lab = labels.sort_values("obs_idx").reset_index(drop=True)
    assert len(lab) == len(tr), "labels and track have different lengths: re-run python -m src.data.labels"

    lon0, lat0 = trip_origin(tr)
    P = project(tr["lon"], tr["lat"], lon0, lat0)
    N = project(nd["lon"], nd["lat"], lon0, lat0)
    u, v = ar["u"].to_numpy(), ar["v"].to_numpy()
    lonlat = nd[["lon", "lat"]].to_numpy()

    def arc_latlon(a):
        return [[lonlat[u[a], 1], lonlat[u[a], 0]], [lonlat[v[a], 1], lonlat[v[a], 0]]]

    m = folium.Map(location=[float(tr["lat"].mean()), float(tr["lon"].mean())], zoom_start=15,
                   control_scale=True, max_zoom=19)
    m.fit_bounds([[float(tr["lat"].min()), float(tr["lon"].min())], [float(tr["lat"].max()), float(tr["lon"].max())]])

    # ---- road network near the track
    g_roads = folium.FeatureGroup(name="Road network (nearby arcs)", show=True)
    for a in _nearby_arcs(P, N, ar, NEAR_ROAD_M):
        folium.PolyLine(arc_latlon(a), color=GREY, weight=2, opacity=0.5).add_to(g_roads)
    g_roads.add_to(m)

    # ---- reference route
    g_ref = folium.FeatureGroup(name="Reference route (.route)", show=True)
    for a in rt:
        folium.PolyLine(arc_latlon(int(a)), color=GREEN, weight=5, opacity=0.65).add_to(g_ref)
    g_ref.add_to(m)

    # ---- raw GPS track as one line
    g_trk = folium.FeatureGroup(name="GPS track (line)", show=False)
    step = max(1, len(tr) // 3000)
    folium.PolyLine(tr[["lat", "lon"]].to_numpy()[::step].tolist(), color=BLUE, weight=1.5, opacity=0.7).add_to(g_trk)
    g_trk.add_to(m)

    # ---- GPS points coloured by label validity
    valid = lab["label_valid"].to_numpy(dtype=bool)
    shown_valid = _evenly(np.flatnonzero(valid), MAX_VALID_POINTS)
    shown_invalid = _evenly(np.flatnonzero(~valid), MAX_INVALID_POINTS)
    for name, idx, colour, show in [("GPS points: valid label (<= 50 m)", shown_valid, GREEN, True),
                                    ("GPS points: INVALID label (> 50 m)", shown_invalid, RED, True)]:
        g = folium.FeatureGroup(name=name, show=show)
        for i in idx:
            tip = (f"obs {i} | t={tr['t'].iat[i]:.0f}s | dist to ref arc {lab['label_dist_m'].iat[i]:.1f} m"
                   f" | ref arc {int(lab['ref_arc_id'].iat[i])}")
            folium.CircleMarker([float(tr["lat"].iat[i]), float(tr["lon"].iat[i])], radius=3, color=colour,
                                fill=True, fill_opacity=0.85, weight=1, tooltip=tip).add_to(g)
        g.add_to(m)

    # ---- links from invalid points to their assigned reference arc
    g_link = folium.FeatureGroup(name="Invalid point -> its reference arc", show=True)
    for i in _evenly(np.flatnonzero(~valid), MAX_LINKS):
        a = int(lab["ref_arc_id"].iat[i])
        c = _closest_point_on_segment(P[i], N[u[a]], N[v[a]])
        # convert the closest point back to lon/lat (local equirectangular inverse)
        clon = lon0 + np.degrees(c[0] / (6371008.8 * np.cos(np.radians(lat0))))
        clat = lat0 + np.degrees(c[1] / 6371008.8)
        folium.PolyLine([[float(tr["lat"].iat[i]), float(tr["lon"].iat[i])], [float(clat), float(clon)]],
                        color=RED, weight=1, opacity=0.6, dash_array="4").add_to(g_link)
    g_link.add_to(m)

    # ---- time gaps
    g_gap = folium.FeatureGroup(name=f"Time gaps > {GAP_S:.0f} s", show=True)
    dt = np.diff(tr["t"].to_numpy(dtype=np.float64))
    for i in np.flatnonzero(dt > GAP_S):
        hrs = dt[i] / 3600
        folium.Marker([float(tr["lat"].iat[i + 1]), float(tr["lon"].iat[i + 1])],
                      tooltip=f"gap of {dt[i]:.0f} s ({hrs:.2f} h) before obs {i + 1}",
                      icon=folium.Icon(color="orange", icon="time", prefix="glyphicon")).add_to(g_gap)
    g_gap.add_to(m)

    # ---- title box
    n_inv = int((~valid).sum())
    html = (f"<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 12px;"
            f"border:1px solid #999;border-radius:6px;font:13px sans-serif;max-width:330px'>"
            f"<b>Label check: trip {tid}</b><br>{len(tr)} GPS points, {len(rt)} route arcs<br>"
            f"invalid labels: {n_inv} ({100 * n_inv / len(tr):.1f}%)<br>"
            f"median dist to ref arc: {lab['label_dist_m'].median():.1f} m<br>"
            f"gaps > {GAP_S:.0f} s: {int((dt > GAP_S).sum())}<br>"
            f"<span style='color:{GREEN}'>&#9679;</span> valid &nbsp;"
            f"<span style='color:{RED}'>&#9679;</span> invalid &nbsp;"
            f"<span style='color:{GREEN}'>&#9644;</span> reference route<br>"
            f"<small>Large trips are subsampled for display only.</small></div>")
    m.get_root().html.add_child(folium.Element(html))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trip", nargs="*", default=DEFAULT_TRIPS, help="trip ids (8 digits), default: 4 examples")
    args = ap.parse_args()

    cfg = load_config()
    root, proc, res = cfg["paths"]["data_root"], cfg["paths"]["processed"], cfg["paths"]["results"]
    out = res / "maps"
    out.mkdir(parents=True, exist_ok=True)
    with open(proc / "splits.json") as f:
        splits = json.load(f)
    cache = {}
    for tid in args.trip:
        part = _find_partition(tid, splits)
        if part not in cache:
            cache[part] = pd.read_parquet(proc / f"labels_{part}.parquet")
        lab = cache[part]
        lab = lab[lab.trip_id == tid]
        trip = load_trip(root / part / tid)
        m = build_label_map(trip, lab)
        path = out / f"label_check_{tid}.html"
        m.save(str(path))
        print(f"trip {tid} ({part}): saved {path}")
    print("\nOpen the HTML files in a browser (map tiles need internet).")


if __name__ == "__main__":
    main()
