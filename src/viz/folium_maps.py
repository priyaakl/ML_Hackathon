"""Render PathSnap route layers and optionally generate a training-trip demo."""

import argparse
from base64 import b64encode
from html import escape
from pathlib import Path
from zipfile import ZipFile

import folium
import numpy as np
import pandas as pd
from branca.element import Element
from scipy.spatial import cKDTree

from src.hmm.graph import RoadGraph, build_graph
from src.hmm.viterbi import decode_trip

_ROADS = "#7b8998"
_GPS = "#215cb5"
_REFERENCE = "#147a58"
_PREDICTION = "#bf4728"


def render_trip_map(
    track: np.ndarray,
    graph: RoadGraph,
    predicted_arcs,
    *,
    reference_arcs=None,
    trip_id: str,
    caption: str = "",
    leaflet_dir: Path | None = None,
) -> folium.Map:
    """Plot a (lon, lat, timestamp) track and its road arcs.

    Supply reference_arcs for development maps with shareable reference data.
    Omit that argument when rendering predictions for evaluator-facing use.
    """
    track = np.asarray(track, dtype=float)
    if track.ndim != 2 or track.shape[1] != 3 or len(track) == 0:
        raise ValueError("track must have at least one row of (lon, lat, timestamp).")
    if not np.isfinite(track).all():
        raise ValueError("Track values must be finite.")

    gps_latlon = track[:, [1, 0]]
    map_ = folium.Map(tiles=None, control_scale=True, zoom_control=True, max_zoom=19)
    if leaflet_dir is not None:
        _embed_leaflet(map_, leaflet_dir)
    map_.fit_bounds([gps_latlon.min(axis=0).tolist(), gps_latlon.max(axis=0).tolist()],
                    padding=(65, 65), max_zoom=17)

    road_geometry = _nearby_roads(track, graph)
    roads = folium.FeatureGroup(name="Nearby roads").add_to(map_)
    folium.PolyLine(
        road_geometry[:, :, ::-1].tolist(),
        color=_ROADS, weight=2, opacity=0.6, interactive=False,
    ).add_to(roads)

    if reference_arcs is not None:
        _add_route(map_, graph, reference_arcs, "Reference route", _REFERENCE, 7)
    _add_route(map_, graph, predicted_arcs, "Viterbi prediction", _PREDICTION, 4, "6 6")

    gps = folium.FeatureGroup(name="Original GPS").add_to(map_)
    for obs, (lat, lon) in enumerate(gps_latlon):
        folium.CircleMarker(
            location=(lat, lon), radius=2.6, color="#ffffff", weight=0.65,
            fill=True, fill_color=_GPS, fill_opacity=0.78,
            tooltip=f"GPS observation {obs + 1}",
        ).add_to(gps)
    folium.LayerControl(collapsed=False).add_to(map_)
    _add_legend(map_, trip_id, caption, include_reference=reference_arcs is not None)
    return map_


def _metres_per_degree(latitude: float) -> np.ndarray:
    return 111_195 * np.array([np.cos(np.deg2rad(latitude)), 1])


def _nearby_roads(track, graph):
    padding = 180 / np.maximum(_metres_per_degree(track[:, 1].mean()), 1)
    low = track[:, :2].min(axis=0) - padding
    high = track[:, :2].max(axis=0) + padding
    endpoints = graph.nodes[graph.arc_endpoints]
    overlaps = np.all((endpoints.max(axis=1) >= low) & (endpoints.min(axis=1) <= high), axis=1)
    # Bidirectional arcs share the same geometry in the context layer.
    pairs = np.unique(np.sort(graph.arc_endpoints[overlaps], axis=1), axis=0)
    return graph.nodes[pairs]


def _route_geometry(graph, arc_ids):
    # Draw each selected arc independently so gaps cannot create invented straight links.
    unique = list(dict.fromkeys(int(arc) for arc in arc_ids if arc != -1))
    return np.asarray([graph.arc_geometry(arc) for arc in unique]).reshape(-1, 2, 2)


def _add_route(map_, graph, arc_ids, name, color, width, dash=None):
    layer = folium.FeatureGroup(name=name).add_to(map_)
    geometry = _route_geometry(graph, arc_ids)
    if len(geometry):
        folium.PolyLine(
            geometry[:, :, ::-1].tolist(), color=color, weight=width, dash_array=dash,
            opacity=0.97, interactive=False,
        ).add_to(layer)


def _add_legend(map_, trip_id: str, caption: str, *, include_reference: bool):
    reference = (
        f'<div><i style="border-color:{_REFERENCE}"></i>Reference route</div>'
        if include_reference else ""
    )
    map_.get_root().header.add_child(Element("""
<style>
  .leaflet-container { background: #f2f5f7; }
  .pathsnap-key { position: absolute; bottom: 32px; left: 16px; z-index: 900;
    padding: 14px 17px; background: #fff; color: #203343; border: 1px solid #cad4db;
    border-radius: 6px; box-shadow: 0 2px 12px #20334322;
    font: 13px/1.6 system-ui, sans-serif; max-width: min(360px, calc(100% - 66px)); }
  .pathsnap-key h1 { font-size: 17px; margin: 0 0 2px; }
  .pathsnap-key p { margin: 0 0 8px; color: #465b6b; }
  .pathsnap-key i { display: inline-block; width: 21px; margin-right: 8px;
    vertical-align: middle; border-top: 3px solid; }
  .pathsnap-key i.gps { width: 7px; height: 7px; border: 0; border-radius: 50%;
    margin-left: 7px; margin-right: 15px; }
</style>
"""))
    map_.get_root().html.add_child(Element(f"""
<aside class="pathsnap-key" aria-label="Map legend">
  <h1>PathSnap · Trip {escape(trip_id)}</h1>
  {f"<p>{escape(caption)}</p>" if caption else ""}
  <div><i class="gps" style="background:{_GPS}"></i>Original GPS observations</div>
  <div><i style="border-color:{_ROADS}"></i>Nearby roads</div>
  {reference}
  <div><i style="border-color:{_PREDICTION};border-top-style:dashed"></i>Viterbi prediction</div>
</aside>
"""))


def _embed_leaflet(map_, directory: Path):
    """Embed a local Leaflet distribution in the demo HTML for offline use."""
    js_path = directory / "leaflet.min.js"
    if not js_path.exists():
        js_path = directory / "leaflet.js"
    css = (directory / "leaflet.css").read_text(encoding="utf-8")
    for filename in ("layers.png", "layers-2x.png", "marker-icon.png"):
        encoded = b64encode((directory / "images" / filename).read_bytes()).decode("ascii")
        css = css.replace(f"images/{filename}", f"data:image/png;base64,{encoded}")
    map_.default_js = []
    map_.default_css = []
    header = map_.get_root().header
    license_file = directory / "LICENSE"
    if not license_file.exists():
        license_file = Path("/usr/share/doc/libjs-leaflet/copyright")
    if license_file.exists():
        notice = license_file.read_text(encoding="utf-8").replace("--", "—")
        header.add_child(Element(f"<!-- Leaflet distribution notice\n{notice}\n-->"))
    header.add_child(Element(f"<script>{js_path.read_text(encoding='utf-8')}</script>"))
    header.add_child(Element(f"<style>{css}</style>"))


def _demo_candidates(track: np.ndarray, graph: RoadGraph):
    """Produce temporary distance-based candidates for the standalone demo."""
    origin = track[0, :2]
    scale = _metres_per_degree(origin[1])
    points = (track[:, :2] - origin) * scale
    nodes = (graph.nodes - origin) * scale
    if len(graph.arc_endpoints) == 0:
        raise ValueError("The trip has no road arcs to score.")
    start, end = nodes[graph.arc_endpoints].transpose(1, 0, 2)
    shortlist_size = min(10, len(start))
    _, arc_ids = cKDTree((start + end) / 2).query(points, k=list(range(1, shortlist_size + 1)))
    vectors = end[arc_ids] - start[arc_ids]
    offsets = points[:, None, :] - start[arc_ids]
    squared_lengths = np.sum(vectors ** 2, axis=2)
    fractions = np.divide(
        np.sum(offsets * vectors, axis=2), squared_lengths,
        out=np.zeros_like(squared_lengths), where=squared_lengths > 0,
    )
    distances = np.linalg.norm(offsets - np.clip(fractions, 0, 1)[..., None] * vectors, axis=2)
    n_candidates = min(5, shortlist_size)
    order = np.argsort(distances, axis=1, kind="stable")[:, :n_candidates]
    arc_ids = np.take_along_axis(arc_ids, order, axis=1)
    ranked_distances = np.take_along_axis(distances, order, axis=1)
    scores = np.exp(-(ranked_distances - ranked_distances[:, :1]) / 20)
    scores /= scores.sum(axis=1, keepdims=True)

    candidates = pd.DataFrame({
        "obs_idx": np.repeat(np.arange(len(track)), n_candidates),
        "cand_rank": np.tile(np.arange(1, n_candidates + 1), len(track)),
        "cand_arc_id": arc_ids.ravel(),
        "x": np.repeat(points[:, 0], n_candidates),
        "y": np.repeat(points[:, 1], n_candidates),
        "t": np.repeat(track[:, 2], n_candidates),
    })
    return candidates, scores.ravel()


def _export_demo_image(output, track, graph, reference, prediction, trip_id, caption):
    """Export the same map layers as a report figure."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    origin = track[:, :2].mean(axis=0)
    scale = _metres_per_degree(origin[1])
    points = (track[:, :2] - origin) * scale
    figure = Figure(figsize=(8, 6), facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.set_facecolor("#f2f5f7")
    paths = [
        (_nearby_roads(track, graph), _ROADS, 0.9, "solid", 0.45, "Nearby roads"),
        (_route_geometry(graph, reference), _REFERENCE, 3.7, "solid", 0.97, "Reference route"),
        (_route_geometry(graph, prediction), _PREDICTION, 1.9, "dashed", 0.97, "Viterbi prediction"),
    ]
    for geometry, color, width, style, opacity, name in paths:
        axes.add_collection(LineCollection(
            (geometry - origin) * scale, colors=color, linewidths=width,
            linestyles=style, alpha=opacity, label=name,
        ))
    axes.scatter(*points.T, s=10, facecolor=_GPS, edgecolor="white", linewidths=0.22,
                 alpha=0.78, label="Original GPS observations", zorder=3)
    axes.set_xlim(points[:, 0].min() - 90, points[:, 0].max() + 90)
    axes.set_ylim(points[:, 1].min() - 90, points[:, 1].max() + 90)
    axes.set_aspect("equal")
    axes.set_axisbelow(True)
    axes.grid(color="white", linewidth=1.0)
    for axis in (axes.xaxis, axes.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
    axes.set_xlabel("Metres east of track centre")
    axes.set_ylabel("Metres north of track centre")
    axes.set_title(caption, loc="left", fontsize=10, color="#465b6b", pad=10)
    figure.suptitle(f"PathSnap · Trip {trip_id}", fontsize=18, fontweight="bold",
                    color="#203343", x=0.06, ha="left")
    axes.legend(loc="best", fontsize=9, facecolor="white", framealpha=0.95)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Generate a Folium map for a supplied training trip.")
    parser.add_argument("archive", type=Path, help="Path to Student folder.zip")
    parser.add_argument("trip_id")
    parser.add_argument("output", type=Path, help="Destination HTML file")
    parser.add_argument("--image", type=Path, help="Optional PNG export for the report")
    parser.add_argument("--leaflet-dir", type=Path, help="Local Leaflet distribution for offline HTML")
    args = parser.parse_args()
    if len(args.trip_id) != 8 or not args.trip_id.isascii() or not args.trip_id.isdigit():
        parser.error("trip_id must be eight digits")

    prefix = f"Student folder/train/{args.trip_id}/{args.trip_id}"
    with ZipFile(args.archive) as archive:
        with archive.open(prefix + ".nodes") as nodes, archive.open(prefix + ".arcs") as arcs:
            graph = build_graph(nodes, arcs)
        with archive.open(prefix + ".track") as track_file:
            track = np.loadtxt(track_file, ndmin=2)
        with archive.open(prefix + ".route") as route_file:
            reference = np.loadtxt(route_file, dtype=int, ndmin=1)

    candidates, scores = _demo_candidates(track, graph)
    prediction = decode_trip(candidates, scores, graph, {"n_observations": len(track)})
    caption = f"Training trip · {len(track)} GPS observations · Distance-based demo scores"
    map_ = render_trip_map(
        track, graph, prediction, reference_arcs=reference, trip_id=args.trip_id,
        caption=caption,
        leaflet_dir=args.leaflet_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    map_.save(args.output)
    if args.image:
        _export_demo_image(args.image, track, graph, reference, prediction, args.trip_id, caption)
    print(f"Saved {args.output} ({len(track)} GPS observations, {len(reference)} reference arcs)")


if __name__ == "__main__":
    main()
