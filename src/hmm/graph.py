"""Directed road network for a single PathSnap trip."""

from functools import lru_cache
from heapq import heappop, heappush
from math import inf, isnan

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_EARTH_RADIUS_M = 6_371_008.8


class RoadGraph:
    """Node coordinates are (longitude, latitude); arc IDs are file row indices."""

    def __init__(self, nodes: np.ndarray, arcs: np.ndarray):
        self.nodes = np.array(nodes, dtype=float, copy=True)
        self.arc_endpoints = np.array(arcs, dtype=np.int64, copy=True)
        if not np.isfinite(self.nodes).all() or np.any(
            np.abs(self.nodes) > [180, 90]
        ):
            raise ValueError("Node coordinates must be valid longitude and latitude.")
        if np.any(self.arc_endpoints < 0) or np.any(
            self.arc_endpoints >= len(self.nodes)
        ):
            raise ValueError("An arc references a node outside the node file.")

        starts, ends = self.arc_endpoints.T
        lon_from, lat_from = np.radians(self.nodes[starts]).T
        lon_to, lat_to = np.radians(self.nodes[ends]).T
        # Haversine converts the geographic endpoints to lengths without choosing a CRS.
        haversine = (
            np.sin((lat_to - lat_from) / 2) ** 2
            + np.cos(lat_from) * np.cos(lat_to) * np.sin((lon_to - lon_from) / 2) ** 2
        )
        self.arc_length_m = 2 * _EARTH_RADIUS_M * np.arcsin(
            np.sqrt(np.clip(haversine, 0, 1))
        )

        outgoing = [[] for _ in self.nodes]
        for arc_id, node_id in enumerate(starts):
            outgoing[node_id].append(arc_id)
        self._outgoing = tuple(tuple(arc_ids) for arc_ids in outgoing)

        adjacency = coo_matrix(
            (np.ones(len(starts), dtype=bool), (starts, ends)),
            shape=(len(self.nodes), len(self.nodes)),
        ).tocsr()
        self.component_count, self.component_labels = connected_components(
            adjacency, directed=True, connection="weak"
        )
        incident = np.bincount(self.arc_endpoints.ravel(), minlength=len(self.nodes))
        self.isolated_node_count = int(np.count_nonzero(incident == 0))

        for array in (self.nodes, self.arc_endpoints, self.arc_length_m, self.component_labels):
            array.setflags(write=False)
        self._cached_node_distance = lru_cache(maxsize=32_768)(self._node_distance)

    def _endpoints(self, arc_id: int) -> tuple[int, int]:
        if not 0 <= arc_id < len(self.arc_endpoints):
            raise IndexError(f"Unknown arc ID: {arc_id}")
        start, end = self.arc_endpoints[arc_id]
        return int(start), int(end)

    def successors(self, arc_id: int) -> tuple[int, ...]:
        """Return arcs whose start node is this arc's end node."""
        _, end = self._endpoints(arc_id)
        return self._outgoing[end]

    def arc_geometry(self, arc_id: int) -> np.ndarray:
        """Return the two endpoints in (longitude, latitude) order."""
        start, end = self._endpoints(arc_id)
        return self.nodes[[start, end]]

    def network_distance_between_arcs(self, a: int, b: int, max_dist: float) -> float:
        """Return directed midpoint-to-midpoint metres, or infinity beyond the cap."""
        _, end = self._endpoints(a)
        start, _ = self._endpoints(b)
        if isnan(max_dist) or max_dist < 0:
            raise ValueError("max_dist must be nonnegative.")
        if a == b:
            return 0.0
        endpoint_distance = float((self.arc_length_m[a] + self.arc_length_m[b]) / 2)
        if endpoint_distance > max_dist or self.component_labels[end] != self.component_labels[start]:
            return inf
        return endpoint_distance + self._cached_node_distance(
            end, start, max_dist - endpoint_distance
        )

    def _node_distance(self, start: int, target: int, max_dist: float) -> float:
        """Run Dijkstra, skipping paths longer than max_dist."""
        distances = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            if node == target:
                return distance
            for arc_id in self._outgoing[node]:
                neighbour = int(self.arc_endpoints[arc_id, 1])
                candidate = distance + float(self.arc_length_m[arc_id])
                if candidate <= max_dist and candidate < distances.get(neighbour, inf):
                    distances[neighbour] = candidate
                    heappush(queue, (candidate, neighbour))
        return inf

    def route_continuity(self, arc_sequence, max_hops: int = 2) -> float:
        """Fraction of consecutive predictions linked within max_hops arc steps.

        Repeated valid arcs count as continuous. Missing or invalid IDs break
        continuity. A sequence with no adjacent pairs has undefined continuity.
        """
        if max_hops < 0:
            raise ValueError("max_hops must be nonnegative.")
        arc_ids = list(arc_sequence)
        pair_count = len(arc_ids) - 1
        if pair_count < 1:
            return float("nan")
        valid_arcs = range(len(self.arc_endpoints))
        continuous = 0
        for previous, current in zip(arc_ids, arc_ids[1:]):
            if previous not in valid_arcs or current not in valid_arcs:
                continue
            frontier = {int(previous)}
            seen = set(frontier)
            for _ in range(max_hops):
                if current in seen or not frontier:
                    break
                frontier = {
                    neighbour for arc_id in frontier for neighbour in self.successors(arc_id)
                } - seen
                seen.update(frontier)
            continuous += current in seen
        return continuous / pair_count


def build_graph(nodes_path, arcs_path) -> RoadGraph:
    """Read the two headerless PathSnap files; paths or open files are accepted."""
    nodes = np.loadtxt(nodes_path, dtype=float, ndmin=2)
    arcs = np.loadtxt(arcs_path, dtype=np.int64, ndmin=2)
    if nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("The nodes and arcs files must each have two columns.")
    return RoadGraph(nodes, arcs)
