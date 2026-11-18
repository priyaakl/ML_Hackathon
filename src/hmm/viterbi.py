"""Decode one PathSnap trip from candidate arcs and aligned classifier scores."""

from math import isfinite
from operator import index

import numpy as np
import pandas as pd

from .graph import RoadGraph


def decode_trip(
    cands: pd.DataFrame,
    probs: np.ndarray,
    graph: RoadGraph,
    params: dict,
) -> list[int]:
    """Return one arc ID per GPS observation, ordered by zero-based obs_idx.

    Pass the raw track length as params["n_observations"] when it is known.
    Candidate rows alone cannot reveal missing observations at the end.
    Leading and all-missing observations use -1; other missing observations
    carry the most recent decoded arc forward. No reference labels are read.

    The default settings are initial values for development, before tuning.
    """
    scale_m = float(params.get("transition_scale_m", 30.0))
    max_dist = float(params.get("max_network_distance_m", 250.0))
    disconnected_penalty = float(params.get("disconnected_penalty", 8.0))
    gap_reset_s = float(params.get("gap_reset_s", 30.0))
    if any(not isfinite(value) or value <= 0 for value in (
        scale_m, max_dist, disconnected_penalty, gap_reset_s
    )):
        raise ValueError("Decoder transition parameters must be finite and positive.")

    probabilities = np.asarray(probs, dtype=float)
    if probabilities.shape != (len(cands),):
        raise ValueError("Provide exactly one probability per candidate row.")
    if not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Candidate probabilities must be finite and between 0 and 1.")
    if "trip_id" in cands and cands["trip_id"].nunique() > 1:
        raise ValueError("decode_trip accepts candidates for just one trip.")

    requested_count = params.get("n_observations")
    if requested_count is not None:
        requested_count = index(requested_count)
        if requested_count < 0:
            raise ValueError("n_observations must be nonnegative.")
    if cands.empty:
        return [-1] * (requested_count or 0)

    observations = cands["obs_idx"].to_numpy()
    arcs = cands["cand_arc_id"].to_numpy()
    if not np.issubdtype(observations.dtype, np.integer) or np.any(observations < 0):
        raise ValueError("obs_idx must contain nonnegative integers.")
    if not np.issubdtype(arcs.dtype, np.integer) or np.any(
        (arcs < 0) | (arcs >= len(graph.arc_endpoints))
    ):
        raise ValueError("cand_arc_id must contain valid arc IDs for this trip.")

    inferred_count = int(observations.max()) + 1
    count = inferred_count if requested_count is None else requested_count
    if count < inferred_count:
        raise ValueError("n_observations does not include all candidate observations.")

    # Sort both the candidates and their emissions together, preserving score alignment.
    order = np.lexsort((cands["cand_rank"].to_numpy(), observations))
    observations = observations[order]
    arcs = arcs[order]
    emissions = np.log(np.maximum(probabilities[order], 1e-12))
    positions = cands[["x", "y", "t"]].to_numpy(dtype=float)[order]
    if not np.isfinite(positions).all():
        raise ValueError("x, y and t must be finite.")
    starts = np.r_[0, np.flatnonzero(observations[1:] != observations[:-1]) + 1]
    ends = np.r_[starts[1:], len(arcs)]
    backpointers = np.full(len(arcs), -1, dtype=np.intp)
    predictions = [-1] * count
    previous_scores = np.empty(0)
    segment_end_rows = []

    for step, (start, end) in enumerate(zip(starts, ends)):
        current_scores = emissions[start:end].copy()
        if step:
            prev_start = starts[step - 1]
            delta_x, delta_y, elapsed = positions[start] - positions[prev_start]
            if elapsed < 0:
                raise ValueError("Timestamps must be nondecreasing in observation order.")
            if elapsed > gap_reset_s:
                segment_end_rows.append(int(prev_start + np.argmax(previous_scores)))
            else:
                displacement = float(np.hypot(delta_x, delta_y))
                # Allow longer searches for larger displacements, up to a hard cap.
                distance_cap = min(max_dist, displacement + 3 * scale_m)
                for row in range(start, end):
                    target = int(arcs[row])
                    best_score = -np.inf
                    best_row = -1
                    for offset, previous_score in enumerate(previous_scores):
                        previous_row = prev_start + offset
                        source = int(arcs[previous_row])
                        distance = graph.network_distance_between_arcs(
                            source, target, distance_cap
                        )
                        if source == target:
                            # A vehicle can advance along an arc without changing state.
                            distance = min(displacement, graph.arc_length_m[source])
                        penalty = (
                            abs(distance - displacement) / scale_m
                            if isfinite(distance)
                            else disconnected_penalty
                        )
                        score = previous_score - penalty
                        if score > best_score:
                            best_score = score
                            best_row = previous_row
                    current_scores[row - start] += best_score
                    backpointers[row] = best_row

        # Subtract a shared constant per observation to keep scores numerically stable.
        previous_scores = current_scores - current_scores.max()

    segment_end_rows.append(int(starts[-1] + np.argmax(previous_scores)))
    for row in segment_end_rows:
        while row >= 0:
            predictions[int(observations[row])] = int(arcs[row])
            row = backpointers[row]

    last_arc = -1
    for obs, arc in enumerate(predictions):
        if arc == -1:
            predictions[obs] = last_arc
        else:
            last_arc = arc
    return predictions
