"""PathSnap prediction pipeline (Person 4).

predict_trip(trip_id) runs the full map-matching pipeline for ONE trip:

    raw trip files  ->  candidates + features   (Person 1: build_candidates_for_trip)
                    ->  candidate probabilities (Person 2: CandidateScorer, or the MOCK stub below)
                    ->  connected route         (Person 3: decode_trip, HMM + Viterbi)

Ground truth (.route) is NEVER read here. Scoring a prediction against the reference
route happens afterwards, in src/api/evaluator.py, on the server only.

Run a quick self-test from the repo root:   python -m src.api.pipeline 00000000
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.data.candidates import build_candidates_for_trip
from src.hmm.graph import build_graph
from src.hmm.viterbi import decode_trip

# Labels that travel with every response, so placeholder output can never be mistaken for real ML results.
MOCK_SCORER_LABEL = "MOCK scorer: plan Section 5.3 stub exp(-dist_to_arc_m / 20), NOT a trained model"
REAL_SCORER_LABEL = "CandidateScorer (models/final_model.joblib)"
MOCK_DECODER_LABEL = "MOCK decoder: per-observation argmax, NO HMM"
REAL_DECODER_LABEL = "HMM + Viterbi (src/hmm/viterbi.py decode_trip)"


# ----------------------------------------------------------------------------------------------
# Small helper functions
# ----------------------------------------------------------------------------------------------
def mock_scores(cands: pd.DataFrame) -> np.ndarray:
    """Placeholder until Person 2 delivers: nearer arcs get higher scores.
    exp(-distance / 20 m), normalised so the candidates of each GPS point sum to 1."""
    raw = np.exp(-cands["dist_to_arc_m"].to_numpy(dtype=float) / 20.0)
    totals = pd.Series(raw).groupby(cands["obs_idx"].to_numpy()).transform("sum").to_numpy()
    return raw / np.where(totals > 0, totals, 1.0)


def classifier_only_arcs(cands: pd.DataFrame, probs: np.ndarray, n_obs: int) -> list:
    """For every GPS point, pick its highest-probability candidate (no road connectivity used).
    GPS points with no candidates get -1."""
    out = np.full(n_obs, -1, dtype=np.int64)
    if len(cands):
        tmp = pd.DataFrame({
            "obs_idx": cands["obs_idx"].to_numpy(),
            "arc": cands["cand_arc_id"].to_numpy(),
            "p": probs,
        })
        best = tmp.loc[tmp.groupby("obs_idx")["p"].idxmax()]
        out[best["obs_idx"].to_numpy()] = best["arc"].to_numpy()
    return out.tolist()


def route_geometry(graph, arcs: list) -> list:
    """Turn a list of arc ids (one per GPS point) into a [lat, lon] polyline for the map.
    Repeated arcs are drawn once; -1 (no prediction) is skipped."""
    coords, prev = [], None
    for arc in arcs:
        if arc < 0 or arc == prev:
            continue
        (lon1, lat1), (lon2, lat2) = graph.arc_geometry(int(arc))
        start = [float(lat1), float(lon1)]
        if not coords or coords[-1] != start:
            coords.append(start)
        coords.append([float(lat2), float(lon2)])
        prev = arc
    return coords


def route_segments(graph, arcs: list) -> list:
    """One [[lat, lon], [lat, lon]] segment per distinct predicted arc, in travel order.
    The UI draws each arc separately, so a gap between two arcs is never drawn as an
    invented straight road (same approach as Person 3's folium_maps._route_geometry)."""
    segments = []
    for arc in dict.fromkeys(int(a) for a in arcs if a >= 0):
        (lon1, lat1), (lon2, lat2) = graph.arc_geometry(arc)
        segments.append([[float(lat1), float(lon1)], [float(lat2), float(lon2)]])
    return segments


# ----------------------------------------------------------------------------------------------
# The pipeline: created ONCE when the API starts, then reused for every request
# ----------------------------------------------------------------------------------------------
class PathSnapPipeline:
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or load_config()
        api = self.cfg["api"]

        # Which folder holds the trips the evaluator can pick (config, or env var at the demo).
        self.trips_dir = Path(os.environ.get("PATHSNAP_EVAL_DIR", api["eval_trips_dir"])).expanduser()

        # Candidate settings MUST match the ones the model was trained with, so they come from config.
        self.radius_m = self.cfg["candidates"]["radius_m"]
        self.max_candidates = self.cfg["candidates"]["max_candidates"]

        self.use_mock_scorer = bool(api.get("use_mock_scorer", True))
        self.use_mock_decoder = bool(api.get("use_mock_decoder", True))

        # Heavy objects, loaded once here (not on every request).
        self.scorer = None if self.use_mock_scorer else self._load_scorer()
        self.hmm_params, self.hmm_params_source = self._load_hmm_params()
        self._graphs = {}                       # trip_id -> RoadGraph (each trip has its own road network)

    # ---- loading ------------------------------------------------------------------------------
    @staticmethod
    def _load_scorer():
        from src.models.scorer import CandidateScorer   # Person 2's file; only imported when switched on
        return CandidateScorer(
            str(ROOT / "models" / "final_model.joblib"),
            str(ROOT / "models" / "feature_columns.json"),
        )

    @staticmethod
    def _load_hmm_params():
        path = ROOT / "models" / "hmm_config.json"
        if path.exists():
            return json.loads(path.read_text()), "models/hmm_config.json"
        return {}, "decode_trip default values (models/hmm_config.json not found)"

    # ---- trips --------------------------------------------------------------------------------
    def list_trips(self) -> list:
        if not self.trips_dir.is_dir():
            raise FileNotFoundError(f"Trips folder not found: {self.trips_dir}")
        return sorted(p.name for p in self.trips_dir.iterdir() if p.is_dir())

    def trip_dir(self, trip_id: str) -> Path:
        if trip_id not in self.list_trips():
            raise KeyError(f"Unknown trip_id '{trip_id}'")
        return self.trips_dir / trip_id

    def graph_for(self, trip_id: str):
        """Build a trip's road graph the first time it is needed, then keep it in memory."""
        if trip_id not in self._graphs:
            d = self.trip_dir(trip_id)
            self._graphs[trip_id] = build_graph(d / f"{trip_id}.nodes", d / f"{trip_id}.arcs")
        return self._graphs[trip_id]

    def warm_up(self) -> int:
        """Pre-build every trip's road graph at startup, so requests do not pay that cost."""
        trips = self.list_trips()
        for tid in trips:
            self.graph_for(tid)
        return len(trips)

    def mode(self) -> dict:
        return {
            "scorer": MOCK_SCORER_LABEL if self.use_mock_scorer else REAL_SCORER_LABEL,
            "decoder": MOCK_DECODER_LABEL if self.use_mock_decoder else REAL_DECODER_LABEL,
            "decoder_params": self.hmm_params_source,
            "is_mock": self.use_mock_scorer or self.use_mock_decoder,
        }

    # ---- one prediction -----------------------------------------------------------------------
    def score(self, cands: pd.DataFrame) -> np.ndarray:
        if len(cands) == 0:
            return np.empty(0)
        probs = mock_scores(cands) if self.use_mock_scorer else self.scorer.predict_proba(cands)
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (len(cands),) or not np.isfinite(probs).all():
            raise ValueError("Scorer must return one finite probability per candidate row")
        return np.clip(probs, 0.0, 1.0)

    def predict_trip(self, trip_id: str) -> dict:
        t0 = time.perf_counter()
        trip_dir = self.trip_dir(trip_id)
        graph = self.graph_for(trip_id)
        t1 = time.perf_counter()

        # 1) candidates + features from raw files (no labels)
        cands, obs = build_candidates_for_trip(trip_dir, radius_m=self.radius_m, max_candidates=self.max_candidates)
        cands = cands.sort_values(["obs_idx", "cand_rank"]).reset_index(drop=True)
        n_obs = len(obs)
        t2 = time.perf_counter()

        # 2) probability that each candidate is the true road
        probs = self.score(cands)
        t3 = time.perf_counter()

        # 3) classifier-only route (kept for the required classifier-only vs HMM comparison)
        clf_arcs = classifier_only_arcs(cands, probs, n_obs)

        # 4) final route: HMM + Viterbi (or the mock = classifier-only)
        if self.use_mock_decoder:
            hmm_arcs = list(clf_arcs)
        else:
            params = {**self.hmm_params, "n_observations": n_obs}
            hmm_arcs = [int(a) for a in decode_trip(cands, probs, graph, params)]
        t4 = time.perf_counter()

        return {
            "trip_id": trip_id,
            "mode": self.mode(),
            "n_observations": n_obs,
            "n_candidate_rows": int(len(cands)),
            "n_obs_without_candidates": int((obs["n_cands"] == 0).sum()),
            "predicted_arcs": hmm_arcs,
            "classifier_only_arcs": clf_arcs,
            "gps": obs[["lat", "lon"]].to_numpy(dtype=float).tolist(),
            "route_geometry": route_geometry(graph, hmm_arcs),
            "route_segments": route_segments(graph, hmm_arcs),
            "timing": {
                "load_graph_s": round(t1 - t0, 3),
                "candidates_s": round(t2 - t1, 3),
                "scoring_s": round(t3 - t2, 3),
                "decoding_s": round(t4 - t3, 3),
                "total_s": round(t4 - t0, 3),
            },
        }


# ----------------------------------------------------------------------------------------------
# Plain function matching the plan's contract (Section 5.5): predict_trip(trip_id) -> dict
# ----------------------------------------------------------------------------------------------
_pipeline = None


def get_pipeline() -> PathSnapPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PathSnapPipeline()
    return _pipeline


def predict_trip(trip_id: str) -> dict:
    return get_pipeline().predict_trip(trip_id)


if __name__ == "__main__":
    pipe = get_pipeline()
    trips = pipe.list_trips()
    tid = sys.argv[1] if len(sys.argv) > 1 else trips[0]
    print("trips folder :", pipe.trips_dir)
    print("trips found  :", len(trips), trips)
    print("mode         :", pipe.mode())
    res = pipe.predict_trip(tid)
    print("trip         :", res["trip_id"])
    print("observations :", res["n_observations"], "| candidate rows:", res["n_candidate_rows"],
          "| obs without candidates:", res["n_obs_without_candidates"])
    print("predicted    :", len(res["predicted_arcs"]), "arcs, first 10 =", res["predicted_arcs"][:10])
    print("route points :", len(res["route_geometry"]))
    print("timing       :", res["timing"])
