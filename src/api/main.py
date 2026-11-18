"""PathSnap FastAPI backend (Person 4). Runs on Laptop 1.

Endpoints
    GET  /health    is the server alive? which scorer/decoder is loaded?
    GET  /trips     the trip ids the evaluator can choose from
    POST /predict   body {"trip_id": "..."}  -> predicted route + timing (+ metrics, once evaluator.py is added)

Start it from the repo root (host and port come from config.yaml):
    python -m src.api.main
or
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.pipeline import PathSnapPipeline
from src.common.config import load_config

# ----------------------------------------------------------------------------------------------
# Logging: one clear line per request on Laptop 1's terminal (the evaluator asks to see these)
# ----------------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")

CFG = load_config()


# ----------------------------------------------------------------------------------------------
# Request / response schemas: FastAPI checks every request and response against these
# ----------------------------------------------------------------------------------------------
class PredictRequest(BaseModel):
    trip_id: str


class Mode(BaseModel):
    scorer: str
    decoder: str
    decoder_params: str
    is_mock: bool


class Timing(BaseModel):
    load_graph_s: float
    candidates_s: float
    scoring_s: float
    decoding_s: float
    total_s: float


class HealthResponse(BaseModel):
    status: str
    n_trips: int
    mode: Mode
    startup_s: float


class TripsResponse(BaseModel):
    count: int
    trips: list[str]


class PredictResponse(BaseModel):
    trip_id: str
    mode: Mode
    n_observations: int
    n_candidate_rows: int
    n_obs_without_candidates: int
    predicted_arcs: list[int]
    classifier_only_arcs: list[int]
    gps: list[list[float]]              # [[lat, lon], ...] one per GPS point
    route_geometry: list[list[float]]   # [[lat, lon], ...] predicted route polyline
    timing: Timing
    metrics: Optional[dict] = None      # filled in by src/api/evaluator.py (next step)
    request_s: float                    # wall-clock time of the whole request on the server


# ----------------------------------------------------------------------------------------------
# Startup: load heavy objects ONCE (model, road graphs), not on every request
# ----------------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    pipe = PathSnapPipeline(CFG)
    n = pipe.warm_up()                       # builds every trip's road graph now
    app.state.pipeline = pipe
    app.state.startup_s = round(time.perf_counter() - t0, 3)
    mode = pipe.mode()
    log.info("PathSnap API ready in %.2fs | %d trips in %s", app.state.startup_s, n, pipe.trips_dir)
    log.info("scorer  : %s", mode["scorer"])
    log.info("decoder : %s (%s)", mode["decoder"], mode["decoder_params"])
    if mode["is_mock"]:
        log.warning("MOCK components are active: results are for testing the app, NOT real ML results")
    yield
    log.info("PathSnap API shutting down")


app = FastAPI(title="PathSnap API", version="0.1.0", lifespan=lifespan)

# CORS: lets a browser-based UI on another laptop call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG["api"].get("cors_origins", ["*"]),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_every_request(request: Request, call_next):
    """Logs who called, what they called, the result code and how long it took."""
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d in %.3fs", request.method, request.url.path, client,
             response.status_code, time.perf_counter() - t0)
    return response


# ----------------------------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    pipe = request.app.state.pipeline
    return {"status": "ok", "n_trips": len(pipe.list_trips()), "mode": pipe.mode(),
            "startup_s": request.app.state.startup_s}


@app.get("/trips", response_model=TripsResponse)
def trips(request: Request):
    try:
        ids = request.app.state.pipeline.list_trips()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"count": len(ids), "trips": ids}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    # A plain `def` (not async): FastAPI runs it in a worker thread, so /health still answers
    # while a long prediction is running.
    pipe = request.app.state.pipeline
    trip_id = body.trip_id.strip()
    t0 = time.perf_counter()
    try:
        result = pipe.predict_trip(trip_id)
    except KeyError:
        log.warning("PREDICT trip=%s -> unknown trip id", trip_id)
        raise HTTPException(status_code=404, detail=f"Unknown trip_id '{trip_id}'. Call GET /trips for valid ids.")
    except Exception as e:
        log.exception("PREDICT trip=%s -> failed", trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed for trip {trip_id}: {e}")

    result["metrics"] = None              # evaluator.py plugs in here in the next step
    result["request_s"] = round(time.perf_counter() - t0, 3)
    t = result["timing"]
    log.info("PREDICT trip=%s | obs=%d | cands=%d | candidates %.3fs, scoring %.3fs, decoding %.3fs | total %.3fs%s",
             trip_id, result["n_observations"], result["n_candidate_rows"], t["candidates_s"], t["scoring_s"],
             t["decoding_s"], result["request_s"], " | MOCK" if result["mode"]["is_mock"] else "")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host=CFG["api"]["host"], port=int(CFG["api"]["port"]))
