# PathSnap: ML-based GPS Map-Matching

PathSnap takes noisy GPS points recorded from a moving vehicle and reconstructs the road route the vehicle actually drove (map-matching). Every GPS point is matched to one road segment (arc) of the trip's road network.

Built for the PES University UE24CS352A Machine Learning hackathon.

---

## How it works

```
 raw trip files (.track .nodes .arcs)
        │
        ▼
 1. Candidate generation      every road arc within 40 m of each GPS point        src/data/candidates.py
 2. Feature engineering       20 features per (GPS point, candidate arc) pair      src/data/candidates.py
 3. ML classifier             probability that each candidate is the true road     src/models/scorer.py
 4. HMM + Viterbi decoding    most likely CONNECTED sequence of roads              src/hmm/viterbi.py
        │
        ▼
 FastAPI backend (Laptop 1)  ◀── HTTP ──▶  Web UI (Laptop 2)
 runs 1-4 + evaluation                      trip picker, map, metrics
```

Ground truth (`.route`) is only read **after** prediction, by the server-side evaluator (`src/api/evaluator.py`). It never enters the prediction pipeline or the UI.

---

## Technologies

| Purpose | Libraries |
|---|---|
| Data processing | pandas, NumPy, SciPy, PyArrow |
| Geometry and road graph | Shapely (spatial index), SciPy sparse graphs |
| Models and tuning | scikit-learn, LightGBM, XGBoost, Optuna (Bayesian optimisation) |
| Maps and plots | Folium, Matplotlib, Seaborn |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML/JavaScript with Leaflet (served by Python), Streamlit fallback |

---

## Repository structure

```
ML_Hackathon/
  README.md
  requirements.txt
  config.yaml                 shared settings: data path, seed, candidate radius, API host/port, UI API URL
  data/
    raw/                      (unused placeholder, the dataset lives outside the repo, see below)
    processed/                splits.json, labels_*.parquet, no_candidate_obs.csv
  src/
    common/                   config loader, raw-file readers, projection helpers
    data/                     inventory, labels, candidates + features, EDA          (Person 1)
    models/                   harness, model zoo, experiments, Bayesian opt, scorer   (Person 2)
    hmm/                      road graph, Viterbi decoder                             (Person 3)
    viz/                      Folium route maps, label-check maps                     (Persons 1, 3)
    api/                      FastAPI app, prediction pipeline, evaluator             (Person 4)
    ui/                       web UI (web/index.html + serve.py), Streamlit fallback  (Person 4)
  models/                     final_model.joblib, feature_columns.json, hmm_config.json
  results/                    experiment logs, CV folds, EDA tables and figures
  report/                     report draft and screenshots
```

---

## Installation

Requires **Python 3.10 or 3.11**.

```bash
git clone https://github.com/priyaakl/ML_Hackathon.git
cd ML_Hackathon
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Apple Silicon Macs, if LightGBM reports an OpenMP error: `brew install libomp`.

---

## Dataset placement

The dataset is **not** included in this repository. Unzip it **outside** the repo. The default location is `~/Downloads/Student folder`:

```
Student folder/
  train/            80 trips
    <trip_id>/
      <trip_id>.track   lon, lat, timestamp (s)
      <trip_id>.nodes   lon, lat            (row index = node id)
      <trip_id>.arcs    from_node, to_node  (row index = arc id)
      <trip_id>.route   reference arc ids in travel order
  student_test/     10 trips, same layout
```

To use another location, either edit `paths.data_root` in `config.yaml` or set:

```bash
export PATHSNAP_DATA="/full/path/to/Student folder"
```

The 10 evaluator-testing trips are not part of this dataset. They are supplied at evaluation time (see *Evaluation day* below).

---

## Reproducing the ML pipeline

Run every command from the repo root, with the virtual environment active.

```bash
# Person 1: data
python -m src.data.inventory                  # dataset inventory -> results/dataset_inventory.csv
python -m src.data.labels                     # per-GPS-point reference labels
python -m src.data.candidates                 # candidate tables + models/feature_columns.json
python -m src.data.eda                        # EDA tables and figures -> results/
python -m src.viz.label_maps                  # label-check Folium maps

# Person 2: models
python -m src.models.selftest                 # harness check on synthetic data
python -m src.models.run_experiments --models lr lgbm         # group-aware CV + student-test metrics
python -m src.models.bayes_opt --n-trials 30                  # Optuna (TPE) on the chosen model
python -m src.models.train_final --model lr                   # export models/final_model.joblib

# Person 3: route maps
python -m src.viz.folium_maps <path/to/Student_folder.zip> <trip_id> results/maps/<trip_id>.html --image report/<trip_id>.png
```

Experiment results are logged to `results/experiments.csv`.

---

## Running the application

### 1. Backend (Laptop 1)

```bash
python -m src.api.main
```

This starts FastAPI on `0.0.0.0:8000` (host and port come from `config.yaml`). At startup it loads the model once and pre-builds every trip's road graph. Every request is logged on this terminal with the caller's IP, the trip id and timings.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check, plus which scorer and decoder are loaded |
| GET | `/trips` | Trip ids available for evaluation |
| POST | `/predict` | Body `{"trip_id": "..."}`. Returns the predicted route, GPS points, timing and metrics |

Interactive API docs: `http://<host>:8000/docs`

Quick test:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"trip_id": "00000009"}'
```

### 2. Frontend (Laptop 2, or the same laptop)

```bash
export PATHSNAP_API_URL="http://<LAPTOP-1-IP>:8000"     # Windows PowerShell: $env:PATHSNAP_API_URL="http://<IP>:8000"
python -m src.ui.serve
```

Opens the PathSnap UI at `http://localhost:8501`. It is a single web page (`src/ui/web/index.html`, Leaflet map) that calls the API from the browser; `src/ui/serve.py` only serves the page and tells it the API address. The address can also be changed in the page, or passed as `http://localhost:8501/?api=http://<IP>:8000`.

The UI machine needs only Python and `pyyaml`: no dataset, no model. Leaflet and the font are bundled in `src/ui/web/vendor/`, so only the map background tiles need internet.

A simpler Streamlit version is kept as a fallback: `streamlit run src/ui/app.py` (needs `streamlit streamlit-folium folium requests pyyaml`).

### 3. Batch evaluation of every trip (Laptop 1)

```bash
python -m src.api.evaluator
```

Runs predict and evaluate for every trip in the API's trips folder and saves `results/api_eval_<folder>.csv`. Metrics are Top-1 accuracy, macro F1 and route continuity, for both HMM and classifier-only.

---

## Two-laptop deployment

1. Connect both laptops to the **same network**. A phone hotspot is the most reliable option, because campus Wi-Fi often blocks device-to-device traffic.
2. **Laptop 1:** find its IP with `ipconfig getifaddr en0` (macOS), `ipconfig` (Windows) or `hostname -I` (Linux).
3. **Laptop 1:** start the backend with `python -m src.api.main`. Allow incoming connections for Python if the OS firewall asks (macOS: System Settings → Network → Firewall → Options).
4. **Laptop 2:** open `http://<LAPTOP-1-IP>:8000/health` in a browser. It should return `"status":"ok"`, and Laptop 1's terminal should log the request from Laptop 2's IP.
5. **Laptop 2:** start the UI with `PATHSNAP_API_URL` set to Laptop 1 (see above), pick a trip and press **Match route**.

| Problem | Fix |
|---|---|
| Request times out | Firewall on Laptop 1, or the network isolates clients. Allow Python in the firewall or use a hotspot |
| Connection refused | Backend not running, or not bound to `0.0.0.0` |
| Worked earlier, fails now | Laptop 1's IP changed. Check it again |

The map background tiles load from the internet, so Laptop 2 needs internet access for the basemap.

---

## Evaluation day

Point the API at the folder containing the evaluator's 10 trips (same per-trip folder layout), **without changing code**:

```bash
export PATHSNAP_EVAL_DIR="/full/path/to/evaluator_trips"
python -m src.api.main
```

Or edit `api.eval_trips_dir` in `config.yaml`. If a trip folder contains a `.route` file, the server computes metrics; the reference route itself is never sent to the UI.

---

## Configuration (`config.yaml`)

| Key | Meaning |
|---|---|
| `paths.data_root` | Dataset folder (override: `PATHSNAP_DATA`) |
| `candidates.radius_m`, `candidates.max_candidates` | Candidate search radius and cap (used for training **and** the live API) |
| `api.host`, `api.port` | Backend address (`0.0.0.0` = reachable from other laptops) |
| `api.cors_origins` | Origins allowed to call the API from a browser |
| `api.eval_trips_dir` | Trips offered by `/trips` (override: `PATHSNAP_EVAL_DIR`) |
| `api.use_mock_scorer`, `api.use_mock_decoder` | Development switches. When `true`, every response and the UI are labelled **MOCK** |
| `ui.api_base_url` | Where the UI sends requests (override: `PATHSNAP_API_URL`) |

---

## Data integrity

- Splits are by trip (80 train / 10 student-test); cross-validation is grouped by `trip_id`.
- Evaluator-testing trips are never used for training, tuning or feature selection.
- The prediction pipeline reads only `.track`, `.nodes` and `.arcs`. Reference routes are read only by the server-side evaluator, after prediction.
- No predictions are hardcoded.

---

## Team

| Person | Name | Responsibility |
|---|---|---|
| 1 | _TBD_ | Data, EDA, labels, candidate generation, features |
| 2 | _TBD_ | Model experiments, evaluation harness, Bayesian optimisation |
| 3 | _TBD_ | Road graph, HMM, Viterbi, Folium maps |
| 4 | _TBD_ | Repository, FastAPI, UI, integration, deployment, README, report |
