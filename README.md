# PathSnap: ML-based GPS Map-Matching

PathSnap takes noisy GPS points recorded from a moving vehicle and reconstructs the actual route the vehicle drove on the road network (map-matching).

Built for the PES University UE24CS352A Machine Learning hackathon.

> **Status:** work in progress. This README is updated as each component is completed. Commands marked *(planned)* do not work yet.

---

## How it works

1. **Candidate generation:** for each GPS point, find nearby road segments (arcs).
2. **Feature engineering:** compute features for each (GPS point, candidate arc) pair, such as distance and heading difference.
3. **ML classification:** a trained classifier scores how likely each candidate is the true road.
4. **HMM + Viterbi decoding:** picks the most likely *connected* sequence of road segments.
5. **Deployment:** a FastAPI backend (Laptop 1) runs the pipeline. A UI (Laptop 2) lets the evaluator pick a trip and see the predicted route and metrics.

---

## Technologies

- Python 3.10 / 3.11
- pandas, NumPy, scikit-learn, LightGBM / XGBoost: data processing and models
- Optuna: Bayesian hyperparameter optimisation
- pyproj, shapely, networkx: geospatial processing and the road graph
- FastAPI + Uvicorn: backend API
- Streamlit + Folium: frontend UI and interactive maps

---

## Repository structure

```
ML_Hackathon/
  README.md
  plan.md                 # team plan and progress log
  requirements.txt
  config.yaml             # paths, seeds, projection, API host/port
  data/
    raw/                  # dataset goes here (NOT committed)
    processed/            # candidate tables, splits.json
  src/
    common/               # config loader, file readers, geo helpers
    data/                 # EDA, candidates, features, labels
    models/               # training scripts, evaluation harness, Bayesian optimisation, CandidateScorer
    hmm/                  # road graph, Viterbi decoder, comparison
    viz/                  # Folium map helpers
    api/                  # FastAPI app, prediction pipeline, server-side evaluator
    ui/                   # frontend
  models/                 # saved model + configs
  results/                # experiment logs, Optuna trials, figures, maps
  report/                 # report assets
```

---

## Installation *(planned: requirements.txt is being added)*

```bash
git clone <repo-url>
cd ML_Hackathon
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset placement

The dataset is **not** included in this repository. Unzip it so the layout looks like this:

```
data/raw/
  train/            # 80 trips
    <trip_id>/
      <trip_id>.track   # lon, lat, timestamp
      <trip_id>.nodes   # lon, lat (row index = node id)
      <trip_id>.arcs    # from_node, to_node (row index = arc id)
      <trip_id>.route   # ground-truth sequence of arc ids
  student_test/     # 10 trips, same layout
```

The 10 evaluator-testing trips are not part of this dataset. They are supplied at evaluation time and kept server-side only.

---

## Running the application *(planned)*

### Backend: Laptop 1
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Endpoints: `GET /health`, `GET /trips`, `POST /predict`

### Frontend: Laptop 2
Set the API URL (Laptop 1's IP) in the config, then:
```bash
streamlit run src/ui/app.py
```

---

## Two-laptop deployment *(planned)*

Step-by-step instructions (finding the IP, firewall, testing `/health`) will be added after the two-laptop test.

---

## Team

| Person | Name | Responsibility |
|---|---|---|
| 1 | _TBD_ | Data, EDA, candidate generation, features, labels |
| 2 | _TBD_ | Model experiments, evaluation harness, Bayesian optimisation |
| 3 | _TBD_ | Road graph, HMM, Viterbi, Folium maps |
| 4 | _TBD_ | Repository, FastAPI, UI, deployment, README, report |
