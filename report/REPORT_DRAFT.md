# PathSnap: Technical Report (DRAFT)

> **Working draft assembled by Person 4.** Structure follows the professor's Deliverables document (plan.md Appendix A).
> Rules for everyone filling this in:
> - Every number must come from a real run and name its source file in `results/` (or the command that produced it).
> - State which partition (train / student_test) every analysis uses.
> - No fabricated metrics, graphs or citations. Write answers from your own experiments.
> - Results produced with the MOCK scorer are NOT reportable.
> - Put figures in `report/figures/` and screenshots in `report/screenshots/`.

**Team:** _names_  ·  **Course:** UE24CS352A Machine Learning, PES University  ·  **Repository:** https://github.com/priyaakl/ML_Hackathon

---

## 1. Overview *(Person 4)*

PathSnap reconstructs the road route behind a noisy GPS trajectory. For each GPS point, nearby road arcs are generated as candidates and described by features. A classifier scores each candidate, and an HMM with Viterbi decoding selects the most likely connected sequence of arcs. The system is served by a FastAPI backend (Laptop 1) and used through a Streamlit UI (Laptop 2).

_Pipeline diagram: to add._

---

## 2.2 Dataset Understanding and EDA *(Person 1)*

**Data used.** The dataset has 100 trips. We have 80 training trips and 10 student-test trips. The 10 evaluator trips are reserved and were never analysed. Unless stated otherwise, every number below is for both partitions, shown separately. Sources are in `results/` (`eda_summary.csv`, `dataset_inventory.csv`, `eda_network.csv`, `label_quality.csv`, `candidate_stats.csv`, `eda_gps_jumps_isolated.csv`); all are produced by `python -m src.data.*` scripts in the repo.

Each trip has four headerless files: `.track` (longitude, latitude, time), `.nodes` (road-network nodes, row index = node id), `.arcs` (directed road segments between nodes, row index = arc id) and `.route` (the reference arcs in travel order). Every trip has its own road network, and trips are in different cities.

| | Train | Student test |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS points | 208,417 | 19,160 |
| Median points per trip | 1,529 | 1,420 |

### (1) How did you preprocess the data?

1. **Reading and splitting.** Files are read by `src/common/io.py`. Trips are split by trip (80 / 10 / evaluator), never by row, and a check confirms no trip is in two partitions. Cross-validation later groups by `trip_id` for the same reason: rows of one trip are strongly correlated.
2. **Projection to metres.** Longitude and latitude are angles, so Euclidean distance in degrees is wrong. Each trip is projected into its own local metric frame (equirectangular projection around the trip's mean position, `src/common/geo.py`). We used one frame per trip because trips are in different cities. This is accurate for city-sized trips. For the longest trips (up to about 75 km north to south) the east-west scale error at the edges is, by our rough estimate, up to about 1%, which we did not measure.
3. **Missing values and timestamps.** We found no missing values and no decreasing timestamps in either partition. 96.5% (train) and 98.62% (student test) of time steps are exactly 1 s. There are 69 gaps above 60 s in 22 train trips and 1 in student test. The longest gap is 153,064 s (about 42 hours), in train trip 00000090. Gaps are not removed. The time-gap features expose them to the model, and the decoder starts a new segment after a gap above 30 s.
4. **GPS jumps.** We counted steps implying a speed above 50 m/s (180 km/h) as potential jumps: 1,523 of 208,337 train steps (0.73%, in 39 trips) and 8 of 19,150 student-test steps. Two trips (00000032 and 00000001) hold 1,227 of the train ones. No step exceeds 200 m within 5 s. We did not establish the cause, and we did not remove these points.
5. **Road network.** There are no isolated nodes (a node used by no arc). Networks are nevertheless not fully connected: 79 of 80 train networks and all 10 student-test networks have more than one connected component. The largest component holds a median 98.1% (train) and 97.05% (student test) of the nodes. A median 95.15% (train) and 93.25% (student test) of arcs have an opposite-direction twin, which matters for the classifier (see below).
6. **Per-point labels.** `.route` lists the arcs of the whole trip, not one arc per GPS point, so we built labels ourselves (`src/data/labels.py`). For each trip, we compute the distance from every GPS point to every route arc. A dynamic-programming step then gives each point a route position that never moves backwards as time advances, with minimal total distance. The assigned arc is the point's reference label.
7. **Label validity.** The reference route is reference information, not perfect truth. A label is marked `label_valid` only if the point is within 50 m of its assigned arc. The threshold is a choice and was not tuned. Valid labels: 185,162 of 208,417 train points (88.8%) and 19,133 of 19,160 student-test points (99.9%). Two training trips have broken reference routes: 00000054 (89.44% of points invalid) and 00000033 (58.86%). Three more are partly mismatched: 00000002 (11.98%), 00000066 (9.02%) and 00000043 (7.11%). Ten train trips and one student-test trip also contain breaks in the route itself (consecutive route arcs that are not connected). Invalid points are kept in the data, because the decoder needs every GPS point, but are excluded from training and evaluation.
8. **Candidate generation.** For every GPS point we keep every arc within 40 m, nearest first, at most 20 (`src/data/candidates.py`). A study on all 90 trips (`results/candidate_cap_study.csv`) set these values: the candidate miss rate, meaning the share of valid points whose true arc is not a candidate, is 0.64% on train and 0.084% on student test. Hence no classifier can exceed 99.36% (train) or 99.92% (student test) Top-1 accuracy. This gives 1,359,888 candidate rows on train (6.52 per point) and 149,328 on student test (7.79 per point). 18,841 train points and 10 student-test points have no candidate within 40 m.
9. **Reproducibility.** The same function that builds the table also serves the live API from raw files only, with no label file. We checked on all 10 student-test trips that it reproduces the batch table (`src/data/test_build_candidates.py`).

### (2) What patterns did you observe, what features did you extract, and why?

**Pattern 1: most roads are two arcs, one per direction, at the same distance.** Among valid points where the true arc and its opposite twin are both candidates (128,713 on train, 14,440 on student test), the median distance difference between the two is 0.0 m. Distance therefore cannot choose between them, and the nearest-arc baseline reaches only about 54% Top-1 (55.47% on train, 53.5% on student test; part of this is a coin flip between twins, and the exact figure varies slightly between machines).

**Pattern 2: direction separates the twins.** Among those points with valid heading (120,316 train, 13,437 student test), the true arc has the higher `cos_heading` in 96.03% of train points and 94.74% of student-test points (`results/feature_twin_stats.csv`).

**Pattern 3: classes are imbalanced.** Only 13.87% of train candidate rows (12.81% on student test) are the true arc. Always answering "not the true arc" would score about 86% per row, so we evaluate per GPS point (Top-1), with macro F1 and log loss, and state the definition each time.

**Features.** Each (GPS point, candidate arc) row has 20 features (`models/feature_columns.json`), in five groups. Full reasoning per feature is in `results/feature_rationale.md`.

| Group | Features | Why |
|---|---|---|
| Geometry | `dist_to_arc_m`, `dist_minus_best_m`, `dist_rank`, `n_cands` | The true road is usually close. Relative measures compare candidates fairly, because GPS noise differs by trip. |
| Position on arc | `arc_len_m`, `offset_along_m`, `dist_to_arc_end_m` | Where along the arc the point sits. Short arcs and arc ends behave differently. |
| Network | `arc_has_reverse`, `arc_in_deg`, `arc_out_deg` | Two-way roads and junction complexity. |
| Direction | `heading_valid`, `heading_diff_deg`, `cos_heading`, `progress_m`, `progress_ratio` | Resolve the twin ambiguity (pattern 2). |
| Motion and timing | `speed_mps`, `dist_prev_m`, `dist_next_m`, `dt_prev_s`, `dt_next_s` | Speed and gaps decide how far heading can be trusted. Fit to earlier and later points is cheap context. |

Heading and speed use all GPS points within +-5 s. If the vehicle moved less than 8 m in that window, heading is marked invalid and set to neutral values. Both settings are our choices and were not tuned.

**Single-feature strength** (train, valid rows; `results/feature_stats.csv`; AUC strength = max(AUC, 1 - AUC)). `dist_minus_best_m` (0.909) and `dist_rank` (0.907) are strongest alone. `dist_to_arc_m`, `heading_diff_deg`, `cos_heading` and `progress_ratio` reach 0.860. The weakest are `heading_valid` (0.513) and `dt_prev_s` / `dt_next_s` (0.501). This is a one-feature view and is not model feature importance. Model-based importance is reported with the models in section 3.

**No leakage.** Features use only the GPS track and the road network. The reference route is read only to build the label. At prediction time, the features are built from raw `.track/.nodes/.arcs` files. Neighbouring GPS points are used legitimately, because the task is offline map-matching and the whole track is available.

### Figures and maps for this section *(Person 1 to insert)*

| Content | Partition | File |
|---|---|---|
| GPS points per trip | train, student test | `results/figures/fig_trip_lengths.png` |
| Time-step distribution and longest gaps | train, student test | `results/figures/fig_time_gaps.png` |
| Vehicle speed distribution | train, student test | `results/figures/fig_speed.png` |
| Distance to reference arc, invalid labels per trip | train, student test | `results/figures/fig_label_distance.png` |
| Candidates per point and miss rate per trip | train, student test | `results/figures/fig_candidates.png` |
| Candidate radius and cap study | train | `results/figures/fig_cap_study.png` |
| Folium map: GPS, network, reference route (trip 00000001) | train | screenshot to be taken |
| Folium map of a broken reference (trip 00000054) | train | screenshot to be taken |

### Limitations
The reference route is not perfect truth (two training trips are unreliable). The 50 m label limit, the 40 m / 20 candidate settings, the +-5 s window and the 8 m movement threshold were set from the data and were not optimised against model results. The nearest-arc baseline depends on tie-breaking between twin arcs. We did not investigate the cause of the high-speed GPS steps.

---

## 3. Model Experimentation *(Person 2)*

### 3.1 Overall model comparison

Source: `results/experiments.csv`

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time |
|---|---|---|---|---|---|---|---|---|
| Model 1 | | | | | | | | |
| Model 2 | | | | | | | | |
| Model 3 | | | | | | | | |
| Model 4 | | | | | | | | |
| Model 5 | | | | | | | | |

Metric definitions used in this table _(Person 2 to confirm)_: Top-1 = observation level over valid labels; Macro F1 = candidate-level, threshold 0.5, mean of F1(class 0) and F1(class 1).

### 3.2 Per-model questions (one subsection per model)

#### Model N: _name_
- **A1.** Why did you consider this algorithm suitable for PathSnap?
- **A2.** Major strengths and limitations for this task?
- **B5.** Important hyperparameters and their initial values?
- **B6.** Loss function or optimisation objective?
- **C7.** Performance on the student-testing dataset?
- **C8.** Observation-level Top-1, Macro F1 and Log Loss on student-test?
- **C9.** Overfitting or underfitting? Evidence?

_(The professor's list is numbered 1, 2, 5, 6, 7, 8, 9. Questions 3 and 4 were not in our copy; asked the TA: _answer_.)_

### 3.3 Visualisations
- [ ] CV macro F1 comparison across all models
- [ ] Student-test Top-1 and macro F1 comparison
- [ ] Log loss comparison
- [ ] Training/validation curves ONLY for models with real epoch histories
- [ ] Plots justifying the final model choice

---

## 4. Bayesian Hyperparameter Optimisation *(Person 2)*

1. Hyperparameters searched: _TODO_
2. Ranges / distributions: _TODO_
3. Objective function: _TODO_
4. Number of trials: _TODO_
5. Best values: _TODO_
6. Did the tuned model beat the untuned one? _TODO_

| | CV Macro F1 | Student-Test Top-1 | Student-Test Macro F1 | Log Loss | Inference Time |
|---|---|---|---|---|---|
| Untuned | | | | | |
| Tuned | | | | | |

---

## 5. HMM and Viterbi Route Reconstruction *(Person 3)*

1. **Hidden states.** A state is a candidate directed road segment for one GPS observation. Person 1's pipeline supplies up to 20 candidates within 40 m.
2. **Emission score.** The decoder uses the log of the classifier's probability for each GPS–road pair.
3. **Transition score.** It compares the straight-line movement between GPS readings with directed road-network distance from a bounded Dijkstra search. A larger mismatch receives a larger penalty. Viterbi combines both scores across observations and traces back the best sequence.
4. **Missing candidates and GPS gaps.** An observation with no candidates carries the previous decoded arc forward, when available. The first missing observations use `-1`. A GPS gap over 30 seconds starts a new sequence.
5. **Disconnected segments.** If the search fails to find a route within its cap, the decoder applies a finite penalty. The network search is capped at 250 m.

We compared transition scales of 15, 30, and 60 m and disconnected-transition penalties of 4 and 8 on eight training trips. The best pooled training accuracy came from 60 m and a penalty of 4. The deployed decoder configuration records these training-selected settings. The classifier was fitted on these training trips. Source [hmm_tuning_summary.csv](../results/hmm_tuning_summary.csv).

### Student-test comparison

The classifier was the logistic-regression model supplied with this project snapshot. The three methods used identical candidate scores across ten student-test trips, with 19,133 valid reference assignments out of 19,160 GPS observations.

| Method | Top-1 accuracy | Macro F1 (arc-level) | Route continuity | Sequence similarity |
|---|---:|---:|---:|---:|
| Classifier only | 86.60% | 77.96% | 95.30% | 65.27% |
| Classifier + default Viterbi | 88.84% | 85.50% | 99.12% | 85.43% |
| Classifier + training-tuned Viterbi | 88.52% | 84.97% | 98.74% | 82.59% |

Top-1 pools matches across valid GPS observations. Arc-level macro F1 averages F1 across road segments within each trip, then averages the trip scores. Route continuity measures repeated arcs or directed connections within two steps. Sequence similarity is one minus normalized edit distance after collapsing consecutive duplicate arc IDs. Continuity and sequence similarity are averaged across trips. Source [hmm_student_test_summary.csv](../results/hmm_student_test_summary.csv).

Training-tuned Viterbi improved pointwise accuracy on nine of ten trips compared with classifier-only matching. The original Viterbi settings scored slightly higher overall on student test. On trip `00000007`, tuned decoding improved continuity from 99.70% to 100.00% while pointwise accuracy fell from 88.48% to 85.52%. It repaired four classifier misses and introduced 44 wrong assignments. Source [hmm_student_test_by_trip.csv](../results/hmm_student_test_by_trip.csv).

---

## 6. Geographical Visualisation Using Folium *(Person 3)*

The map for student-test trip `00000065` shows raw GPS points in blue, nearby roads in grey, the reference route in green, and the Viterbi prediction in dashed orange. The interactive Folium map includes layer controls and zoom. The figure below is a static export of the same four layers. The reference layer is included for this report figure. The evaluator UI receives the predicted route and aggregate metrics.

![Road network, reference route, GPS observations and Viterbi prediction for student-test trip 00000065](figures/hmm_student_test_00000065.png)

The interactive file is generated under `results/maps/` by running `python -m src.hmm.compare --map-trip 00000065` with the student dataset configured. Map generation is implemented in `src/viz/folium_maps.py`.

---

## 7. API, UI and Deployment *(Person 4)*

### 7.1 Architecture

Two laptops on the same local network:

- **Laptop 1 (server):** FastAPI backend (`src/api/main.py`) runs the full pipeline: candidate generation (Person 1), classifier scoring (Person 2's `CandidateScorer`), HMM/Viterbi decoding (Person 3's `decode_trip`) and server-side evaluation (`src/api/evaluator.py`). It holds the dataset, model and reference routes.
- **Laptop 2 (client):** web UI (`src/ui/web/index.html`, served by `src/ui/serve.py`) with a Leaflet map. It holds no data and no model, and talks to Laptop 1 only over HTTP from the browser (CORS enabled on the API).

### 7.2 Backend

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check; reports the loaded scorer, decoder and decoder-parameter source |
| `GET /trips` | Trip ids in the configured evaluation folder |
| `POST /predict` | Body `{"trip_id": ...}`. Runs the pipeline, then evaluates. Returns predicted arcs (HMM and classifier-only), GPS points, route geometry, per-stage timing and summary metrics |

Design decisions:
- **Load once at startup:** the model is loaded and every trip's road graph is built when the server starts, not on each request.
- **Configuration, not hardcoding:** host, port, CORS origins, the trips folder and the UI's API address come from `config.yaml`, with environment-variable overrides (`PATHSNAP_EVAL_DIR`, `PATHSNAP_API_URL`).
- **Request logging:** each request logs the caller's IP, the trip id, the number of GPS points and candidates, time per stage, Top-1 (HMM vs classifier-only) and total time.
- **Validation and errors:** request and response schemas are enforced with Pydantic. An unknown trip returns 404, a malformed body 422, and a pipeline failure 500 with the error logged. If evaluation fails, the route is still returned.
- **Same candidate settings as training:** the live API calls `build_candidates_for_trip` with the radius and cap from `config.yaml`, so features match training.

### 7.3 Ground truth stays server-side
The prediction pipeline (`src/api/pipeline.py`) reads only `.track`, `.nodes` and `.arcs`. After prediction, `src/api/evaluator.py` reads the `.route` file on Laptop 1, aligns it to GPS points with Person 1's `label_trip()` (the same labelling used for training), excludes unreliable labels (`label_valid == False`) and returns only summary metrics. Reference arcs are never sent to the UI.

### 7.4 UI
_Describe and show:_ backend connection status (scorer and decoder in use), trip selector, Match route button with live elapsed-time indicator, Top-1 headline, HMM vs classifier-only bars for Top-1, macro F1 (arc-level) and route continuity, per-stage timing on Laptop 1, a log of every API request the page made, and an interactive map (predicted route over the GPS points, no reference layer).

### 7.5 Deployment test
- Network used: _hotspot / Wi-Fi_
- Laptop 1 IP: _x.x.x.x_ · Port: 8000 · Laptop 2 OS: _..._
- Firewall settings: _..._
- Result: all 10 trips predicted from Laptop 2. Per-trip request time: _from results/api_eval_student_test.csv / logs_

### 7.6 Screenshots *(final model only, MOCK banner must NOT be visible)*
- [ ] UI on Laptop 2 after a prediction (map + metrics)
- [ ] Laptop 1 terminal showing request logs from Laptop 2's IP
- [ ] Successful `/predict` response on the receiving laptop's terminal

---

## AI Assistance Statement *(all)*

_TODO: honest description of how AI tools were used, per the event policy. Format confirmed with TA: _answer_._