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