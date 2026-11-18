# Feature rationale (Person 1, task 1.5)

Code: `src/data/candidates.py` (`FEATURE_COLUMNS`, `build_candidates`). Column order: `models/feature_columns.json`.
Evidence: `results/feature_stats.csv` and `results/feature_twin_stats.csv` (produced by `python -m src.data.feature_stats`).

## 1. What a row is
One row = one GPS point x one candidate road arc (every arc within 40 m, nearest first, at most 20). The classifier answers: *given this point and this arc, is this the true arc?* Twenty features describe the pair. The label (`label`) is 1 if the arc is the reference arc for that point. It is never used as a feature.

## 2. The central problem the features are built around
Most roads are stored as two arcs, one per direction, at exactly the same distance from a GPS point (median 95% of arcs in train have an opposite twin). **Distance alone cannot tell them apart**, so the nearest-arc baseline is about 54% (partly a coin flip). Direction of travel can. So the features have two jobs: (a) say how plausible the arc is by position, and (b) say whether the vehicle is moving along the arc or against it.

## 3. The 20 features

| Group | Feature | How it is computed | Why it is there |
|---|---|---|---|
| Geometry | `dist_to_arc_m` | Point to the nearest point on the arc segment, metres (projected frame) | The true road is usually close. |
| | `dist_minus_best_m` | `dist_to_arc_m` minus the smallest distance among this point's candidates | Noise level differs between trips. A relative measure is comparable across trips. |
| | `dist_rank` | 1 = nearest candidate | Rank does not depend on the noise level. Ties between twins are broken by floating-point noise, which is why direction features are needed. |
| | `n_cands` | Number of candidates for this point | Many candidates means a dense junction or city. Tells the model how ambiguous the point is. |
| Position on arc | `arc_len_m` | Arc length | Short arcs sit at junctions and behave differently from long road sections. |
| | `offset_along_m` | Distance from arc start to the point's projection (clamped to the arc) | Where along the arc the vehicle is. |
| | `dist_to_arc_end_m` | Arc length minus the offset | Near the end the vehicle is about to enter the next arc. |
| Network | `arc_has_reverse` | 1 if the opposite-direction arc exists | Marks two-way roads, where direction matters. |
| | `arc_in_deg` | Arcs entering the arc's start node | Junction complexity at the start. |
| | `arc_out_deg` | Arcs leaving the arc's end node | Junction complexity at the end. |
| Direction | `heading_valid` | 1 if the vehicle moved at least 8 m across the +-5 s window | A parked or slow vehicle has no trustworthy heading. |
| | `heading_diff_deg` | Angle between vehicle bearing and arc bearing, 0 to 180; 90 if heading invalid | Separates a road from its opposite twin. 90 is the neutral value. |
| | `cos_heading` | Cosine of that angle (+1 same way, -1 opposite); 0 if invalid | Smooth version of the same signal, easier for linear models. |
| | `progress_m` | How far the window's end point moved along the arc direction compared with its start point (negative = against) | Direct evidence of travelling with or against the arc. |
| | `progress_ratio` | `progress_m` divided by the straight-line movement (at least 1 m); 0 if invalid | Same, normalised for speed. About +1 along, -1 against. |
| Motion and timing | `speed_mps` | Window displacement divided by window duration | Lets the model weigh heading differently when slow or fast. |
| | `dist_prev_m`, `dist_next_m` | Distance of the window's first and last GPS points to this arc | If the arc also fits the earlier and later points, it is more likely the true one. A cheap form of context. |
| | `dt_prev_s`, `dt_next_s` | Time to the previous and next GPS point | Flags gaps, where speed and heading are less reliable. |

Window = all GPS points within +-5 s (`WINDOW_S`). Minimum movement = 8 m (`MIN_MOVE_M`). Both are our choices and were **not tuned**.

## 4. Evidence from the data
Produced by `python -m src.data.feature_stats` on valid-label rows (`label_valid == True`): 1,326,757 train rows and 149,275 student-test rows.
- **Twin separation** (`results/feature_twin_stats.csv`): among valid points where the true arc and its opposite-direction twin are both candidates (128,713 on train, 14,440 on student test), the median distance difference between the two is **0.0 m**. Among those with valid heading (120,316 train, 13,437 student test), the true arc has the higher `cos_heading` in **96.03%** of points on train and **94.74%** on student test. The result is identical for `progress_ratio`, because the two features agree in sign. They are two views of the same signal, not independent evidence.
- **Heading valid** share of valid-label rows: **X% on train and Y% on student test** (`results/feature_stats.csv`, column `heading_valid_share_pct`; replace X and Y).
- **Single-feature AUC** (`results/feature_stats.csv`, train, valid rows; strength = max(AUC, 1 - AUC)): `dist_minus_best_m` (0.909) and `dist_rank` (0.907) are strongest alone. `dist_to_arc_m`, `heading_diff_deg`, `cos_heading` and `progress_ratio` all reach 0.860. `progress_m` is 0.830. The weakest are `arc_has_reverse` (0.528), `arc_in_deg` (0.523), `arc_out_deg` (0.522), `heading_valid` (0.513) and `dt_prev_s` / `dt_next_s` (0.501).
  Distance features score high alone but cannot split twins, because twins are the same distance away. Direction features split twins. Features near 0.5 alone may still help in combination, for example by telling the model *when* to trust heading.
  This is a one-feature view, **not** model feature importance. Real importance comes from Person 2's trained models (LightGBM and RF are available there).

## 5. No leakage
- Features use only the GPS track and the road network. The reference route is read only to build `label` and `label_valid`.
- At inference `build_candidates_for_trip` runs on `.track/.nodes/.arcs` only. We tested this on all 10 student_test trips with the `.route` file removed (`src/data/test_build_candidates.py`, 10 of 10 pass).
- Neighbouring GPS points (+-5 s, previous and next) are used. That is legitimate because map-matching here is offline: the whole track is available at prediction time.
- No scaler is fitted in preprocessing. Models that need scaling fit it on training folds only.

## 6. Limitations to state
1. Heading and progress are only reliable when the vehicle is moving (`heading_valid` = 1).
2. Window size (5 s) and the 8 m threshold are untuned choices.
3. Per-trip local projection is accurate for city-sized trips; for very long trips the scale error can reach roughly 1% at the edges (our estimate).
4. Feature strength shown above is per feature alone. Interactions were not studied by us.

## 7. Likely viva questions
- *Why do distance features fail on two-way roads?* The twin arcs are the same distance away. See section 2.
- *Why relative distance (`dist_minus_best_m`, `dist_rank`)?* GPS noise differs per trip.
- *Why +-5 s?* One step is too noisy for a direction, a long window smears turns. Chosen, not tuned.
- *What does `heading_valid` do?* It marks when heading can be trusted. Otherwise heading features are neutral (90 degrees, 0, 0).
- *Is using future GPS points cheating?* No. The task is offline: the whole track is available. The same function runs in the API.
- *Which feature matters most?* Answer from Person 2's model importances, not from the single-feature AUC.