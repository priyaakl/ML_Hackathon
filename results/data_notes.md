# Data notes: quality issues, assumptions and decisions (Person 1)

Every number names its source. Partition is stated for each item. The 10 evaluator trips are not in our dataset and were never analysed.
Sources: `results/eda_summary.csv`, `results/dataset_inventory.csv`, `results/eda_network.csv`, `results/label_quality.csv`,
`results/candidate_stats.csv`, `results/candidate_cap_study.csv`, `results/eda_gps_jumps_isolated.csv`.

## 1. Data-quality findings

| # | Finding | Train (80 trips) | Student test (10 trips) | Source |
|---|---|---|---|---|
| 1 | Missing values in any file | 0 | 0 | eda_summary |
| 2 | Trips with non-increasing timestamps | 0 | 0 | eda_summary |
| 3 | GPS steps exactly 1 s apart | 96.5% | 98.62% | eda_summary |
| 4 | Time gaps above 60 s | 69 gaps in 22 trips | 1 gap in 1 trip | eda_summary |
| 5 | Longest gap | 153,064 s (about 42 h), trip 00000090; next: 00000078 (52,810 s) | 210 s | eda_summary, dataset_inventory |
| 6 | **GPS speed jumps** (step implied speed above 50 m/s = 180 km/h, step dt <= 60 s) | 1,523 of 208,337 steps (0.73%), in 39 trips | 8 of 19,150 steps, in 3 trips | eda_gps_jumps_isolated |
| 7 | Steps longer than 200 m within 5 s | 0 | 0 | eda_gps_jumps_isolated |
| 8 | **Isolated nodes** (node used by no arc) | 0 | 0 | eda_gps_jumps_isolated |
| 9 | Road networks not fully connected | 79 of 80 trips | 10 of 10 | eda_summary |
| 10 | Largest connected component (median share of nodes) | 98.1% | 97.05% | eda_summary |
| 11 | Arcs with an opposite-direction twin (median per trip) | 95.15% | 93.25% | eda_summary |
| 12 | Trips whose reference route has a break (consecutive route arcs not connected) | 10 trips | 1 trip | eda_summary |
| 13 | Lowest share of connected route steps in any trip | 98.13% | 99.73% | eda_summary |

Notes on the new checks (6, 7, 8):
- **Jumps are concentrated.** Two trips hold most of them: 00000032 (887 steps) and 00000001 (340 steps), so 1,227 of the 1,523 train jumps.
  Only 7 trips have 10 or more. The fastest implied speeds are 141 m/s (00000013) and 132 m/s (00000043; its largest step is 638 m).
  We did not investigate the cause. Sustained speeds above 180 km/h could be a fast vehicle such as a motorway or train, or they could be GPS noise. Do not claim either in the report.
- **Check 7 shows no teleporting.** No step exceeds 200 m within 5 s, so large single-step jumps are not a problem at 1 Hz.
- **Isolated nodes: 0.** Every node appears in some arc. The disconnection (item 9) comes from separate components, not from orphan nodes.
- The 50 m/s threshold is our choice and was not tuned.

## 2. Reference route and labels

- `.route` is one arc sequence per trip, not one arc per GPS point. Labels come from a monotone alignment (`src/data/labels.py`): each point is assigned a route position that never moves backwards in time, minimising total point-to-arc distance.
- `label_valid` = the point is within **50 m** of its assigned arc. The threshold is a choice, not a tuned value. We did not measure how results change with it.
- Valid labels: **185,162 of 208,417 train points (88.8%)**, **19,133 of 19,160 student-test points (99.9%)** (candidate_stats).
- Invalid points: train 23,255, student test 27 (label_quality). No student-test trip has more than 0.68% invalid.
- **Broken reference routes (train):** 00000054 (89.44% invalid, 19,313 points, median distance to its assigned arc 140 m) and 00000033 (58.86% invalid, median 122 m). Partly mismatched: 00000002 (11.98%), 00000066 (9.02%), 00000043 (7.11%).
- These trips are kept in the files but excluded from training and metrics through `label_valid`. The decoder still receives every GPS point.
- Reference routes are reference information, not perfect truth. The route-break trips (item 12) and the mismatched trips above are the evidence.

## 3. Assumptions and decisions

| Decision | Value | Why | Tuned? |
|---|---|---|---|
| Split | 80 / 10 / evaluator (not in our zip) by trip; no trip in two partitions | Avoid leakage | n/a |
| Projection | Local equirectangular frame around each trip's mean lon/lat (`src/common/geo.py`) | Trips are in different cities; one global frame does not fit | n/a |
| Candidate radius and cap | 40 m, at most 20 per point | Study on all 90 trips: miss rate 0.64% (train) and 0.084% (student test) at 1.33M valid train rows | Measured, not tuned against model results |
| Label validity | 50 m | Separates normal noise (99th percentile about 21 to 25 m) from real mismatches | No |
| Motion window | +-5 s; heading valid if displacement >= 8 m | Single steps are too noisy for heading; long windows smear turns | No |
| Scaling | None in preprocessing | Models that need scaling fit it on training folds only (Person 2) | n/a |
| Feature leakage | Features use only track and network; `.route` is read only to build `label` | Same function serves the live API | n/a |

## 4. Known limitations (state these in the report)

1. **Twin arcs and the nearest-arc baseline.** Most roads exist as two arcs, one per direction, at exactly the same distance from a point. Which twin ranks first is decided by floating-point noise, so the nearest-arc Top-1 baseline moves between machines
   (53.47% on one, 53.51% on another for student test). Quote it as about 54%, not as a precise figure. Candidate sets, row counts and miss rates do not depend on this. Only `cand_rank` and `dist_rank` can swap between twins.
2. **Candidate ceiling.** No classifier can beat 100 minus the miss rate: 99.36% (train), 99.92% (student test). 641 of the 1,185 train misses come from trips 00000033 and 00000054.
3. **Points with no candidate.** 18,841 train points have none within 40 m (16,587 in trip 00000054); student test has 10. They stay in the data and the decoder handles them.
4. **Per-trip projection distortion.** The longest trip spans about 75 km north to south (dataset_inventory, `extent_y_m` of train trips). A rough estimate puts the east-west scale error at the trip edges at up to about 1 percent there, and much less for typical trips.
   This is our own estimate, not a measurement. The docstring in `geo.py` says "well under a metre", which holds for city-sized trips only.
5. **Long time gaps.** Time features (`dt_prev_s`, `dt_next_s`) expose them, and the decoder starts a new segment after 30 s.
6. **Heading is unreliable when slow.** Heading is valid for about 91% of train rows. Otherwise it is set to neutral values and `heading_valid` = 0.
