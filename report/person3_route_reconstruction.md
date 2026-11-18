# PathSnap route reconstruction with LightGBM

For the decoder and gap-handling details, see [Sections 5 and 6 of the team report](REPORT_DRAFT.md#5-hmm-and-viterbi-route-reconstruction-person-3).

## Training-only decoder tuning

LightGBM fold classifiers used the family, features, and hyperparameters of the published model. We grouped the 80 training trips into 5 folds. For each of eight trips sampled by trip length, we generated classifier probabilities using a model fitted on the other four folds. We searched transition scales of 15, 30, and 60 m and disconnected-transition penalties of 4 and 8. Pooled top-1 selected the configuration, with mean continuity as a tiebreaker. The selected scale was 30 m and the penalty was 4. It scored 97.51% across 19,095 valid tuning observations. This is the score used for parameter selection. The student-test trips were held out of tuning, and the evaluator split was outside this workflow. Sources [hmm_tuning_summary.csv](../results/hmm_tuning_summary.csv) and [hmm_tuning_metadata.json](../results/hmm_tuning_metadata.json).

## Student-test comparison

We used the saved LightGBM scorer published in commit `d3d9747`. The three methods used the same scores across the 10 student-test trips, with 19,133 valid reference assignments out of 19,160 GPS observations.

| Method | Top-1 accuracy | Macro F1 (arc-level) | Route continuity | Sequence similarity |
|---|---:|---:|---:|---:|
| LightGBM | 90.11% | 83.17% | 95.72% | 70.10% |
| LightGBM + default Viterbi | 93.10% | 91.64% | 99.37% | 90.25% |
| LightGBM + retuned Viterbi | 92.89% | 91.67% | 99.30% | 90.31% |

Top-1 pools matches across valid GPS observations. Arc-level macro F1 averages F1 across road segments within each trip, then averages the trip scores. Route continuity measures repeated arcs or directed connections within two steps. Sequence similarity is one minus normalized edit distance after collapsing consecutive duplicate arc IDs. Continuity and sequence similarity are averaged across trips. Source [hmm_student_test_summary.csv](../results/hmm_student_test_summary.csv).

Retuned Viterbi raised pooled student-test top-1 by 2.78 percentage points over LightGBM alone, with gains on all 10 trips. Default Viterbi retained a 0.21-percentage-point advantage in student-test top-1. Source [hmm_student_test_by_trip.csv](../results/hmm_student_test_by_trip.csv).

![Student-test route reconstructed with LightGBM + Viterbi](figures/hmm_student_test_00000065.png)

## Reproduce

From the repository root, set `PATHSNAP_DATA` to the folder containing `train/` and `student_test/`. Run `python -m src.hmm.compare --tune --map-trip 00000065`. Omit `--tune` to reuse the existing HMM config. The earlier logistic-regression run was a baseline. The results here apply to the LightGBM artifact recorded in the tuning metadata.
