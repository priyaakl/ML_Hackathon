# PathSnap route reconstruction results

## Method

The decoder gives each GPS observation one candidate road arc as its hidden state. It uses the log of the classifier probability as the emission score. The transition penalty compares directed network distance with observed GPS displacement. The road graph uses limited Dijkstra search between arc midpoints. For consecutive observations assigned to the same arc, the decoder allows motion up to that arc's length. If the search cannot connect two candidates within the distance cap, the decoder applies a fixed penalty. Viterbi keeps the best preceding state for each candidate and traces backward to recover the route. It starts a new sequence after a gap over 30 seconds. Missing observations carry the last decoded arc forward, with `-1` used when there is no preceding prediction.

Person 1's code generated candidate arcs within 40 m, capped at 20 per observation. The logistic-regression artifact in the updated project snapshot supplied probabilities for the same candidate rows in all three comparisons. The classifier artifact has the same SHA-256 as the earlier model used for training-only tuning. We reran the student-test comparison on the updated snapshot. The experiment used scikit-learn 1.9.1, matching the saved model.

## Decoder tuning

We fixed the maximum network search distance at 250 m and the gap reset threshold at 30 s. We searched transition scales of 15, 30, and 60 m and disconnected-transition penalties of 4 and 8 across eight training trips sampled by trip length. We selected by pooled pointwise accuracy, with mean continuity as a tiebreaker. The resulting settings are 60 m and penalty 4. On the training subset they scored 96.50%, versus 95.89% for the original defaults. The classifier had been fit on these training trips, so the tuning scores are in-sample for the classifier.

## Student-test results

Both methods used the same GPS data, candidates, and model scores. The reference labels were used only for measurement. The ten trips contained 19,160 observations, of which 19,133 had a valid aligned reference label.

| Method | Top-1 accuracy | Macro F1 (arc-level) | Route continuity | Sequence similarity |
|---|---:|---:|---:|---:|
| Classifier only | 86.60% | 77.96% | 95.30% | 65.27% |
| Classifier + default Viterbi | 88.84% | 85.50% | 99.12% | 85.43% |
| Classifier + training-tuned Viterbi | 88.52% | 84.97% | 98.74% | 82.59% |

Pointwise accuracy pools correct road assignments across valid observations. Arc-level macro F1 is computed per trip and averaged across trips. Sequence similarity is one minus normalized Levenshtein distance after collapsing successive duplicate arcs in each valid-observation sequence. Continuity is the proportion of adjacent predictions that repeat an arc or can connect within two directed arc steps. Sequence and continuity figures are means across trips.

Training-tuned Viterbi recovered a net 367 correct point assignments and improved accuracy on nine of ten trips relative to classifier-only matching. The original defaults scored 0.32 percentage points higher than the training-tuned settings on student test. The configuration file records the training-selected settings, which were fixed before the formal student-test comparison.

Trip `00000007` shows a trade-off. Training-tuned Viterbi increased continuity from 99.70% to 100.00% and sequence similarity from 65.38% to 96.75%. Pointwise accuracy fell from 88.48% to 85.52%. Viterbi fixed four classifier errors but changed 44 previously correct observations, including a nine-observation run assigned to arc 282 instead of the aligned reference arc 285. A smoother route can still assign individual observations to the wrong segment.

![Training-tuned Viterbi on student-test trip 00000065](figures/hmm_student_test_00000065.png)

On trip `00000065`, training-tuned Viterbi improved accuracy from 83.85% to 86.33% and continuity from 95.31% to 99.35%. The image overlays the GPS observations, nearby roads, reference route and decoded arcs. The evaluator UI receives the predicted route and summary metrics.

## Reproduction and API handoff

Use the extracted student dataset and Person 2’s model artifacts. From the repository root:

```bash
PATHSNAP_DATA="/path/to/Student folder" python -m src.hmm.compare --tune --map-trip 00000065
```

On PowerShell, set `$env:PATHSNAP_DATA` first and run `python -m src.hmm.compare --tune --map-trip 00000065`. Omit `--tune` to re-evaluate the saved configuration. The script saves tuning trials, aggregate results, and per-trip results in `results/`. If Person 2 replaces the classifier, rerun the evaluation and tuning with the new artifact before using the results in the final report.

FastAPI already reads `models/hmm_config.json`. Person 4 can set both `api.use_mock_scorer` and `api.use_mock_decoder` to `false` and point `PATHSNAP_EVAL_DIR` at the supplied student-test folder to repeat the development smoke test. The local FastAPI test also used the API from the updated project snapshot. It returned successful prediction responses with one arc per observation, drawable route segments, and metrics for all ten student-test trips. The two-laptop UI and reserved evaluator trips remain for the team’s deployment test.
