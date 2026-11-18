### 3.1 Overall model comparison

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time (s per trip) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear (logistic regression) | C=1.0, max_iter=1000 | 0.913 ± 0.005 | 0.866 | 0.923 | 0.908 | 0.1051 | 0.0012 |
| LightGBM | Gradient boosting (LightGBM) | n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0 | 0.935 ± 0.011 | 0.901 | 0.931 | 0.940 | 0.0734 | 0.0232 |
| Random Forest | Bagging ensemble (random forest) | n_estimators=100, max_depth=14, min_samples_leaf=5, max_samples=0.3 | 0.936 ± 0.010 | 0.892 | 0.931 | 0.937 | 0.0705 | 0.0133 |
| MLP | Neural network (shallow MLP) | hidden_layer_sizes=[64, 32], alpha=0.0001, learning_rate_init=0.001, batch_size=512, max_iter=40, early_stopping=True, n_iter_no_change=5 | 0.930 ± 0.009 | 0.887 | 0.936 | 0.933 | 0.0832 | 0.0180 |
| Gaussian Naive Bayes | Probabilistic (Gaussian naive Bayes) | library defaults | 0.850 ± 0.010 | 0.831 | 0.917 | 0.838 | 0.5741 | 0.0022 |
| KNN | Instance-based (distance-weighted KNN) | n_neighbors=25, weights=distance, max_train_rows=250000 | 0.912 ± 0.007 | 0.869 | 0.924 | 0.913 | 0.1333 | 2.0951 |

Partitions: CV columns = 5-fold GroupKFold over the 80 train trips (valid-label rows only). Top-1 Accuracy, Route continuity, Student-Test Macro F1, Log Loss and Inference Time = the 10 student_test trips (never used for fitting). Top-1 = per GPS point, points whose true arc is not a candidate count as wrong. Macro F1 = candidate-level, threshold 0.5. Route continuity = share of consecutive predicted arcs that are the same or directly connected (classifier only, no HMM).

#### Extended columns (precision, recall, train score, cost)

| Model | CV Top-1 | CV log loss | Student precision | Student recall | Train Top-1 | Train time (s) | Seed | CV folds |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.881 ± 0.013 | 0.117 ± 0.016 | 0.834 | 0.844 | 0.888 | 3.1 | 42 | 5 |
| LightGBM | 0.890 ± 0.017 | 0.079 ± 0.018 | 0.906 | 0.884 | 0.932 | 11.6 | 42 | 5 |
| Random Forest | 0.889 ± 0.016 | 0.077 ± 0.015 | 0.915 | 0.866 | 0.923 | 31.8 | 42 | 5 |
| MLP | 0.882 ± 0.017 | 0.093 ± 0.021 | 0.890 | 0.878 | 0.921 | 70.5 | 42 | 5 |
| Gaussian Naive Bayes | 0.860 ± 0.013 | 0.521 ± 0.085 | 0.622 | 0.873 | 0.871 | 0.4 | 42 | 5 |
| KNN | 0.867 ± 0.010 | 0.152 ± 0.034 | 0.854 | 0.842 |  | 0.1 | 42 | 5 |
