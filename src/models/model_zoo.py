"""Person 2: the models. One function builds any of them from a short name.

Every model is wrapped so it exposes .fit(X, y) and .predict_proba(X) -> (n, 2).
Models that need scaling get a StandardScaler INSIDE a Pipeline, so the scaler is fit on training data only
(and refit inside every CV fold).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAMILY = {
    "lr": "Linear (logistic regression)",
    "rf": "Bagging ensemble (random forest)",
    "lgbm": "Gradient boosting (LightGBM)",
    "knn": "Instance-based (distance-weighted KNN)",
    "mlp": "Neural network (shallow MLP)",
    "gnb": "Probabilistic (Gaussian naive Bayes)",
}

# Initial hyperparameters (report question 5). Change them here, not inside the code below.
DEFAULTS = {
    "lr": dict(C=1.0, max_iter=1000),
    "rf": dict(n_estimators=100, max_depth=14, min_samples_leaf=5, max_samples=0.3, n_jobs=-1),
    "lgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0,
                 n_jobs=-1, verbose=-1),
    "knn": dict(n_neighbors=25, weights="distance", n_jobs=-1, max_train_rows=250_000),
    "mlp": dict(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, batch_size=512,
                max_iter=40, early_stopping=True, n_iter_no_change=5),
    "gnb": dict(),
}

# Loss / objective of each model (report question 6).
LOSS = {
    "lr": "log loss + L2 penalty",
    "rf": "Gini impurity (split criterion); probability = mean of tree votes",
    "lgbm": "binary log loss (gradient boosting)",
    "knn": "none (no training loss); probability = distance-weighted vote of neighbours",
    "mlp": "binary cross-entropy (log loss) + L2 penalty",
    "gnb": "none (class-conditional Gaussian likelihoods, Bayes rule)",
}


class Subsampled:
    """Fits the inner model on a random subset of rows (used for KNN to keep it fast).
    A uniform random sample keeps the positive/negative ratio, so probabilities stay on the right scale."""

    def __init__(self, est, max_rows, seed=42):
        self.est, self.max_rows, self.seed = est, max_rows, seed

    def fit(self, X, y):
        if len(X) > self.max_rows:
            idx = np.random.default_rng(self.seed).choice(len(X), size=self.max_rows, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]
        self.est.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)


def config_of(name, params=None):
    """The full config dict that gets logged for an experiment."""
    cfg = dict(DEFAULTS[name])
    cfg.update(params or {})
    return cfg


def build_model(name, seed=42, params=None):
    p = config_of(name, params)
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **p))
    if name == "rf":
        return RandomForestClassifier(random_state=seed, **p)
    if name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise SystemExit("lightgbm is not installed: pip install lightgbm") from e
        return LGBMClassifier(random_state=seed, **p)
    if name == "knn":
        rows = p.pop("max_train_rows")
        return Subsampled(make_pipeline(StandardScaler(), KNeighborsClassifier(**p)), rows, seed)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(random_state=seed, **p))
    if name == "gnb":
        return make_pipeline(StandardScaler(), GaussianNB(**p))
    raise ValueError(f"unknown model '{name}'. Choose from {list(DEFAULTS)}")
