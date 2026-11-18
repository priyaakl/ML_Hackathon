"""Person 2: the model contract (plan Section 5.3). Persons 3 and 4 call this class.

    scorer = CandidateScorer("models/final_model.joblib", "models/feature_columns.json")
    probs = scorer.predict_proba(cands_df)     # 1-D numpy array, same length and order as cands_df

Run from the repo root so that `src` is importable (the saved model references src.models.model_zoo).
"""
import json

import joblib
import numpy as np
import pandas as pd


class CandidateScorer:
    def __init__(self, model_path: str, feature_columns_path: str):
        self.model = joblib.load(model_path)
        with open(feature_columns_path) as f:
            self.feature_columns = json.load(f)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probability that each candidate row is the true arc (class 1)."""
        if len(df) == 0:
            return np.zeros(0, dtype=np.float64)
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"candidate table is missing feature columns: {missing}")
        X = df[self.feature_columns]
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype=np.float64)
