"""
ML utility functions: training, saving/loading, inference, metrics, and slice metrics.
This version matches the Starter Kit layout and can drop-in replace ml/model.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# expected categorical columns in the Adult/Census dataset
CATEGORICAL_FEATURES: List[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
TARGET = "salary"  # "<=50K" or ">50K" (strings) or 0/1

@dataclass
class Artifacts:
    model: Pipeline
    categorical_features: List[str] | None = None
    def save(self, outdir: str) -> None:
        joblib.dump(self.model, f"{outdir}/model.joblib")
        meta = {"categorical_features": self.categorical_features or CATEGORICAL_FEATURES, "target": TARGET}
        joblib.dump(meta, f"{outdir}/meta.joblib")
    @staticmethod
    def load(outdir: str) -> "Artifacts":
        model = joblib.load(f"{outdir}/model.joblib")
        meta = joblib.load(f"{outdir}/meta.joblib")
        return Artifacts(model=model, categorical_features=meta["categorical_features"])

def _build_pipeline(cat_features: List[str]) -> Pipeline:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer([("cat", enc, cat_features)], remainder="passthrough")
    clf = LogisticRegression(max_iter=1000)
    return Pipeline([("pre", pre), ("clf", clf)])

def train_model(X_train: pd.DataFrame, y_train: pd.Series, categorical_features: List[str] | None = None) -> Pipeline:
    cats = categorical_features or CATEGORICAL_FEATURES
    pipe = _build_pipeline(cats)
    pipe.fit(X_train, y_train)
    return pipe

def inference(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)

def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Return precision, recall, f1 (beta=1)."""
    def to01(a):
        a = np.asarray(a)
        if a.dtype.kind in {"i", "u", "b"}:
            return a
        return np.array([1 if str(v).strip().startswith(">") or str(v) in {"1","True"} else 0 for v in a])
    yt, yp = to01(y_true), to01(y_pred)
    precision = precision_score(yt, yp, zero_division=0)
    recall = recall_score(yt, yp, zero_division=0)
    f1 = fbeta_score(yt, yp, beta=1.0, zero_division=0)
    return precision, recall, f1

def performance_on_categorical_slice(model: Pipeline, data: pd.DataFrame, feature: str, target_col: str = TARGET) -> Dict[str, Tuple[float, float, float]]:
    """Compute metrics for each unique value of a categorical feature; returns {value: (p,r,f1)}"""
    if feature not in data.columns:
        raise ValueError(f"Feature '{feature}' not found.")
    out: Dict[str, Tuple[float, float, float]] = {}
    for val, df_s in data.groupby(feature):
        Xs = df_s.drop(columns=[target_col])
        ys = df_s[target_col]
        preds = inference(model, Xs)
        out[str(val)] = compute_model_metrics(ys, preds)
    return out
