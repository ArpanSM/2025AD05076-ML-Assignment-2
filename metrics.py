"""Shared evaluation metrics for training and the Streamlit app."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_class_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return a continuous positive-class score for ROC AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def calculate_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Calculate the six metrics required by the assignment."""
    predictions = model.predict(features)
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, positive_class_scores(model, features)),
        "Precision": precision_score(target, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(target, predictions, average="weighted", zero_division=0),
        "F1": f1_score(target, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }