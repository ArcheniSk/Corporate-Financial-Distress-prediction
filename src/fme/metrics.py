"""Imbalanced classification metrics and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve, recall_score, roc_auc_score


def choose_recall_threshold(y_true: np.ndarray, scores: np.ndarray, recall_target: float = 0.8) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.flatnonzero(recall[:-1] >= recall_target)
    if not len(valid):
        return float(0.0)
    best = valid[np.argmax(precision[valid])]
    return float(thresholds[best])


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "pr_auc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "brier": float(brier_score_loss(y_true, scores)),
        "threshold": float(threshold),
        "precision": float(matrix[1, 1] / max(matrix[:, 1].sum(), 1)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }
