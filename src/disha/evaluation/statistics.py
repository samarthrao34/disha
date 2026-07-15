"""Reproducible classification statistics used by DISHA experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError("probabilities must be a two-dimensional row per target")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(
                float((prediction[selected] == y_true[selected]).mean())
                - float(confidence[selected].mean())
            )
    return float(result)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: list[str],
) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, prediction, average="weighted", zero_division=0)
        ),
        "negative_log_likelihood": float(
            log_loss(y_true, probabilities, labels=np.arange(len(labels)))
        ),
        "ece_15_bin": expected_calibration_error(y_true, probabilities, bins=15),
    }


def cluster_bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Percentile CI resampling whole dialogue/speaker clusters."""
    if not (len(y_true) == len(y_pred) == len(clusters)):
        raise ValueError("targets, predictions, and clusters must have equal length")
    unique = np.unique(clusters)
    indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[cluster] for cluster in sampled])
        scores[iteration] = metric(y_true[rows], y_pred[rows])
    return {
        "estimate": float(metric(y_true, y_pred)),
        "ci_95_low": float(np.percentile(scores, 2.5)),
        "ci_95_high": float(np.percentile(scores, 97.5)),
        "bootstrap_iterations": iterations,
        "cluster_count": int(len(unique)),
    }
