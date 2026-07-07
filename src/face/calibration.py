"""Calibration and uncertainty utilities for the DISHA face pipeline.

All functions operate on real model outputs. Nothing here produces or
assumes any pre-computed calibration numbers.
"""

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over ``axis``."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def entropy_from_probs(probs: np.ndarray, eps: float = 1e-12, normalize: bool = False) -> np.ndarray:
    """Shannon entropy (nats) of probability vectors.

    Args:
        probs: Array of shape (..., n_classes) with rows summing to 1.
        eps: Small constant for numerical stability.
        normalize: If True, divide by log(n_classes) so entropy is in [0, 1].

    Returns:
        Entropy per row, shape (...,).
    """
    probs = np.asarray(probs, dtype=np.float64)
    clipped = np.clip(probs, eps, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    if normalize:
        n_classes = probs.shape[-1]
        entropy = entropy / np.log(n_classes)
    return entropy


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width confidence bins.

    Args:
        probs: Array of shape (n_samples, n_classes) of predicted probabilities.
        labels: Integer ground-truth labels, shape (n_samples,).
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        Scalar ECE value.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D, got shape {probs.shape}")
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)
        count = in_bin.sum()
        if count == 0:
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = accuracies[in_bin].mean()
        ece += (count / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)
