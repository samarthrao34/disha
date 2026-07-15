import numpy as np

from disha.evaluation.statistics import (
    cluster_bootstrap_interval,
    expected_calibration_error,
)


def test_perfect_calibration_error_is_zero_for_certain_correct_predictions():
    targets = np.array([0, 1, 0])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert expected_calibration_error(targets, probabilities) == 0.0


def test_cluster_bootstrap_is_deterministic_and_reports_clusters():
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    clusters = np.array([10, 10, 20, 20])
    metric = lambda actual, predicted: float((actual == predicted).mean())
    first = cluster_bootstrap_interval(
        targets, predictions, clusters, metric, iterations=100, seed=7
    )
    second = cluster_bootstrap_interval(
        targets, predictions, clusters, metric, iterations=100, seed=7
    )
    assert first == second
    assert first["cluster_count"] == 2
    assert first["estimate"] == 0.75
