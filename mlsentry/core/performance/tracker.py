"""Model performance evaluation engine for matched prediction and ground truth pairs.

Computes classification performance metrics (F1 score and ROC-AUC) against
baseline values established at model registration.

Thresholds (from core/constants.py):
  - Sample count floor: Minimum 50 matched pairs required per evaluation window.
  - Performance delta (delta = value - baseline_value):
      delta > -0.05: INFO (normal performance)
      -0.15 < delta <= -0.05: WARNING (moderate degradation)
      delta <= -0.15: CRITICAL (severe degradation triggering alert & retraining eval)

This is a pure domain logic module:
  - No database access or ORM imports
  - No HTTP or API dependencies
  - Vectorized array operations using numpy and scipy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from mlsentry.core.constants import (
    MIN_MATCHED_PAIRS,
    PERF_DELTA_CRITICAL,
    PERF_DELTA_WARNING,
    AlertSeverity,
    PerformanceMetric,
    PredictionType,
)


@dataclass(frozen=True)
class PerformanceMetricResult:
    """Result of performance metric evaluation for a single metric.

    Attributes:
        metric: PerformanceMetric (F1 or AUC).
        value: Computed metric value in [0.0, 1.0].
        baseline_value: Frozen baseline metric value from model registration.
        delta: Metric shift (value - baseline_value) in [-1.0, 1.0].
        severity: INFO, WARNING, or CRITICAL.
        is_degraded: True if severity is WARNING or CRITICAL.
        sample_count: Number of matched prediction-label pairs evaluated.
    """

    metric: PerformanceMetric
    value: float
    baseline_value: float
    delta: float
    severity: AlertSeverity
    is_degraded: bool
    sample_count: int


def _normalize_binary_labels(
    labels: Sequence[Any],
    pos_label: Any = "1",
) -> np.ndarray:
    """Convert sequence of arbitrary label representations to binary 0/1 integers."""
    pos_str = str(pos_label).strip().lower()
    truthy_aliases = {pos_str, "1", "true", "positive", "yes"}

    binary = []
    for val in labels:
        val_str = str(val).strip().lower()
        binary.append(1 if val_str in truthy_aliases else 0)
    return np.array(binary, dtype=np.int32)


def calculate_f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = "1",
) -> float:
    """Calculate binary F1 score from matched true labels and predictions.

    Formula:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        y_true: Sequence of actual ground truth target values.
        y_pred: Sequence of model predicted target values.
        pos_label: Positive class label identifier (default "1").

    Returns:
        float: F1 score rounded to 4 decimal places in [0.0, 1.0].
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must match."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot calculate F1 score on empty inputs.")

    yt = _normalize_binary_labels(y_true, pos_label=pos_label)
    yp = _normalize_binary_labels(y_pred, pos_label=pos_label)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0

    f1 = 2.0 * (precision * recall) / (precision + recall)
    return round(float(np.clip(f1, 0.0, 1.0)), 4)


def calculate_auc_score(
    y_true: Sequence[Any],
    y_scores: Sequence[float],
    pos_label: Any = "1",
) -> float | None:
    """Calculate Area Under the ROC Curve (ROC-AUC) using Mann-Whitney U ranking.

    Args:
        y_true: Sequence of actual ground truth target values.
        y_scores: Sequence of predicted probability / confidence scores for positive class.
        pos_label: Positive class label identifier (default "1").

    Returns:
        float | None: ROC-AUC score rounded to 4 decimal places in [0.0, 1.0],
            or None if only one class exists in y_true.
    """
    if len(y_true) != len(y_scores):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) and y_scores ({len(y_scores)}) must match."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot calculate AUC score on empty inputs.")

    yt = _normalize_binary_labels(y_true, pos_label=pos_label)
    scores = np.asarray(y_scores, dtype=np.float64)

    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))

    if n_pos == 0 or n_neg == 0:
        # AUC is undefined when only a single class is present in true labels
        return None

    # Compute fractional ranks for tied confidence scores
    ranks = rankdata(scores)
    pos_rank_sum = float(np.sum(ranks[yt == 1]))

    # Mann-Whitney U formula for ROC-AUC
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1.0)) / 2.0
    auc = u_stat / (n_pos * n_neg)

    return round(float(np.clip(auc, 0.0, 1.0)), 4)


def calculate_metric_delta(current_value: float, baseline_value: float) -> float:
    """Calculate signed performance shift (current_value - baseline_value).

    Args:
        current_value: Current window evaluated metric score in [0.0, 1.0].
        baseline_value: Registration baseline metric score in [0.0, 1.0].

    Returns:
        float: Signed delta rounded to 4 decimal places in [-1.0, 1.0].
    """
    delta = current_value - baseline_value
    return round(float(np.clip(delta, -1.0, 1.0)), 4)


def classify_perf_severity(delta: float) -> AlertSeverity:
    """Classify performance degradation delta into AlertSeverity level.

    Args:
        delta: Signed performance drop (current - baseline).

    Returns:
        AlertSeverity: INFO, WARNING, or CRITICAL.
    """
    if delta <= PERF_DELTA_CRITICAL:
        return AlertSeverity.CRITICAL
    if delta <= PERF_DELTA_WARNING:
        return AlertSeverity.WARNING
    return AlertSeverity.INFO


def evaluate_performance(
    prediction_type: PredictionType,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_scores: Sequence[float] | None = None,
    baseline_f1: float | None = None,
    baseline_auc: float | None = None,
    pos_label: Any = "1",
    enforce_min_samples: bool = True,
) -> list[PerformanceMetricResult]:
    """Evaluate all configured performance metrics for matched prediction-label window.

    Args:
        prediction_type: Model prediction output type (BINARY, MULTICLASS, REGRESSION).
        y_true: Sequence of ground truth target labels.
        y_pred: Sequence of model predicted target labels.
        y_scores: Optional sequence of predicted confidence/probability scores.
        baseline_f1: Optional registered baseline F1 score.
        baseline_auc: Optional registered baseline ROC-AUC score.
        pos_label: Positive class label identifier.
        enforce_min_samples: If True, raises ValueError when sample count < 50.

    Returns:
        list[PerformanceMetricResult]: List of evaluated metric results with deltas.
    """
    sample_count = len(y_true)

    if enforce_min_samples and sample_count < MIN_MATCHED_PAIRS:
        raise ValueError(
            f"INSUFFICIENT_DATA: {sample_count} matched pairs, minimum {MIN_MATCHED_PAIRS} required."
        )

    results: list[PerformanceMetricResult] = []

    # Evaluate F1 metric if baseline_f1 is registered and model is classification
    if baseline_f1 is not None and prediction_type in (
        PredictionType.BINARY,
        PredictionType.MULTICLASS,
    ):
        f1_val = calculate_f1_score(y_true, y_pred, pos_label=pos_label)
        f1_delta = calculate_metric_delta(f1_val, baseline_f1)
        f1_sev = classify_perf_severity(f1_delta)

        results.append(
            PerformanceMetricResult(
                metric=PerformanceMetric.F1,
                value=f1_val,
                baseline_value=baseline_f1,
                delta=f1_delta,
                severity=f1_sev,
                is_degraded=(f1_sev in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)),
                sample_count=sample_count,
            )
        )

    # Evaluate AUC metric if baseline_auc is registered and confidences are available
    if (
        baseline_auc is not None
        and y_scores is not None
        and prediction_type == PredictionType.BINARY
    ):
        auc_val = calculate_auc_score(y_true, y_scores, pos_label=pos_label)
        if auc_val is not None:
            auc_delta = calculate_metric_delta(auc_val, baseline_auc)
            auc_sev = classify_perf_severity(auc_delta)

            results.append(
                PerformanceMetricResult(
                    metric=PerformanceMetric.AUC,
                    value=auc_val,
                    baseline_value=baseline_auc,
                    delta=auc_delta,
                    severity=auc_sev,
                    is_degraded=(
                        auc_sev in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)
                    ),
                    sample_count=sample_count,
                )
            )

    return results
