"""Unit tests for the Core Performance Tracker (F1 + AUC).

Validates:
  - F1 score calculation across perfect, balanced, zero-precision, and varied label types
  - ROC-AUC calculation across perfect ranking, inverse ranking, tied scores, and single class
  - Delta calculations and severity classification:
      WARNING threshold boundary: delta = -0.0500
      CRITICAL threshold boundary: delta = -0.1500
  - Sample count floor enforcement (minimum 50 matched pairs)
  - End-to-end evaluate_performance with binary and regression scenarios
"""
import pytest

from mlsentry.core.constants import (
    AlertSeverity,
    PerformanceMetric,
    PredictionType,
)
from mlsentry.core.performance.tracker import (
    PerformanceMetricResult,
    calculate_auc_score,
    calculate_f1_score,
    calculate_metric_delta,
    classify_perf_severity,
    evaluate_performance,
)


class TestF1ScoreCalculation:
    """Tests for calculate_f1_score."""

    def test_perfect_predictions(self) -> None:
        y_true = ["1", "1", "0", "0", "1"]
        y_pred = ["1", "1", "0", "0", "1"]
        f1 = calculate_f1_score(y_true, y_pred, pos_label="1")
        assert f1 == 1.0

    def test_all_wrong_predictions(self) -> None:
        y_true = ["1", "1", "1", "1"]
        y_pred = ["0", "0", "0", "0"]
        f1 = calculate_f1_score(y_true, y_pred, pos_label="1")
        assert f1 == 0.0

    def test_known_f1_score(self) -> None:
        # TP=2, FP=1, FN=1, TN=2 -> Precision = 2/3, Recall = 2/3 -> F1 = 2/3 ≈ 0.6667
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 0, 1, 0, 0]
        f1 = calculate_f1_score(y_true, y_pred, pos_label=1)
        assert f1 == 0.6667

    def test_string_boolean_label_aliases(self) -> None:
        y_true = ["positive", "positive", "negative"]
        y_pred = ["positive", "negative", "negative"]
        f1 = calculate_f1_score(y_true, y_pred, pos_label="positive")
        # TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5 -> F1 = 2/3 ≈ 0.6667
        assert f1 == 0.6667

    def test_length_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_f1_score([1, 0], [1, 0, 1])

    def test_empty_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty inputs"):
            calculate_f1_score([], [])


class TestAUCScoreCalculation:
    """Tests for calculate_auc_score."""

    def test_perfect_auc_ranking(self) -> None:
        y_true = ["1", "1", "0", "0"]
        y_scores = [0.95, 0.85, 0.20, 0.10]
        auc = calculate_auc_score(y_true, y_scores, pos_label="1")
        assert auc == 1.0

    def test_inverted_auc_ranking(self) -> None:
        y_true = ["1", "1", "0", "0"]
        y_scores = [0.10, 0.20, 0.85, 0.95]
        auc = calculate_auc_score(y_true, y_scores, pos_label="1")
        assert auc == 0.0

    def test_random_auc_ranking(self) -> None:
        y_true = [1, 0, 1, 0]
        y_scores = [0.8, 0.7, 0.3, 0.4]
        # Pos pairs: (0.8 vs 0.7) -> 1 win, (0.8 vs 0.4) -> 1 win, (0.3 vs 0.7) -> 0, (0.3 vs 0.4) -> 0 -> AUC = 2/4 = 0.5
        auc = calculate_auc_score(y_true, y_scores, pos_label=1)
        assert auc == 0.5

    def test_tied_confidence_scores(self) -> None:
        y_true = [1, 0, 1, 0]
        y_scores = [0.5, 0.5, 0.5, 0.5]
        auc = calculate_auc_score(y_true, y_scores, pos_label=1)
        assert auc == 0.5

    def test_single_class_returns_none(self) -> None:
        y_true = [1, 1, 1, 1]
        y_scores = [0.9, 0.8, 0.7, 0.6]
        auc = calculate_auc_score(y_true, y_scores, pos_label=1)
        assert auc is None

    def test_length_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_auc_score([1, 0], [0.9])


class TestMetricDeltaAndSeverity:
    """Tests for calculate_metric_delta and classify_perf_severity."""

    def test_delta_calculation(self) -> None:
        assert calculate_metric_delta(0.8500, 0.9000) == -0.0500
        assert calculate_metric_delta(0.6800, 0.8700) == -0.1900
        assert calculate_metric_delta(0.9200, 0.8800) == 0.0400

    def test_severity_boundaries(self) -> None:
        assert classify_perf_severity(-0.0499) == AlertSeverity.INFO
        assert classify_perf_severity(-0.0500) == AlertSeverity.WARNING
        assert classify_perf_severity(-0.1499) == AlertSeverity.WARNING
        assert classify_perf_severity(-0.1500) == AlertSeverity.CRITICAL
        assert classify_perf_severity(-0.2500) == AlertSeverity.CRITICAL
        assert classify_perf_severity(0.0200) == AlertSeverity.INFO


class TestEvaluatePerformanceEndToEnd:
    """Tests for evaluate_performance."""

    def test_insufficient_samples_raises_value_error(self) -> None:
        y_true = ["1"] * 25 + ["0"] * 24  # 49 samples (< 50)
        y_pred = ["1"] * 49
        with pytest.raises(ValueError, match="INSUFFICIENT_DATA"):
            evaluate_performance(
                prediction_type=PredictionType.BINARY,
                y_true=y_true,
                y_pred=y_pred,
                baseline_f1=0.90,
                enforce_min_samples=True,
            )

    def test_evaluate_binary_healthy(self) -> None:
        # 60 matched samples
        y_true = ["1"] * 30 + ["0"] * 30
        y_pred = ["1"] * 28 + ["0"] * 2 + ["1"] * 2 + ["0"] * 28
        y_scores = [0.9] * 30 + [0.1] * 30

        results = evaluate_performance(
            prediction_type=PredictionType.BINARY,
            y_true=y_true,
            y_pred=y_pred,
            y_scores=y_scores,
            baseline_f1=0.9300,
            baseline_auc=0.9500,
        )

        assert len(results) == 2
        f1_res = next(r for r in results if r.metric == PerformanceMetric.F1)
        auc_res = next(r for r in results if r.metric == PerformanceMetric.AUC)

        assert isinstance(f1_res, PerformanceMetricResult)
        assert f1_res.sample_count == 60
        assert f1_res.severity == AlertSeverity.INFO
        assert f1_res.is_degraded is False

        assert isinstance(auc_res, PerformanceMetricResult)
        assert auc_res.value == 1.0
        assert auc_res.is_degraded is False

    def test_evaluate_binary_critical_degradation(self) -> None:
        # Baseline F1 = 0.87, current F1 = 0.68 -> delta = -0.19 (CRITICAL)
        # Construct 60 samples with TP=20, FP=10, FN=10, TN=20
        y_true = ["1"] * 30 + ["0"] * 30
        y_pred = ["1"] * 20 + ["0"] * 10 + ["1"] * 10 + ["0"] * 20
        y_scores = [0.8] * 20 + [0.2] * 10 + [0.8] * 10 + [0.2] * 20

        results = evaluate_performance(
            prediction_type=PredictionType.BINARY,
            y_true=y_true,
            y_pred=y_pred,
            y_scores=y_scores,
            baseline_f1=0.8700,
            baseline_auc=0.9100,
        )

        f1_res = next(r for r in results if r.metric == PerformanceMetric.F1)
        assert f1_res.delta <= -0.15
        assert f1_res.severity == AlertSeverity.CRITICAL
        assert f1_res.is_degraded is True

    def test_evaluate_regression_skips_auc(self) -> None:
        y_true = [1.0] * 50
        y_pred = [1.0] * 50

        results = evaluate_performance(
            prediction_type=PredictionType.REGRESSION,
            y_true=y_true,
            y_pred=y_pred,
            baseline_f1=0.8000,
            baseline_auc=None,
        )

        # Regression does not evaluate F1 or AUC classification metrics
        assert len(results) == 0
