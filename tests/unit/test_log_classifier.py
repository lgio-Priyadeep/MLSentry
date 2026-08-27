"""Unit tests for the Core Log Anomaly Classifier (DistilBERT).

Validates:
  - Anomaly classification with alert trigger (label="anomalous", confidence >= 0.85)
  - Anomaly classification below alert threshold (confidence = 0.849)
  - Normal classification (label="normal", no alert)
  - Input validation: empty strings, whitespace, min length (10), max length (2000)
  - SLA breach tracking (latency > 500ms)
  - In-process circuit breaker: 5 consecutive failures -> OPEN (503/CircuitBreakerOpenError)
  - Circuit breaker recovery (HALF_OPEN probe success -> CLOSED)
  - Zero raw data in results (SHA256 log_line_hash verification)
"""
import time
from unittest.mock import MagicMock

import pytest

from mlsentry.core.anomaly.log_classifier import (
    CircuitBreakerOpenError,
    DistilBERTUnavailableError,
    LogClassificationResult,
    LogClassifier,
)
from mlsentry.core.constants import LogLabel


def _make_mock_pipeline(label: str = "anomalous", score: float = 0.92, delay_s: float = 0.0):
    """Factory creating mock HuggingFace text-classification pipeline."""
    def mock_pipe(text: str):
        if delay_s > 0.0:
            time.sleep(delay_s)
        return [{"label": label, "score": score}]
    return mock_pipe


class TestLogClassificationHappyPaths:
    """Tests for standard log classification outcomes."""

    def test_classify_anomalous_alert_triggered(self) -> None:
        classifier = LogClassifier(
            checkpoint="test-checkpoint",
            pipeline_factory=lambda cp: _make_mock_pipeline(label="anomalous", score=0.92),
        )
        res = classifier.classify_log("Feature 'monthly_income' shifted by 3.2 sigma")

        assert isinstance(res, LogClassificationResult)
        assert res.label == LogLabel.ANOMALOUS
        assert res.confidence == 0.9200
        assert res.alert_created is True
        assert res.sla_breached is False
        assert len(res.log_line_hash) == 64  # SHA256

    def test_classify_anomalous_boundary_floor_no_alert(self) -> None:
        classifier = LogClassifier(
            checkpoint="test-checkpoint",
            pipeline_factory=lambda cp: _make_mock_pipeline(label="anomalous", score=0.8490),
        )
        res = classifier.classify_log("Feature 'monthly_income' shifted by 2.1 sigma")

        assert res.label == LogLabel.ANOMALOUS
        assert res.confidence == 0.8490
        assert res.alert_created is False

    def test_classify_anomalous_boundary_floor_alert_created(self) -> None:
        classifier = LogClassifier(
            checkpoint="test-checkpoint",
            pipeline_factory=lambda cp: _make_mock_pipeline(label="anomalous", score=0.8500),
        )
        res = classifier.classify_log("Feature 'monthly_income' shifted by 2.5 sigma")

        assert res.label == LogLabel.ANOMALOUS
        assert res.confidence == 0.8500
        assert res.alert_created is True

    def test_classify_normal_no_alert(self) -> None:
        classifier = LogClassifier(
            checkpoint="test-checkpoint",
            pipeline_factory=lambda cp: _make_mock_pipeline(label="normal", score=0.99),
        )
        res = classifier.classify_log("Worker thread #4 started successfully at 10:00:00")

        assert res.label == LogLabel.NORMAL
        assert res.confidence == 0.9900
        assert res.alert_created is False


class TestInputValidation:
    """Tests for defensive log line payload validation."""

    def test_empty_string_raises_value_error(self) -> None:
        classifier = LogClassifier(
            pipeline_factory=lambda cp: _make_mock_pipeline(),
        )
        with pytest.raises(ValueError, match="empty or whitespace"):
            classifier.classify_log("")

    def test_whitespace_only_raises_value_error(self) -> None:
        classifier = LogClassifier(
            pipeline_factory=lambda cp: _make_mock_pipeline(),
        )
        with pytest.raises(ValueError, match="empty or whitespace"):
            classifier.classify_log("      ")

    def test_short_string_below_min_length_raises_value_error(self) -> None:
        classifier = LogClassifier(
            pipeline_factory=lambda cp: _make_mock_pipeline(),
        )
        with pytest.raises(ValueError, match="between 10 and 2000"):
            classifier.classify_log("Too short")

    def test_oversized_string_raises_value_error(self) -> None:
        classifier = LogClassifier(
            pipeline_factory=lambda cp: _make_mock_pipeline(),
        )
        oversized = "a" * 2001
        with pytest.raises(ValueError, match="between 10 and 2000"):
            classifier.classify_log(oversized)


class TestSLABreachTracking:
    """Tests for SLA latency monitoring (> 500ms)."""

    def test_sla_breach_flagged_when_latency_exceeds_threshold(self) -> None:
        # Mock pipeline with simulated 510ms delay
        classifier = LogClassifier(
            checkpoint="test-checkpoint",
            pipeline_factory=lambda cp: _make_mock_pipeline(delay_s=0.51),
        )
        res = classifier.classify_log("Simulated slow inference pipeline execution log line")
        assert res.sla_breached is True
        assert res.latency_ms > 500.0


class TestCircuitBreaker:
    """Tests for in-process DistilBERT circuit breaker."""

    def test_circuit_trips_open_after_5_consecutive_failures(self) -> None:
        failing_pipe = MagicMock(side_effect=RuntimeError("GPU OOM / CUDA error"))
        classifier = LogClassifier(
            pipeline_factory=lambda cp: failing_pipe,
        )

        # 5 consecutive failures
        for _ in range(5):
            with pytest.raises(DistilBERTUnavailableError):
                classifier.classify_log("Valid log string for pipeline test failure")

        # 6th call should trip circuit breaker without calling pipeline again
        failing_pipe.reset_mock()
        with pytest.raises(CircuitBreakerOpenError, match="circuit breaker is OPEN"):
            classifier.classify_log("Valid log string after circuit trip")

        # Pipeline was not invoked when OPEN
        failing_pipe.assert_not_called()

    def test_circuit_recovers_after_reset_window(self) -> None:
        failing_pipe = MagicMock(side_effect=RuntimeError("Pipeline crash"))
        classifier = LogClassifier(
            pipeline_factory=lambda cp: failing_pipe,
        )
        # Use short reset window (0.1s) for test speed
        classifier._circuit_breaker.reset_seconds = 0.1

        for _ in range(5):
            with pytest.raises(DistilBERTUnavailableError):
                classifier.classify_log("Valid log string for pipeline test failure")

        assert classifier._circuit_breaker.is_open is True

        # Wait for reset timeout to elapse
        time.sleep(0.12)
        assert classifier._circuit_breaker.is_open is False

        # Switch pipeline to healthy
        classifier._pipeline = _make_mock_pipeline(label="normal", score=0.95)

        # Half-open probe succeeds -> resets circuit to CLOSED
        res = classifier.classify_log("Valid log string probing half-open circuit")
        assert res.label == LogLabel.NORMAL
        assert classifier._circuit_breaker.is_open is False

    def test_circuit_breaker_property_access(self) -> None:
        classifier = LogClassifier(pipeline_factory=lambda cp: _make_mock_pipeline())
        assert classifier.circuit_breaker is not None
        assert classifier.circuit_breaker.is_open is False

    def test_model_load_failure_raises_distilbert_unavailable(self) -> None:
        def bad_factory(cp: str):
            raise ImportError("transformers not installed")

        classifier = LogClassifier(
            checkpoint="missing-checkpoint",
            pipeline_factory=bad_factory,
        )
        with pytest.raises(DistilBERTUnavailableError, match="Failed to load"):
            classifier.load_model()
