"""Unit tests for MLflow integration client and circuit breaker."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from mlsentry.config.settings import Settings
from mlsentry.integrations.mlflow_client import (
    MLflowCircuitBreaker,
    MLflowCircuitBreakerOpenError,
    MLflowClient,
    MLflowUnavailableError,
)


class TestMLflowCircuitBreaker:
    """Test suite for MLflow in-process circuit breaker state transitions."""

    def test_initial_state_is_closed(self) -> None:
        cb = MLflowCircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb.is_open is False

    def test_trips_open_after_three_failures_within_window(self) -> None:
        cb = MLflowCircuitBreaker(failure_threshold=3, window_seconds=60, reset_seconds=30)
        cb.record_failure()
        assert cb.state == "CLOSED"
        cb.record_failure()
        assert cb.state == "CLOSED"
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.is_open is True

    def test_recovers_after_reset_timeout(self) -> None:
        cb = MLflowCircuitBreaker(failure_threshold=3, window_seconds=60, reset_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True

        time.sleep(0.12)
        assert cb.state == "HALF_OPEN"
        assert cb.is_open is False

        # Successful probe call resets circuit to CLOSED
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.is_open is False

    def test_half_open_failure_re_opens_circuit(self) -> None:
        cb = MLflowCircuitBreaker(failure_threshold=3, window_seconds=60, reset_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True

        time.sleep(0.12)
        assert cb.state == "HALF_OPEN"

        # Failed probe call extends OPEN state
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.is_open is True

    def test_manual_reset(self) -> None:
        cb = MLflowCircuitBreaker()
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True
        cb.reset()
        assert cb.state == "CLOSED"
        assert cb.is_open is False


class TestMLflowClient:
    """Test suite for MLflowClient metadata fetching and error handling."""

    @patch("mlsentry.integrations.mlflow_client.requests.get")
    def test_fetch_metadata_success(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model_version": {
                "name": "churn_classifier",
                "version": "1.0",
                "run_id": "run-abc-123",
                "description": "Production churn model",
                "tags": [{"key": "env", "value": "prod"}],
            }
        }
        mock_get.return_value = mock_resp

        client = MLflowClient(tracking_uri="http://mlflow.example.com:5000")
        meta = client.fetch_model_metadata("churn_classifier", "1.0")

        assert meta.name == "churn_classifier"
        assert meta.version == "1.0"
        assert meta.run_id == "run-abc-123"
        assert meta.description == "Production churn model"
        assert meta.tags == {"env": "prod"}
        assert client.circuit_breaker.state == "CLOSED"

    @patch("mlsentry.integrations.mlflow_client.requests.get")
    def test_fetch_metadata_http_error_raises_unavailable(
        self, mock_get: MagicMock
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        client = MLflowClient(tracking_uri="http://mlflow.example.com:5000")
        with pytest.raises(MLflowUnavailableError, match="non-200"):
            client.fetch_model_metadata("churn_classifier", "1.0")

    @patch("mlsentry.integrations.mlflow_client.requests.get")
    def test_fetch_metadata_network_timeout_trips_breaker(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.side_effect = requests.Timeout("Connection timed out")

        client = MLflowClient(
            tracking_uri="http://mlflow.example.com:5000",
            circuit_breaker=MLflowCircuitBreaker(failure_threshold=3, reset_seconds=60),
        )

        for _ in range(3):
            with pytest.raises(MLflowUnavailableError):
                client.fetch_model_metadata("churn_classifier", "1.0")

        assert client.circuit_breaker.is_open is True

        # 4th call should immediately raise CircuitBreakerOpenError without network call
        mock_get.reset_mock()
        with pytest.raises(MLflowCircuitBreakerOpenError, match="circuit breaker is OPEN"):
            client.fetch_model_metadata("churn_classifier", "1.0")
        mock_get.assert_not_called()

    @patch("mlsentry.integrations.mlflow_client.requests.get")
    def test_check_health_true_on_200(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        client = MLflowClient()
        assert client.check_health() is True

    @patch("mlsentry.integrations.mlflow_client.requests.get")
    def test_check_health_false_on_exception(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.ConnectionError("Refused")

        client = MLflowClient()
        assert client.check_health() is False

    def test_from_settings(self) -> None:
        settings = Settings(
            mlsentry_api_key="secret-test-key",
            database_url="postgresql://user:pass@localhost:5432/db",
            mlflow_tracking_uri="http://mlflow.internal:5000",
            mlflow_timeout_ms=7500,
        )
        client = MLflowClient.from_settings(settings)
        assert client.tracking_uri == "http://mlflow.internal:5000"
        assert client.read_timeout == 7.5
