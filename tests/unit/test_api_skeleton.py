"""Unit tests for FastAPI application skeleton, lifespan, error handlers, and health check."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from mlsentry.api.main import (
    create_app,
    get_scheduler_heartbeat,
    set_scheduler_heartbeat,
)
from mlsentry.config.settings import Settings


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        mlsentry_api_key="test-api-key-999",
        database_url="postgresql://user:pass@localhost:5432/mlsentry_test",
        log_level="DEBUG",
    )


class TestAPISkeletonAndHealth:
    """Test suite for application initialization, health check, and error envelope formatting."""

    @patch("mlsentry.api.main.init_engine")
    @patch("mlsentry.api.main.get_session")
    def test_health_check_healthy(
        self, mock_get_session: MagicMock, mock_init_engine: MagicMock, test_settings: Settings
    ) -> None:
        # Mock database session execution
        mock_session = MagicMock()
        mock_get_session.return_value = [mock_session]

        # Mock loaded classifier
        mock_classifier = MagicMock()
        mock_classifier._pipeline = MagicMock()
        mock_classifier._circuit_breaker.is_open = False

        with patch("mlsentry.api.main._distilbert_classifier", mock_classifier):
            app = create_app(test_settings)
            client = TestClient(app)

            set_scheduler_heartbeat(datetime.now(timezone.utc))

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["components"]["database"] == "ok"
            assert data["components"]["classifier"] == "ok"
            assert data["components"]["scheduler"] == "ok"
            assert data["last_scheduler_cycle_at"] is not None

    @patch("mlsentry.api.main.get_session")
    def test_health_check_database_down_returns_503(
        self, mock_get_session: MagicMock, test_settings: Settings
    ) -> None:
        mock_get_session.side_effect = Exception("DB Connection refused")

        mock_classifier = MagicMock()
        mock_classifier._pipeline = MagicMock()
        mock_classifier._circuit_breaker.is_open = False

        with patch("mlsentry.api.main._distilbert_classifier", mock_classifier):
            app = create_app(test_settings)
            client = TestClient(app)

            set_scheduler_heartbeat(datetime.now(timezone.utc))

            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["error"] == "SERVICE_UNAVAILABLE"
            assert "request_id" in data

    @patch("mlsentry.api.main.get_session")
    def test_health_check_unloaded_classifier_returns_503(
        self, mock_get_session: MagicMock, test_settings: Settings
    ) -> None:
        mock_session = MagicMock()
        mock_get_session.return_value = [mock_session]

        # Classifier exists but _pipeline is None (load failed or pending)
        mock_classifier = MagicMock()
        mock_classifier._pipeline = None
        mock_classifier._circuit_breaker.is_open = False

        with patch("mlsentry.api.main._distilbert_classifier", mock_classifier):
            app = create_app(test_settings)
            client = TestClient(app)

            set_scheduler_heartbeat(datetime.now(timezone.utc))

            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["error"] == "SERVICE_UNAVAILABLE"

    @patch("mlsentry.api.main.get_session")
    def test_health_check_circuit_breaker_open_returns_503(
        self, mock_get_session: MagicMock, test_settings: Settings
    ) -> None:
        mock_session = MagicMock()
        mock_get_session.return_value = [mock_session]

        mock_classifier = MagicMock()
        mock_classifier._pipeline = MagicMock()
        mock_classifier._circuit_breaker.is_open = True

        with patch("mlsentry.api.main._distilbert_classifier", mock_classifier):
            app = create_app(test_settings)
            client = TestClient(app)

            set_scheduler_heartbeat(datetime.now(timezone.utc))

            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["error"] == "SERVICE_UNAVAILABLE"

    def test_health_check_stale_scheduler_heartbeat_returns_503(
        self, test_settings: Settings
    ) -> None:
        mock_classifier = MagicMock()
        mock_classifier._pipeline = MagicMock()
        mock_classifier._circuit_breaker.is_open = False

        with patch("mlsentry.api.main._distilbert_classifier", mock_classifier):
            app = create_app(test_settings)
            client = TestClient(app)

            # Set heartbeat 35 minutes in the past (> 30 min staleness limit)
            stale_time = datetime.now(timezone.utc) - timedelta(minutes=35)
            set_scheduler_heartbeat(stale_time)

            with patch("mlsentry.api.main.get_session") as mock_get_session:
                mock_session = MagicMock()
                mock_get_session.return_value = [mock_session]
                response = client.get("/health")

            assert response.status_code == 503
            data = response.json()
            assert data["error"] == "SERVICE_UNAVAILABLE"

    def test_request_id_middleware_attaches_header(
        self, test_settings: Settings
    ) -> None:
        app = create_app(test_settings)
        client = TestClient(app)

        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0

    def test_not_found_returns_standard_error_envelope(
        self, test_settings: Settings
    ) -> None:
        app = create_app(test_settings)
        client = TestClient(app)

        response = client.get("/non_existent_route")
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "NOT_FOUND"
        assert "request_id" in data
