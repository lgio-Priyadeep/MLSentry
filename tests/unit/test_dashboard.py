"""Unit tests for Jinja2 Operations Dashboard: GET /dashboard and static assets."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlsentry.api.main import create_app
from mlsentry.config.settings import Settings
from mlsentry.db.models import ModelRecord


@pytest.fixture
def mock_settings():
    return Settings(
        database_url="sqlite:///:memory:",
        mlsentry_api_key="test_api_key",
        monitoring_interval_minutes=15,
    )


@pytest.fixture
def client(mock_settings):
    app = create_app(mock_settings)
    return TestClient(app)


def _make_mock_db(sample_model: ModelRecord) -> MagicMock:
    """Create a query-aware mock database session returning typed mocks per entity."""
    mock_db = MagicMock()

    def mock_execute(statement, *args, **kwargs):
        stmt_str = str(statement).lower()
        mock_result = MagicMock()

        # Only model queries return the sample model
        if "from models" in stmt_str or "models." in stmt_str:
            mock_result.scalars.return_value.all.return_value = [sample_model]
            mock_result.scalars.return_value.first.return_value = sample_model
        else:
            # Drift, alerts, performance logs, triggers return empty lists / None
            mock_result.scalars.return_value.all.return_value = []
            mock_result.scalars.return_value.first.return_value = None

        return mock_result

    mock_db.execute.side_effect = mock_execute
    return mock_db


def test_get_dashboard_unauthenticated_200(client):
    """Assert GET /dashboard renders 200 OK without requiring X-API-Key."""
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    mock_db.execute.return_value.scalars.return_value.first.return_value = None

    from mlsentry.db.session import get_session
    client.app.dependency_overrides[get_session] = lambda: mock_db

    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "MLSentry Observability" in response.text
    assert "1. Model Summary" in response.text
    assert "2. Drift & Alert Status" in response.text
    assert "3. Performance Trends" in response.text

    client.app.dependency_overrides.clear()


def test_get_dashboard_with_models(client):
    """Assert GET /dashboard renders model summary table with model metadata."""
    sample_model = MagicMock(spec=ModelRecord)
    sample_model.model_id = uuid.uuid4()
    sample_model.name = "credit_risk"
    sample_model.version = "1.0.0"
    sample_model.prediction_type = "binary"
    sample_model.status = "active"
    sample_model.sample_count = 150
    sample_model.warm_up_threshold = 30
    sample_model.baseline_f1 = 0.88
    sample_model.baseline_auc = 0.92

    mock_db = _make_mock_db(sample_model)

    from mlsentry.db.session import get_session
    client.app.dependency_overrides[get_session] = lambda: mock_db

    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "credit_risk" in response.text
    assert "v1.0.0" in response.text
    assert "150" in response.text

    client.app.dependency_overrides.clear()


def test_get_dashboard_filter_model_id(client):
    """Assert GET /dashboard?model_id=<uuid> filters dashboard view."""
    target_id = uuid.uuid4()
    sample_model = MagicMock(spec=ModelRecord)
    sample_model.model_id = target_id
    sample_model.name = "fraud_detection"
    sample_model.version = "2.1.0"
    sample_model.prediction_type = "binary"
    sample_model.status = "warming_up"
    sample_model.sample_count = 12
    sample_model.warm_up_threshold = 30
    sample_model.baseline_f1 = 0.90
    sample_model.baseline_auc = 0.95

    mock_db = _make_mock_db(sample_model)

    from mlsentry.db.session import get_session
    client.app.dependency_overrides[get_session] = lambda: mock_db

    response = client.get(f"/dashboard?model_id={target_id}")
    assert response.status_code == 200
    assert "fraud_detection" in response.text
    assert "Warming Up (12/30)" in response.text

    client.app.dependency_overrides.clear()


def test_get_static_chart_js(client):
    """Assert GET /static/js/chart.min.js serves offline Chart.js bundle."""
    response = client.get("/static/js/chart.min.js")
    assert response.status_code == 200
    assert "Chart.js" in response.text


def test_get_static_css(client):
    """Assert GET /static/css/dashboard.css serves dashboard styling."""
    response = client.get("/static/css/dashboard.css")
    assert response.status_code == 200
    assert "MLSentry" in response.text
