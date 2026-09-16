"""Global pytest test fixtures, database session isolation, and test client factories."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from mlsentry.api.main import create_app
from mlsentry.config.settings import Settings
from mlsentry.core.anomaly.log_classifier import LogClassificationResult, LogClassifier
from mlsentry.core.constants import LogLabel
from mlsentry.db.models import Base
from mlsentry.db.session import get_session
from mlsentry.integrations.mlflow_client import MLflowModelMetadata

# In-memory SQLite engine for isolated and fast integration test execution
_TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Fixture providing deterministic test configuration settings."""
    return Settings(
        database_url=_TEST_DB_URL,
        mlsentry_api_key="test_secret_api_key_v1",
        log_anomaly_confidence_threshold=0.85,
        monitoring_interval_minutes=15,
        github_token="ghp_test_token_12345",
        github_repo_owner="test-org",
        github_repo_name="test-repo",
        github_workflow_id="retrain.yml",
        mlflow_tracking_uri="http://mock-mlflow:5000",
    )


@pytest.fixture(scope="function")
def db_engine():
    """Create a fresh in-memory SQLite database per test function."""
    engine = create_engine(
        _TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Provide an isolated database session with automatic transaction rollback."""
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = testing_session_local()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(test_settings: Settings, db_session: Session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with overridden database session and mock integrations."""
    app = create_app(test_settings)

    # Dependency override for database session
    def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    # Mock DistilBERT pipeline to prevent slow neural net downloads during unit/integration tests
    mock_classifier = MagicMock(spec=LogClassifier)
    mock_classifier.classify_log.return_value = LogClassificationResult(
        log_line_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        label=LogLabel.NORMAL,
        confidence=0.95,
        alert_created=False,
        latency_ms=10.0,
        sla_breached=False,
        model_checkpoint="distilbert-base-uncased",
    )
    mock_classifier._circuit_breaker = MagicMock()
    mock_classifier._circuit_breaker.is_open = False
    mock_classifier.circuit_breaker.is_open = False
    mock_classifier._pipeline = MagicMock()
    app.state.classifier = mock_classifier

    # Mock MLflow client
    mock_mlflow = MagicMock()
    mock_mlflow.fetch_model_metadata.return_value = MLflowModelMetadata(
        name="credit_default_predictor",
        version="1.0.0",
        run_id="mock-run-001",
        description=None,
        tags={},
        baseline_stats=None,
    )
    app.state.mlflow_client = mock_mlflow

    # Patch start_scheduler / stop_scheduler during test runs to avoid background worker race conditions
    with patch("mlsentry.api.main.start_scheduler"), patch("mlsentry.api.main.stop_scheduler"):
        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Standard authentication headers with valid API key."""
    return {"X-API-Key": "test_secret_api_key_v1"}


@pytest.fixture
def valid_registration_payload() -> dict[str, Any]:
    """Standard valid model registration request payload."""
    return {
        "name": "credit_default_predictor",
        "version": "1.0.0",
        "prediction_type": "binary",
        "baseline_f1": 0.85,
        "baseline_auc": 0.90,
        "features": [
            {
                "feature_name": "annual_income",
                "dtype": "float",
                "required": True,
                "min_value": 0.0,
                "max_value": 1000000.0,
                "baseline_stats": {
                    "mean": 65000.0,
                    "std": 15000.0,
                    "min": 10000.0,
                    "max": 250000.0,
                    "p25": 45000.0,
                    "p50": 62000.0,
                    "p75": 80000.0,
                    "p95": 120000.0,
                    "histogram_bin_edges": [10000.0, 50000.0, 100000.0, 250000.0],
                    "histogram_counts": [300, 500, 200],
                },
            },
            {
                "feature_name": "employment_type",
                "dtype": "category",
                "required": True,
                "baseline_stats": {
                    "frequency_map": {
                        "salaried": 0.60,
                        "self_employed": 0.30,
                        "unemployed": 0.10,
                    }
                },
            },
        ],
    }
