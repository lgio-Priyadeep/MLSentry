"""Unit tests for POST /v1/logs/classify route handler."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.exceptions import HTTPException as StarletteHTTPException

from mlsentry.api.errors import (
    MLSentryAPIException,
    generic_exception_handler,
    http_exception_handler,
    mlsentry_api_exception_handler,
    validation_exception_handler,
)
from mlsentry.api.middleware import RequestIDMiddleware, get_api_key_dependency
from mlsentry.api.routes.logs import router as logs_router
from mlsentry.core.anomaly.log_classifier import (
    CircuitBreakerOpenError,
    DistilBERTUnavailableError,
    LogClassificationResult,
    LogClassifier,
)
from mlsentry.core.constants import AlertType, LogLabel, ModelStatus, PredictionType
from mlsentry.db.models import (
    AlertRecord,
    Base,
    LogClassificationRecord,
    ModelRecord,
)
from mlsentry.db.session import get_session

TEST_API_KEY = "test-secret-key-12345"


@pytest.fixture
def test_db_session():
    """In-memory SQLite session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db_session: Session):
    """FastAPI TestClient with mounted logs router and auth dependency."""
    app = FastAPI()
    app.state.classifier = None
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(MLSentryAPIException, mlsentry_api_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    auth_dep = get_api_key_dependency(TEST_API_KEY)

    def override_get_session():
        yield test_db_session

    app.dependency_overrides[get_session] = override_get_session

    api_v1 = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])
    api_v1.include_router(logs_router)
    app.include_router(api_v1)

    return TestClient(app)


@pytest.fixture
def registered_model(test_db_session: Session) -> ModelRecord:
    """Create a registered test model."""
    model_id = uuid.uuid4()
    model = ModelRecord(
        model_id=model_id,
        name="nlp-pipeline",
        version="1.0.0",
        status=ModelStatus.ACTIVE,
        prediction_type=PredictionType.BINARY,
        sample_count=100,
        registered_at=datetime.now(timezone.utc),
    )
    test_db_session.add(model)
    test_db_session.commit()
    return model


class TestLogClassifyRoute:
    """Test suite for POST /v1/logs/classify."""

    def test_empty_log_line_returns_422_without_classifier(
        self, client: TestClient, registered_model: ModelRecord
    ) -> None:
        payload = {
            "model_id": str(registered_model.model_id),
            "log_line": "   ",
        }
        res = client.post(
            "/v1/logs/classify",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 422
        assert res.json()["error"] == "EMPTY_LOG_LINE"

    def test_short_log_line_returns_422_invalid_log_line(
        self, client: TestClient, registered_model: ModelRecord
    ) -> None:
        mock_classifier = MagicMock(spec=LogClassifier)
        mock_classifier.circuit_breaker.is_open = False
        mock_classifier.classify_log.side_effect = ValueError(
            "log_line length (5) must be between 10 and 2000 characters."
        )
        client.app.state.classifier = mock_classifier

        payload = {
            "model_id": str(registered_model.model_id),
            "log_line": "short",
        }
        res = client.post(
            "/v1/logs/classify",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 422
        assert res.json()["error"] == "INVALID_LOG_LINE"

    def test_nonexistent_model_returns_404(self, client: TestClient) -> None:
        payload = {
            "model_id": str(uuid.uuid4()),
            "log_line": "Processing batch 100 with zero errors",
        }
        res = client.post(
            "/v1/logs/classify",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 404
        assert res.json()["error"] == "NOT_FOUND"

    def test_deprecated_model_returns_409(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        registered_model.status = ModelStatus.DEPRECATED
        registered_model.deprecated_at = datetime.now(timezone.utc)
        test_db_session.commit()

        payload = {
            "model_id": str(registered_model.model_id),
            "log_line": "Batch job completed normally",
        }
        res = client.post(
            "/v1/logs/classify",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 409
        assert res.json()["error"] == "MODEL_DEPRECATED"

    def test_classifier_circuit_breaker_open_returns_503(
        self, client: TestClient, registered_model: ModelRecord
    ) -> None:
        mock_classifier = MagicMock(spec=LogClassifier)
        mock_classifier.circuit_breaker.is_open = True
        client.app.state.classifier = mock_classifier

        payload = {
            "model_id": str(registered_model.model_id),
            "log_line": "Feature drift detected on column age",
        }
        res = client.post(
            "/v1/logs/classify",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 503
        assert res.json()["error"] == "DISTILBERT_UNAVAILABLE"

    def test_successful_classification_normal(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        mock_result = LogClassificationResult(
            log_line_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            label=LogLabel.NORMAL,
            confidence=0.96,
            alert_created=False,
            latency_ms=45.2,
            sla_breached=False,
            model_checkpoint="distilbert-base-uncased",
        )
        mock_classifier = MagicMock(spec=LogClassifier)
        mock_classifier.circuit_breaker.is_open = False
        mock_classifier.classify_log.return_value = mock_result
        client.app.state.classifier = mock_classifier

        payload = {
            "model_id": str(registered_model.model_id),
            "log_line": "Batch 450 finished successfully in 120ms",
        }
        res = client.post(
            "/v1/logs/classify",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["label"] == "normal"
        assert data["confidence"] == 0.96

        # Verify DB persistence
        log_rec = (
            test_db_session.query(LogClassificationRecord)
            .filter_by(model_id=registered_model.model_id)
            .first()
        )
        assert log_rec is not None
        assert log_rec.label == LogLabel.NORMAL
        assert log_rec.alert_created is False

    def test_successful_classification_anomalous_triggers_alert(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        mock_result = LogClassificationResult(
            log_line_hash="a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
            label=LogLabel.ANOMALOUS,
            confidence=0.92,
            alert_created=True,
            latency_ms=78.5,
            sla_breached=False,
            model_checkpoint="distilbert-base-uncased",
        )
        mock_classifier = MagicMock(spec=LogClassifier)
        mock_classifier.circuit_breaker.is_open = False
        mock_classifier.classify_log.return_value = mock_result
        client.app.state.classifier = mock_classifier

        payload = {
            "model_id": str(registered_model.model_id),
            "log_line": "Feature 'income' mean shifted by 3.8 sigma during pipeline run",
        }
        res = client.post(
            "/v1/logs/classify",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["label"] == "anomalous"
        assert data["confidence"] == 0.92

        # Verify alert created in DB
        alert = (
            test_db_session.query(AlertRecord)
            .filter_by(model_id=registered_model.model_id, type=AlertType.LOG_ANOMALY)
            .first()
        )
        assert alert is not None
        assert "0.92" in alert.message
