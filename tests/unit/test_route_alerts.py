"""Unit tests for alert management routes: GET /v1/alerts, PATCH /v1/alerts/{alert_id}/resolve.

Covers:
  - 401 UNAUTHORIZED on missing or invalid API key
  - 404 NOT_FOUND on non-existent model_id
  - 404 NOT_FOUND on non-existent alert_id for resolution
  - 422 SCHEMA_VALIDATION_FAILED on invalid status query parameter
  - 422 SCHEMA_VALIDATION_FAILED on invalid severity query parameter
  - 200 OK on GET /v1/alerts with active alerts (default status)
  - 200 OK on GET /v1/alerts with status=resolved filter
  - 200 OK on GET /v1/alerts with severity filter
  - 200 OK on PATCH /v1/alerts/{alert_id}/resolve setting resolved=true and resolved_at
  - 409 CONFLICT on repeated PATCH /v1/alerts/{alert_id}/resolve (immutable resolution)
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import APIRouter, Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
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
from mlsentry.api.routes.alerts import router as alerts_router
from mlsentry.config.settings import Settings
from mlsentry.core.constants import AlertSeverity, AlertType, ModelStatus, PredictionType
from mlsentry.db.models import AlertRecord, Base, ModelRecord
from mlsentry.db.session import get_session

TEST_API_KEY = "test_secret_alerts_key_12345"


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    app = FastAPI()
    app.state.settings = Settings(
        mlsentry_api_key=TEST_API_KEY,
        database_url="sqlite:///:memory:",
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(MLSentryAPIException, mlsentry_api_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    auth_dep = get_api_key_dependency(TEST_API_KEY)

    def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    api_v1 = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])
    api_v1.include_router(alerts_router)
    app.include_router(api_v1)

    return TestClient(app)


@pytest.fixture
def sample_model(db_session) -> ModelRecord:
    model = ModelRecord(
        model_id=uuid.uuid4(),
        name="risk_scorer",
        version="1.0.0",
        prediction_type=PredictionType.BINARY,
        status=ModelStatus.ACTIVE,
        sample_count=200,
        registered_at=datetime.now(timezone.utc),
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


@pytest.fixture
def sample_alerts(db_session, sample_model) -> list[AlertRecord]:
    now = datetime.now(timezone.utc)
    a1 = AlertRecord(
        alert_id=uuid.uuid4(),
        model_id=sample_model.model_id,
        type=AlertType.DRIFT,
        severity=AlertSeverity.CRITICAL,
        feature_name="income",
        message="PSI score 0.31 on feature 'income' exceeds CRITICAL threshold 0.25",
        triggered_at=now - timedelta(minutes=10),
        resolved=False,
        resolved_at=None,
        cooldown_until=now + timedelta(minutes=20),
    )
    a2 = AlertRecord(
        alert_id=uuid.uuid4(),
        model_id=sample_model.model_id,
        type=AlertType.LOG_ANOMALY,
        severity=AlertSeverity.WARNING,
        feature_name=None,
        message="DistilBERT classified log line as anomalous with confidence 0.92",
        triggered_at=now - timedelta(minutes=5),
        resolved=False,
        resolved_at=None,
        cooldown_until=None,
    )
    a3 = AlertRecord(
        alert_id=uuid.uuid4(),
        model_id=sample_model.model_id,
        type=AlertType.SCHEMA_WARNING,
        severity=AlertSeverity.INFO,
        feature_name="extra_field",
        message="Extra feature encountered in prediction payload",
        triggered_at=now - timedelta(hours=2),
        resolved=True,
        resolved_at=now - timedelta(hours=1),
        cooldown_until=None,
    )
    db_session.add_all([a1, a2, a3])
    db_session.commit()
    return [a1, a2, a3]


# ─── Auth Tests ──────────────────────────────────────────────────


def test_alerts_routes_require_auth(client):
    fake_id = str(uuid.uuid4())
    # GET /v1/alerts
    resp = client.get(f"/v1/alerts/{fake_id}")
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert resp.json()["error"] == "UNAUTHORIZED"

    # PATCH /v1/alerts/{id}/resolve
    resp = client.patch(f"/v1/alerts/{fake_id}/resolve")
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert resp.json()["error"] == "UNAUTHORIZED"


# ─── GET /v1/alerts/{model_id} Tests ─────────────────────────────


def test_get_alerts_not_found(client):
    fake_id = str(uuid.uuid4())
    resp = client.get(
        f"/v1/alerts/{fake_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert resp.json()["error"] == "NOT_FOUND"


def test_get_alerts_invalid_status_filter(client, sample_model):
    resp = client.get(
        f"/v1/alerts/{sample_model.model_id}?status=invalid_status",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert resp.json()["error"] == "SCHEMA_VALIDATION_FAILED"


def test_get_alerts_invalid_severity_filter(client, sample_model):
    resp = client.get(
        f"/v1/alerts/{sample_model.model_id}?severity=EXTREME",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert resp.json()["error"] == "SCHEMA_VALIDATION_FAILED"


def test_get_alerts_default_active(client, sample_model, sample_alerts):
    resp = client.get(
        f"/v1/alerts/{sample_model.model_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["model_id"] == str(sample_model.model_id)
    assert len(body["data"]) == 2  # 2 active alerts (a1, a2)
    assert body["meta"]["total"] == 2

    # Check alert items structure
    drift_item = next(d for d in body["data"] if d["type"] == "DRIFT")
    assert drift_item["feature_name"] == "income"
    assert drift_item["severity"] == "CRITICAL"
    assert drift_item["resolved_at"] is None

    log_item = next(d for d in body["data"] if d["type"] == "LOG_ANOMALY")
    assert log_item["feature_name"] is None
    assert log_item["severity"] == "WARNING"
    assert log_item["resolved_at"] is None


def test_get_alerts_resolved_filter(client, sample_model, sample_alerts):
    resp = client.get(
        f"/v1/alerts/{sample_model.model_id}?status=resolved",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert len(body["data"]) == 1  # a3
    assert body["data"][0]["type"] == "SCHEMA_WARNING"
    assert body["data"][0]["resolved_at"] is not None


def test_get_alerts_severity_filter(client, sample_model, sample_alerts):
    resp = client.get(
        f"/v1/alerts/{sample_model.model_id}?status=active&severity=CRITICAL",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["type"] == "DRIFT"
    assert body["data"][0]["severity"] == "CRITICAL"


# ─── PATCH /v1/alerts/{alert_id}/resolve Tests ───────────────────


def test_resolve_alert_not_found(client):
    fake_id = str(uuid.uuid4())
    resp = client.patch(
        f"/v1/alerts/{fake_id}/resolve",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert resp.json()["error"] == "NOT_FOUND"


def test_resolve_alert_success_and_conflict(client, sample_alerts, db_session):
    active_alert = sample_alerts[0]  # a1
    assert active_alert.resolved is False

    # 1. Resolve alert
    resp = client.patch(
        f"/v1/alerts/{active_alert.alert_id}/resolve",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["alert_id"] == str(active_alert.alert_id)
    assert body["resolved"] is True
    assert "resolved_at" in body

    # 2. Second resolve attempt -> 409 CONFLICT (immutable)
    resp2 = client.patch(
        f"/v1/alerts/{active_alert.alert_id}/resolve",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp2.status_code == status.HTTP_409_CONFLICT
    assert resp2.json()["error"] == "CONFLICT"


def test_resolve_already_resolved_alert_returns_conflict(client, sample_alerts):
    resolved_alert = sample_alerts[2]  # a3
    resp = client.patch(
        f"/v1/alerts/{resolved_alert.alert_id}/resolve",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_409_CONFLICT
    assert resp.json()["error"] == "CONFLICT"
