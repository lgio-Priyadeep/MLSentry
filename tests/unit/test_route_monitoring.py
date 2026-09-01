"""Unit tests for monitoring routes: GET /v1/drift, GET /v1/performance, POST /v1/monitoring/run.

Covers:
  - 401 UNAUTHORIZED on missing or invalid API key
  - 404 NOT_FOUND on non-existent model_id
  - 409 MODEL_WARMING_UP on manual monitoring run when model is warming_up
  - 409 MODEL_DEPRECATED on manual monitoring run when model is deprecated
  - 200 OK on manual monitoring run when model is active
  - 200 OK with insufficient_data=True on drift query with < 30 samples
  - 200 OK with populated drift report & dynamic severity computation (OK/WARNING/CRITICAL)
  - 200 OK with insufficient_data=True on performance query with < 50 matched pairs
  - 200 OK with populated performance log & dynamic severity computation
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
from mlsentry.api.routes.monitoring import router as monitoring_router
from mlsentry.config.settings import Settings
from mlsentry.core.constants import (
    DriftMethod,
    FeatureKind,
    ModelStatus,
    PerformanceMetric,
    PredictionType,
)
from mlsentry.db.models import (
    Base,
    DriftReportRecord,
    GroundTruthRecord,
    ModelRecord,
    PerformanceLogRecord,
    PredictionRecord,
)
from mlsentry.db.session import get_session

TEST_API_KEY = "test_secret_monitoring_key_12345"


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
    api_v1.include_router(monitoring_router)
    app.include_router(api_v1)

    return TestClient(app)


@pytest.fixture
def active_model(db_session) -> ModelRecord:
    model = ModelRecord(
        model_id=uuid.uuid4(),
        name="fraud_detector",
        version="1.0.0",
        prediction_type=PredictionType.BINARY,
        status=ModelStatus.ACTIVE,
        sample_count=100,
        baseline_f1=0.90,
        baseline_auc=0.95,
        registered_at=datetime.now(timezone.utc),
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


@pytest.fixture
def warming_up_model(db_session) -> ModelRecord:
    model = ModelRecord(
        model_id=uuid.uuid4(),
        name="credit_scorer",
        version="1.0.0",
        prediction_type=PredictionType.BINARY,
        status=ModelStatus.WARMING_UP,
        sample_count=15,
        baseline_f1=0.85,
        baseline_auc=0.90,
        registered_at=datetime.now(timezone.utc),
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


@pytest.fixture
def deprecated_model(db_session) -> ModelRecord:
    now = datetime.now(timezone.utc)
    model = ModelRecord(
        model_id=uuid.uuid4(),
        name="old_model",
        version="0.9.0",
        prediction_type=PredictionType.BINARY,
        status=ModelStatus.DEPRECATED,
        sample_count=500,
        baseline_f1=0.80,
        baseline_auc=0.85,
        registered_at=now - timedelta(days=30),
        deprecated_at=now,
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


# ─── Auth Tests ──────────────────────────────────────────────────


def test_monitoring_routes_require_auth(client):
    fake_id = str(uuid.uuid4())
    # GET /v1/drift
    resp = client.get(f"/v1/drift/{fake_id}")
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert resp.json()["error"] == "UNAUTHORIZED"

    # GET /v1/performance
    resp = client.get(f"/v1/performance/{fake_id}")
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert resp.json()["error"] == "UNAUTHORIZED"

    # POST /v1/monitoring/run
    resp = client.post(f"/v1/monitoring/run/{fake_id}")
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert resp.json()["error"] == "UNAUTHORIZED"


# ─── GET /v1/drift/{model_id} Tests ──────────────────────────────


def test_get_drift_not_found(client):
    fake_id = str(uuid.uuid4())
    resp = client.get(
        f"/v1/drift/{fake_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert resp.json()["error"] == "NOT_FOUND"


def test_get_drift_warming_up_insufficient_data(client, warming_up_model):
    resp = client.get(
        f"/v1/drift/{warming_up_model.model_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["model_id"] == str(warming_up_model.model_id)
    assert body["insufficient_data"] is True
    assert body["sample_count"] == 15
    assert body["model_status"] == "warming_up"
    assert body["data"] == []


def test_get_drift_with_populated_reports(client, active_model, db_session):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    w_start = now - timedelta(minutes=15)
    w_end = now

    # Add numerical report (PSI = 0.28 -> CRITICAL)
    num_report = DriftReportRecord(
        report_id=uuid.uuid4(),
        model_id=active_model.model_id,
        feature_name="income",
        window_start=w_start,
        window_end=w_end,
        sample_count=100,
        method=DriftMethod.PSI,
        score=0.28,
        p_value=None,
        feature_type=FeatureKind.NUMERICAL,
        psi=0.28,
        computed_at=now,
    )
    # Add categorical report (Chi2 p_value = 0.03 -> WARNING)
    cat_report = DriftReportRecord(
        report_id=uuid.uuid4(),
        model_id=active_model.model_id,
        feature_name="category",
        window_start=w_start,
        window_end=w_end,
        sample_count=100,
        method=DriftMethod.CHI_SQUARE,
        score=7.82,
        p_value=0.03,
        feature_type=FeatureKind.CATEGORICAL,
        psi=None,
        computed_at=now,
    )
    db_session.add_all([num_report, cat_report])
    db_session.commit()

    resp = client.get(
        f"/v1/drift/{active_model.model_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["model_id"] == str(active_model.model_id)
    assert body["insufficient_data"] is False
    assert len(body["data"]) == 2

    # Check dynamic severities
    income_item = next(d for d in body["data"] if d["feature_name"] == "income")
    assert income_item["method"] == "psi"
    assert income_item["severity"] == "CRITICAL"
    assert income_item["psi"] == 0.28

    cat_item = next(d for d in body["data"] if d["feature_name"] == "category")
    assert cat_item["method"] == "chi-square"
    assert cat_item["severity"] == "WARNING"
    assert cat_item["psi"] is None


# ─── GET /v1/performance/{model_id} Tests ────────────────────────


def test_get_performance_not_found(client):
    fake_id = str(uuid.uuid4())
    resp = client.get(
        f"/v1/performance/{fake_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert resp.json()["error"] == "NOT_FOUND"


def test_get_performance_insufficient_data(client, active_model):
    resp = client.get(
        f"/v1/performance/{active_model.model_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["model_id"] == str(active_model.model_id)
    assert body["insufficient_data"] is True
    assert body["matched_pairs"] == 0
    assert body["data"] == []


def test_get_performance_with_populated_logs(client, active_model, db_session):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    w_start = now - timedelta(minutes=15)
    w_end = now

    f1_log = PerformanceLogRecord(
        perf_id=uuid.uuid4(),
        model_id=active_model.model_id,
        window_start=w_start,
        window_end=w_end,
        metric=PerformanceMetric.F1,
        value=0.74,
        baseline_value=0.90,
        delta=-0.16,
        sample_count=80,
        computed_at=now,
    )
    auc_log = PerformanceLogRecord(
        perf_id=uuid.uuid4(),
        model_id=active_model.model_id,
        window_start=w_start,
        window_end=w_end,
        metric=PerformanceMetric.AUC,
        value=0.88,
        baseline_value=0.95,
        delta=-0.07,
        sample_count=80,
        computed_at=now,
    )
    db_session.add_all([f1_log, auc_log])
    db_session.commit()

    resp = client.get(
        f"/v1/performance/{active_model.model_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["model_id"] == str(active_model.model_id)
    assert body["insufficient_data"] is False
    assert body["matched_pairs"] == 80
    assert len(body["data"]) == 2

    # Check dynamic severities
    f1_item = next(d for d in body["data"] if d["metric"] == "f1")
    assert f1_item["delta"] == -0.16
    assert f1_item["severity"] == "CRITICAL"

    auc_item = next(d for d in body["data"] if d["metric"] == "auc")
    assert auc_item["delta"] == -0.07
    assert auc_item["severity"] == "WARNING"


# ─── POST /v1/monitoring/run/{model_id} Tests ────────────────────


def test_monitoring_run_not_found(client):
    fake_id = str(uuid.uuid4())
    resp = client.post(
        f"/v1/monitoring/run/{fake_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert resp.json()["error"] == "NOT_FOUND"


def test_monitoring_run_warming_up_rejected(client, warming_up_model):
    resp = client.post(
        f"/v1/monitoring/run/{warming_up_model.model_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_409_CONFLICT
    assert resp.json()["error"] == "MODEL_WARMING_UP"


def test_monitoring_run_deprecated_rejected(client, deprecated_model):
    resp = client.post(
        f"/v1/monitoring/run/{deprecated_model.model_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_409_CONFLICT
    assert resp.json()["error"] == "MODEL_DEPRECATED"


def test_monitoring_run_active_success(client, active_model):
    resp = client.post(
        f"/v1/monitoring/run/{active_model.model_id}",
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["model_id"] == str(active_model.model_id)
    assert body["status"] == "triggered"
    assert "run_id" in body
    assert "triggered_at" in body
