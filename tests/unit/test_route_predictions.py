"""Unit tests for POST /v1/predictions/log and POST /v1/ground_truth/log."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

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
from mlsentry.api.routes.predictions import router as predictions_router
from mlsentry.core.constants import AlertType, FeatureDtype, ModelStatus, PredictionType
from mlsentry.db.models import (
    AlertRecord,
    Base,
    GroundTruthRecord,
    ModelRecord,
    ModelSchemaRecord,
    PredictionRecord,
)
from mlsentry.db.session import get_session

TEST_API_KEY = "test-secret-key-12345"


@pytest.fixture
def test_db_session():
    """In-memory SQLite session with test fixture models."""
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
    """FastAPI TestClient with mounted predictions router and auth dependency."""
    app = FastAPI()
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
    api_v1.include_router(predictions_router)
    app.include_router(api_v1)

    return TestClient(app)


@pytest.fixture
def registered_model(test_db_session: Session) -> ModelRecord:
    """Create a registered test model in warming_up status."""
    model_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    model = ModelRecord(
        model_id=model_id,
        name="credit-risk",
        version="1.0.0",
        status=ModelStatus.WARMING_UP,
        prediction_type=PredictionType.BINARY,
        sample_count=29,  # Next prediction promotes to active
        registered_at=now,
    )
    test_db_session.add(model)

    # Add schemas: age (float, required), segment (category, required)
    s1 = ModelSchemaRecord(
        schema_id=uuid.uuid4(),
        model_id=model_id,
        feature_name="age",
        dtype=FeatureDtype.FLOAT,
        required=True,
        min_value=18.0,
        max_value=120.0,
        created_at=now,
    )
    s2 = ModelSchemaRecord(
        schema_id=uuid.uuid4(),
        model_id=model_id,
        feature_name="segment",
        dtype=FeatureDtype.CATEGORY,
        required=True,
        allowed_values=["standard", "premium"],
        created_at=now,
    )
    test_db_session.add_all([s1, s2])
    test_db_session.commit()
    return model


class TestPredictionLoggingRoute:
    """Test suite for POST /v1/predictions/log."""

    def test_successful_prediction_log_and_promotion(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        now = datetime.now(timezone.utc)
        payload = {
            "model_id": str(registered_model.model_id),
            "features_json": {"age": 35.0, "segment": "premium"},
            "prediction_label": 1,
            "confidence": 0.94,
            "timestamp": now.isoformat(),
        }
        res = client.post(
            "/v1/predictions/log",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 200
        data = res.json()
        assert "pred_id" in data

        # Verify DB entry
        pred_id = uuid.UUID(data["pred_id"])
        pred = test_db_session.query(PredictionRecord).filter_by(pred_id=pred_id).first()
        assert pred is not None
        assert pred.schema_valid is True
        assert pred.model_version == "1.0.0"

        # Verify auto-promotion to active (sample_count reached 30)
        test_db_session.refresh(registered_model)
        assert registered_model.sample_count == 30
        assert registered_model.status == ModelStatus.ACTIVE

    def test_nonexistent_model_returns_404(self, client: TestClient) -> None:
        payload = {
            "model_id": str(uuid.uuid4()),
            "features_json": {"age": 35.0},
            "prediction_label": 1,
            "confidence": 0.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        res = client.post(
            "/v1/predictions/log",
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
            "features_json": {"age": 35.0, "segment": "premium"},
            "prediction_label": 1,
            "confidence": 0.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        res = client.post(
            "/v1/predictions/log",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 409
        assert res.json()["error"] == "MODEL_DEPRECATED"

    def test_hard_violation_nested_json_returns_422(
        self, client: TestClient, registered_model: ModelRecord
    ) -> None:
        payload = {
            "model_id": str(registered_model.model_id),
            "features_json": {"age": 35.0, "segment": {"nested": "dict"}},
            "prediction_label": 1,
            "confidence": 0.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        res = client.post(
            "/v1/predictions/log",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 422
        assert res.json()["error"] == "FEATURES_JSON_NESTED"

    def test_soft_violation_missing_required_logs_and_alerts(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        payload = {
            "model_id": str(registered_model.model_id),
            "features_json": {"segment": "premium"},  # missing 'age'
            "prediction_label": 1,
            "confidence": 0.8,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        res = client.post(
            "/v1/predictions/log",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 200
        pred_id = uuid.UUID(res.json()["pred_id"])

        pred = test_db_session.query(PredictionRecord).filter_by(pred_id=pred_id).first()
        assert pred is not None
        assert pred.schema_valid is False

        # Verify alert
        alert = (
            test_db_session.query(AlertRecord)
            .filter_by(model_id=registered_model.model_id, type=AlertType.SCHEMA_VIOLATION)
            .first()
        )
        assert alert is not None
        assert alert.feature_name == "age"


class TestGroundTruthLoggingRoute:
    """Test suite for POST /v1/ground_truth/log."""

    def test_successful_ground_truth_log(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        pred_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        pred = PredictionRecord(
            pred_id=pred_id,
            model_id=registered_model.model_id,
            model_version=registered_model.version,
            features_json={"age": 30.0},
            prediction_label="1",
            confidence=0.9,
            schema_valid=True,
            window_bucket=now,
            logged_at=now,
        )
        test_db_session.add(pred)
        test_db_session.commit()

        gt_payload = {
            "pred_id": str(pred_id),
            "label": "1",
            "model_id": str(registered_model.model_id),
        }
        res = client.post(
            "/v1/ground_truth/log",
            json=gt_payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 201
        data = res.json()
        assert data["pred_id"] == str(pred_id)
        assert "gt_id" in data

    def test_duplicate_ground_truth_returns_409(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        pred_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        pred = PredictionRecord(
            pred_id=pred_id,
            model_id=registered_model.model_id,
            model_version=registered_model.version,
            features_json={"age": 30.0},
            prediction_label="1",
            confidence=0.9,
            schema_valid=True,
            window_bucket=now,
            logged_at=now,
        )
        test_db_session.add(pred)
        test_db_session.commit()

        gt_payload = {"pred_id": str(pred_id), "label": "1"}
        res1 = client.post(
            "/v1/ground_truth/log",
            json=gt_payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res1.status_code == 201

        res2 = client.post(
            "/v1/ground_truth/log",
            json=gt_payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res2.status_code == 409
        assert res2.json()["error"] == "CONFLICT"

    def test_model_id_mismatch_returns_422(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        pred_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        pred = PredictionRecord(
            pred_id=pred_id,
            model_id=registered_model.model_id,
            model_version=registered_model.version,
            features_json={"age": 30.0},
            prediction_label="1",
            confidence=0.9,
            schema_valid=True,
            window_bucket=now,
            logged_at=now,
        )
        test_db_session.add(pred)
        test_db_session.commit()

        gt_payload = {
            "pred_id": str(pred_id),
            "label": "1",
            "model_id": str(uuid.uuid4()),  # Mismatched model_id
        }
        res = client.post(
            "/v1/ground_truth/log",
            json=gt_payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 422
        assert res.json()["error"] == "MODEL_ID_MISMATCH"

    def test_late_label_arrival_returns_422(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        pred_id = uuid.uuid4()
        logged_73h_ago = datetime.now(timezone.utc) - timedelta(hours=73)
        pred = PredictionRecord(
            pred_id=pred_id,
            model_id=registered_model.model_id,
            model_version=registered_model.version,
            features_json={"age": 30.0},
            prediction_label="1",
            confidence=0.9,
            schema_valid=True,
            window_bucket=logged_73h_ago,
            logged_at=logged_73h_ago,
        )
        test_db_session.add(pred)
        test_db_session.commit()

        gt_payload = {"pred_id": str(pred_id), "label": "1"}
        res = client.post(
            "/v1/ground_truth/log",
            json=gt_payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 422
        assert res.json()["error"] == "LATE_LABEL_REJECTED"

    def test_deprecated_model_ground_truth_returns_409(
        self, client: TestClient, registered_model: ModelRecord, test_db_session: Session
    ) -> None:
        pred_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        pred = PredictionRecord(
            pred_id=pred_id,
            model_id=registered_model.model_id,
            model_version=registered_model.version,
            features_json={"age": 30.0},
            prediction_label="1",
            confidence=0.9,
            schema_valid=True,
            window_bucket=now,
            logged_at=now,
        )
        test_db_session.add(pred)
        registered_model.status = ModelStatus.DEPRECATED
        registered_model.deprecated_at = now
        test_db_session.commit()

        gt_payload = {"pred_id": str(pred_id), "label": "1"}
        res = client.post(
            "/v1/ground_truth/log",
            json=gt_payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert res.status_code == 409
        assert res.json()["error"] == "MODEL_DEPRECATED"
