"""Unit tests for POST /v1/models/register route handler and schema validation."""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

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
from mlsentry.api.routes.models import router as models_router
from mlsentry.config.settings import Settings
from mlsentry.core.constants import ModelStatus, PredictionType
from mlsentry.db.models import Base, ModelRecord, ModelSchemaRecord, ReferenceStatRecord
from mlsentry.db.session import get_session
from mlsentry.integrations.mlflow_client import (
    MLflowCircuitBreakerOpenError,
    MLflowClient,
    MLflowModelMetadata,
    MLflowUnavailableError,
)

TEST_API_KEY = "test-secret-key-12345"


@pytest.fixture
def test_db_session():
    """In-memory SQLite session with all tables created."""
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
    """FastAPI TestClient with mounted models router and auth dependency."""
    app = FastAPI()
    app.state.settings = Settings(
        mlsentry_api_key=TEST_API_KEY,
        database_url="sqlite:///:memory:",
    )
    app.state.mlflow_client = MLflowClient()
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
    api_v1.include_router(models_router)
    app.include_router(api_v1)

    return TestClient(app)


def make_valid_registration_payload() -> dict[str, Any]:
    return {
        "name": "fraud-detector",
        "version": "1.2.0",
        "prediction_type": "binary",
        "features": [
            {
                "feature_name": "income",
                "dtype": "float",
                "required": True,
                "min_value": 0.0,
                "max_value": 1000000.0,
                "baseline_stats": {
                    "mean": 55000.0,
                    "std": 12000.0,
                    "min": 1000.0,
                    "max": 950000.0,
                    "p25": 40000.0,
                    "p50": 52000.0,
                    "p75": 68000.0,
                    "p95": 90000.0,
                    "histogram_bin_edges": [0.0, 25000.0, 50000.0, 75000.0, 100000.0],
                    "histogram_counts": [50, 200, 180, 70],
                    "sample_count": 500,
                },
            },
            {
                "feature_name": "segment",
                "dtype": "category",
                "required": True,
                "baseline_stats": {
                    "frequency_map": {
                        "retail": 0.60,
                        "corporate": 0.35,
                        "institutional": 0.05,
                    },
                    "sample_count": 500,
                },
            },
            {
                "feature_name": "notes",
                "dtype": "string",
                "required": False,
                "max_length": 255,
                "baseline_stats": {},
            },
        ],
        "baseline_f1": 0.91,
        "baseline_auc": 0.88,
        "description": "Fraud detection model v1.2.0",
    }


class TestModelRegistrationRoute:
    """Test suite for POST /v1/models/register."""

    def test_successful_registration(self, client: TestClient, test_db_session: Session) -> None:
        mock_meta = MLflowModelMetadata(
            name="fraud-detector",
            version="1.2.0",
            run_id="test-run-123",
            description=None,
            tags={},
            baseline_stats=None,
        )
        with patch.object(MLflowClient, "fetch_model_metadata", return_value=mock_meta):
            payload = make_valid_registration_payload()
            response = client.post(
                "/v1/models/register",
                json=payload,
                headers={"X-API-Key": TEST_API_KEY},
            )
            assert response.status_code == 201
            data = response.json()
            assert "model_id" in data
            assert data["status"] == "warming_up"
            assert data["sample_count"] == 0
            assert data["warm_up_threshold"] == 30

            # Verify DB records
            model_id = uuid.UUID(data["model_id"])
            model = test_db_session.query(ModelRecord).filter_by(model_id=model_id).first()
            assert model is not None
            assert model.name == "fraud-detector"
            assert model.version == "1.2.0"
            assert model.prediction_type == PredictionType.BINARY
            assert model.status == ModelStatus.WARMING_UP

            schemas = test_db_session.query(ModelSchemaRecord).filter_by(model_id=model_id).all()
            assert len(schemas) == 3

            stats = test_db_session.query(ReferenceStatRecord).filter_by(model_id=model_id).all()
            assert len(stats) >= 10

    def test_duplicate_registration_returns_409_conflict(self, client: TestClient) -> None:
        mock_meta = MLflowModelMetadata(
            name="fraud-detector",
            version="1.2.0",
            run_id="test-run-123",
            description=None,
            tags={},
            baseline_stats=None,
        )
        with patch.object(MLflowClient, "fetch_model_metadata", return_value=mock_meta):
            payload = make_valid_registration_payload()
            res1 = client.post(
                "/v1/models/register",
                json=payload,
                headers={"X-API-Key": TEST_API_KEY},
            )
            assert res1.status_code == 201

            res2 = client.post(
                "/v1/models/register",
                json=payload,
                headers={"X-API-Key": TEST_API_KEY},
            )
            assert res2.status_code == 409
            err = res2.json()
            assert err["error"] == "CONFLICT"
            assert "already registered" in err["message"]

    def test_mlflow_unavailable_returns_503_and_rolls_back(
        self, client: TestClient, test_db_session: Session
    ) -> None:
        with patch.object(
            MLflowClient,
            "fetch_model_metadata",
            side_effect=MLflowUnavailableError("MLflow service unreachable"),
        ):
            payload = make_valid_registration_payload()
            payload["name"] = "unreachable-model"
            response = client.post(
                "/v1/models/register",
                json=payload,
                headers={"X-API-Key": TEST_API_KEY},
            )
            assert response.status_code == 503
            err = response.json()
            assert err["error"] == "MLFLOW_REGISTRATION_FAILED"

            # Verify atomic DB rollback
            model = test_db_session.query(ModelRecord).filter_by(name="unreachable-model").first()
            assert model is None

    def test_mlflow_circuit_open_returns_503(self, client: TestClient) -> None:
        with patch.object(
            MLflowClient,
            "fetch_model_metadata",
            side_effect=MLflowCircuitBreakerOpenError("Circuit OPEN"),
        ):
            payload = make_valid_registration_payload()
            payload["name"] = "circuit-open-model"
            response = client.post(
                "/v1/models/register",
                json=payload,
                headers={"X-API-Key": TEST_API_KEY},
            )
            assert response.status_code == 503
            assert response.json()["error"] == "MLFLOW_REGISTRATION_FAILED"

    def test_invalid_model_name_pattern_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        payload["name"] = "Fraud Detector! Upper & Spaces"
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "INVALID_MODEL_NAME"

    def test_invalid_model_version_pattern_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        payload["version"] = "v1-beta"
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "INVALID_MODEL_VERSION"

    def test_invalid_frequency_map_sum_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        payload["features"][1]["baseline_stats"]["frequency_map"] = {
            "retail": 0.50,
            "corporate": 0.20,
        }  # Sum = 0.70 != 1.0
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "FREQUENCY_MAP_INVALID"

    def test_min_value_on_category_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        payload["features"][1]["min_value"] = 10.0
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "VALIDATION_ERROR"

    def test_max_length_on_float_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        payload["features"][0]["max_length"] = 100
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "VALIDATION_ERROR"

    def test_empty_features_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        payload["features"] = []
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "VALIDATION_ERROR"

    def test_duplicate_feature_names_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        payload["features"].append(payload["features"][0].copy())
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "VALIDATION_ERROR"

    def test_invalid_histogram_bins_length_mismatch_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        # 5 bin edges requires 4 counts; provide 2 counts
        payload["features"][0]["baseline_stats"]["histogram_counts"] = [50, 200]
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "INVALID_BASELINE_STATS"

    def test_non_monotonic_histogram_bins_returns_422(self, client: TestClient) -> None:
        payload = make_valid_registration_payload()
        # Non-monotonic bin edges
        payload["features"][0]["baseline_stats"]["histogram_bin_edges"] = [0.0, 50000.0, 25000.0, 100000.0]
        payload["features"][0]["baseline_stats"]["histogram_counts"] = [50, 200, 180]
        response = client.post(
            "/v1/models/register",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "INVALID_BASELINE_STATS"
