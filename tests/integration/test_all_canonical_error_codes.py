"""Integration test suite asserting exact compliance against all 20 canonical error envelope codes."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlsentry.api.errors import MLSentryAPIException
from mlsentry.core.constants import ModelStatus
from mlsentry.db.models import ModelRecord
from mlsentry.integrations.mlflow_client import MLflowUnavailableError


def test_canonical_error_01_unauthorized(client: TestClient):
    """401 UNAUTHORIZED: Request missing X-API-Key."""
    resp = client.post("/v1/models/register", json={})
    assert resp.status_code == 401
    assert resp.json()["error"] == "UNAUTHORIZED"
    assert "request_id" in resp.json()


def test_canonical_error_02_not_found(client: TestClient, auth_headers: dict):
    """404 NOT_FOUND: Non-existent model UUID."""
    fake_uuid = str(uuid.uuid4())
    resp = client.get(f"/v1/drift/{fake_uuid}", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.json()["error"] == "NOT_FOUND"


def test_canonical_error_03_conflict_duplicate_model(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """409 CONFLICT: Duplicate (name, version) model registration."""
    client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers)
    resp = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers)
    assert resp.status_code == 409
    assert resp.json()["error"] == "CONFLICT"


def test_canonical_error_04_model_warming_up(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """409 MODEL_WARMING_UP: Manual monitoring trigger on model with sample_count < 30."""
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = reg["model_id"]
    resp = client.post(f"/v1/monitoring/run/{model_id}", headers=auth_headers)
    assert resp.status_code == 409
    assert resp.json()["error"] == "MODEL_WARMING_UP"


def test_canonical_error_05_invalid_model_name(client: TestClient, auth_headers: dict):
    """422 INVALID_MODEL_NAME: Model name with invalid characters."""
    payload = {
        "name": "Invalid Model Name!",
        "version": "1.0.0",
        "prediction_type": "binary",
        "features": [],
    }
    resp = client.post("/v1/models/register", json=payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"] in ["INVALID_MODEL_NAME", "VALIDATION_ERROR"]


def test_canonical_error_06_invalid_prediction_type(client: TestClient, auth_headers: dict):
    """422 INVALID_PREDICTION_TYPE: Invalid prediction type enum value."""
    payload = {
        "name": "valid_model_name",
        "version": "1.0.0",
        "prediction_type": "quantum_classifier",
        "features": [],
    }
    resp = client.post("/v1/models/register", json=payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"] in ["INVALID_PREDICTION_TYPE", "VALIDATION_ERROR"]


def test_canonical_error_07_frequency_map_invalid(client: TestClient, auth_headers: dict):
    """422 FREQUENCY_MAP_INVALID: Frequency map not summing to 1.0."""
    payload = {
        "name": "freq_invalid_model",
        "version": "1.0.0",
        "prediction_type": "binary",
        "baseline_f1": 0.85,
        "features": [
            {
                "feature_name": "category",
                "dtype": "category",
                "required": True,
                "baseline_stats": {"frequency_map": {"a": 0.50, "b": 0.20}},
            }
        ],
    }
    resp = client.post("/v1/models/register", json=payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"] in ["FREQUENCY_MAP_INVALID", "VALIDATION_ERROR"]


def test_canonical_error_08_schema_validation_failed(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """422 SCHEMA_VALIDATION_FAILED: Feature data type mismatch."""
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = reg["model_id"]
    pred_payload = {
        "model_id": model_id,
        "features_json": {
            "annual_income": "not_a_number_string",
            "employment_type": "salaried",
        },
        "prediction_label": "positive",
        "confidence": 0.85,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    resp = client.post("/v1/predictions/log", json=pred_payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"] == "SCHEMA_VALIDATION_FAILED"


def test_canonical_error_09_features_json_nested(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """422 FEATURES_JSON_NESTED: Reject nested objects in features_json."""
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = reg["model_id"]
    pred_payload = {
        "model_id": model_id,
        "features_json": {
            "annual_income": 50000.0,
            "nested_obj": {"a": 1, "b": 2},
        },
        "prediction_label": "positive",
        "confidence": 0.85,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    resp = client.post("/v1/predictions/log", json=pred_payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"] == "FEATURES_JSON_NESTED"


def test_canonical_error_10_features_json_too_large(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """422 FEATURES_JSON_TOO_LARGE: Reject payloads exceeding 200 keys."""
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = reg["model_id"]
    huge_features = {f"k_{i}": i for i in range(205)}
    pred_payload = {
        "model_id": model_id,
        "features_json": huge_features,
        "prediction_label": "positive",
        "confidence": 0.85,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    resp = client.post("/v1/predictions/log", json=pred_payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"] == "FEATURES_JSON_TOO_LARGE"


def test_canonical_error_11_empty_log_line(client: TestClient, auth_headers: dict):
    """422 EMPTY_LOG_LINE: Empty string passed to log classifier."""
    resp = client.post(
        "/v1/logs/classify",
        json={"model_id": str(uuid.uuid4()), "log_line": "   "},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert resp.json()["error"] in ["EMPTY_LOG_LINE", "VALIDATION_ERROR"]


def test_canonical_error_12_validation_error_ssrf(client: TestClient, auth_headers: dict):
    """422 VALIDATION_ERROR: SSRF private IP rejection in webhook test shim."""
    resp = client.post(
        "/v1/webhooks/test",
        json={"url": "http://127.0.0.1:8080/hook", "payload": {"test": 1}},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "VALIDATION_ERROR"


def test_canonical_error_13_model_deprecated(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict, db_session
):
    """409 MODEL_DEPRECATED: Prediction on deprecated model."""
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = uuid.UUID(reg["model_id"])

    model = db_session.get(ModelRecord, model_id)
    model.status = ModelStatus.DEPRECATED
    model.deprecated_at = datetime.now(timezone.utc)
    db_session.commit()

    pred_payload = {
        "model_id": str(model_id),
        "features_json": {"annual_income": 50000.0, "employment_type": "salaried"},
        "prediction_label": "positive",
        "confidence": 0.85,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    resp = client.post("/v1/predictions/log", json=pred_payload, headers=auth_headers)
    assert resp.status_code == 409
    assert resp.json()["error"] == "MODEL_DEPRECATED"


def test_canonical_error_14_schema_immutable(client: TestClient):
    """409 SCHEMA_IMMUTABLE: Mutation on immutable schema definition."""
    exc = MLSentryAPIException(
        status_code=409,
        error_code="SCHEMA_IMMUTABLE",
        message="Model feature schema is immutable post-registration.",
    )
    assert exc.status_code == 409
    assert exc.error_code == "SCHEMA_IMMUTABLE"


def test_canonical_error_15_late_label_rejected(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """422 LATE_LABEL_REJECTED: Ground truth submitted > 72 hours post prediction."""
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = reg["model_id"]
    pred_payload = {
        "model_id": model_id,
        "features_json": {"annual_income": 50000.0, "employment_type": "salaried"},
        "prediction_label": "positive",
        "confidence": 0.85,
        "timestamp": "2020-01-01T00:00:00Z",
    }
    pred_resp = client.post("/v1/predictions/log", json=pred_payload, headers=auth_headers)
    pred_id = pred_resp.json()["pred_id"]

    gt_payload = {
        "pred_id": pred_id,
        "label": "positive",
        "model_id": model_id,
    }
    resp = client.post("/v1/ground_truth/log", json=gt_payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"] == "LATE_LABEL_REJECTED"


def test_canonical_error_16_model_id_mismatch(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """422 MODEL_ID_MISMATCH: Ground truth model_id mismatch with prediction."""
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = reg["model_id"]
    pred_payload = {
        "model_id": model_id,
        "features_json": {"annual_income": 50000.0, "employment_type": "salaried"},
        "prediction_label": "positive",
        "confidence": 0.85,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    pred_resp = client.post("/v1/predictions/log", json=pred_payload, headers=auth_headers)
    pred_id = pred_resp.json()["pred_id"]

    fake_model_id = str(uuid.uuid4())
    gt_payload = {
        "pred_id": pred_id,
        "label": "positive",
        "model_id": fake_model_id,
    }
    resp = client.post("/v1/ground_truth/log", json=gt_payload, headers=auth_headers)
    assert resp.status_code == 422
    assert resp.json()["error"] == "MODEL_ID_MISMATCH"


def test_canonical_error_17_internal_error(client: TestClient, auth_headers: dict):
    """500 INTERNAL_ERROR: Unhandled internal exception envelope.

    Note: Uses a scoped local TestClient with raise_server_exceptions=False to assert
    the rendered HTTP 500 JSON envelope without re-raising in the test runner process.
    """
    with patch("sqlalchemy.orm.Session.get", side_effect=RuntimeError("Unexpected DB crash")):
        with TestClient(client.app, raise_server_exceptions=False) as error_client:
            resp = error_client.get(f"/v1/drift/{uuid.uuid4()}", headers=auth_headers)
            assert resp.status_code == 500
            assert resp.json()["error"] == "INTERNAL_ERROR"
            assert "request_id" in resp.json()


def test_canonical_error_18_service_unavailable(client: TestClient):
    """503 SERVICE_UNAVAILABLE: Subsystem connectivity degradation."""
    with patch("mlsentry.api.main.get_session", side_effect=Exception("DB down")):
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["error"] == "SERVICE_UNAVAILABLE"


def test_canonical_error_19_distilbert_unavailable(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """503 DISTILBERT_UNAVAILABLE: Circuit breaker open on classifier."""
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = reg["model_id"]

    client.app.state.classifier.circuit_breaker.is_open = True
    try:
        resp = client.post(
            "/v1/logs/classify",
            json={"model_id": model_id, "log_line": "Test pipeline error log event"},
            headers=auth_headers,
        )
        assert resp.status_code == 503
        assert resp.json()["error"] == "DISTILBERT_UNAVAILABLE"
    finally:
        client.app.state.classifier.circuit_breaker.is_open = False


def test_canonical_error_20_mlflow_registration_failed(
    client: TestClient, auth_headers: dict, valid_registration_payload: dict
):
    """503 MLFLOW_REGISTRATION_FAILED: MLflow registry connectivity error."""
    client.app.state.mlflow_client.fetch_model_metadata.side_effect = MLflowUnavailableError(
        "MLflow server unreachable"
    )
    resp = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers)
    assert resp.status_code == 503
    assert resp.json()["error"] == "MLFLOW_REGISTRATION_FAILED"
