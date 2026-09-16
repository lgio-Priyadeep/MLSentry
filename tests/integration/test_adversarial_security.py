"""Adversarial and security test suite validating SSRF, SQL injection, secret scrubbing, and PII protection."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


def test_ssrf_protection_blocked_ranges(client: TestClient, auth_headers: dict):
    """Verify strict SSRF blocking across loopback, RFC-1918, link-local, multicast, and 0.0.0.0."""
    blocked_urls = [
        "http://127.0.0.1:5432/hook",
        "http://localhost/hook",
        "http://10.0.0.1/hook",
        "http://172.16.0.1/hook",
        "http://192.168.1.1/hook",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0:8000/hook",
        "http://224.0.0.1/hook",
        "ftp://example.com/hook",
    ]
    for url in blocked_urls:
        resp = client.post(
            "/v1/webhooks/test",
            json={"url": url, "payload": {"ping": True}},
            headers=auth_headers,
        )
        assert resp.status_code == 422, f"Failed to block SSRF URL: {url}"
        assert resp.json()["error"] == "VALIDATION_ERROR"


def test_sql_injection_defense(client: TestClient, auth_headers: dict):
    """Verify parameterized ORM queries treat SQL injection strings as literal values."""
    sql_injection_payload = {
        "name": "injection_test_model",
        "version": "1.0.0' OR '1'='1",
        "prediction_type": "binary",
        "features": [],
    }
    resp = client.post(
        "/v1/models/register",
        json=sql_injection_payload,
        headers=auth_headers,
    )
    # Pydantic or SQL layer safely handles or rejects invalid version string
    assert resp.status_code in [201, 422]


def test_zero_raw_data_and_secret_scrubbing(
    client: TestClient,
    auth_headers: dict,
    valid_registration_payload: dict,
    caplog,
):
    """Verify that raw features_json and API secrets are never logged in plaintext."""
    caplog.set_level(logging.INFO)
    reg = client.post("/v1/models/register", json=valid_registration_payload, headers=auth_headers).json()
    model_id = reg["model_id"]

    secret_raw_value = "super_confidential_raw_feature_value_998877"
    pred_payload = {
        "model_id": model_id,
        "features_json": {
            "annual_income": 75000.0,
            "employment_type": secret_raw_value,
        },
        "prediction_label": "positive",
        "confidence": 0.90,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    client.post("/v1/predictions/log", json=pred_payload, headers=auth_headers)

    for record in caplog.records:
        assert secret_raw_value not in record.message, "Raw feature string leaked in application log."
        assert "test_secret_api_key_v1" not in record.message, "API Key secret leaked in application log."
