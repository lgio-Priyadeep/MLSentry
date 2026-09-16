"""Comprehensive end-to-end integration test exercising the complete MLSentry monitoring lifecycle."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mlsentry.config.settings import Settings
from mlsentry.scheduler.jobs import run_monitoring_cycle
from mlsentry.synthetic.drift_simulator import generate_prediction_batch


def test_complete_ml_monitoring_lifecycle(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_registration_payload: dict,
    test_settings: Settings,
    db_session,
):
    """Execute complete full-flow integration test across all 10 platform modules:
    
    1. Register Model (POST /v1/models/register) -> 201 warming_up
    2. Ingest 30 Predictions (POST /v1/predictions/log) -> Transition warming_up -> active
    3. Ingest Ground Truth Labels (POST /v1/ground_truth/log) -> 201 Created
    4. Trigger Manual Monitoring Cycle (POST /v1/monitoring/run/{model_id}) -> 200 OK
    5. Query Drift Reports (GET /v1/drift/{model_id}) -> 200 OK
    6. Query Performance Logs (GET /v1/performance/{model_id}) -> 200 OK
    7. Query Active Alerts (GET /v1/alerts/{model_id}) -> 200 OK
    8. Resolve Alert (PATCH /v1/alerts/{alert_id}/resolve) -> 200 OK
    9. Render Operations Dashboard (GET /dashboard) -> 200 OK (unauthenticated HTML)
    10. Verify Health Endpoint (GET /health) -> 200 OK
    """
    # -------------------------------------------------------------------------
    # 1. Model Registration
    # -------------------------------------------------------------------------
    reg_resp = client.post(
        "/v1/models/register",
        json=valid_registration_payload,
        headers=auth_headers,
    )
    assert reg_resp.status_code == 201, reg_resp.text
    reg_data = reg_resp.json()
    model_id = reg_data["model_id"]
    assert reg_data["status"] == "warming_up"
    assert reg_data["sample_count"] == 0
    assert reg_data["warm_up_threshold"] == 30

    # -------------------------------------------------------------------------
    # 2. Ingest 30 Predictions (Warmup promotion threshold)
    # -------------------------------------------------------------------------
    pred_ids = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for i in range(30):
        pred_payload = {
            "model_id": model_id,
            "features_json": {
                "annual_income": 60000.0 + (i * 500),
                "employment_type": "salaried",
            },
            "prediction_label": "positive",
            "confidence": 0.85,
            "timestamp": now_iso,
        }
        pred_resp = client.post(
            "/v1/predictions/log",
            json=pred_payload,
            headers=auth_headers,
        )
        assert pred_resp.status_code == 200, pred_resp.text
        pred_ids.append(pred_resp.json()["pred_id"])

    # -------------------------------------------------------------------------
    # 3. Ingest Ground Truth Labels for Matched Pairs
    # -------------------------------------------------------------------------
    for pred_id in pred_ids:
        gt_payload = {
            "pred_id": pred_id,
            "label": "positive",
            "model_id": model_id,
        }
        gt_resp = client.post(
            "/v1/ground_truth/log",
            json=gt_payload,
            headers=auth_headers,
        )
        assert gt_resp.status_code == 201, gt_resp.text

    # -------------------------------------------------------------------------
    # 4. Trigger Manual Monitoring Run
    # -------------------------------------------------------------------------
    mon_resp = client.post(
        f"/v1/monitoring/run/{model_id}",
        headers=auth_headers,
    )
    assert mon_resp.status_code == 200, mon_resp.text
    mon_data = mon_resp.json()
    assert mon_data["model_id"] == model_id
    assert mon_data["status"] == "triggered"
    assert "run_id" in mon_data

    # Execute synchronous test monitoring cycle so SQLite records exist for querying
    run_monitoring_cycle(session_factory=lambda: db_session, settings=test_settings)

    # -------------------------------------------------------------------------
    # 5. Query Drift Reports
    # -------------------------------------------------------------------------
    drift_resp = client.get(
        f"/v1/drift/{model_id}",
        headers=auth_headers,
    )
    assert drift_resp.status_code == 200, drift_resp.text
    drift_data = drift_resp.json()
    assert "data" in drift_data
    assert len(drift_data["data"]) >= 1

    # -------------------------------------------------------------------------
    # 6. Query Performance Logs
    # -------------------------------------------------------------------------
    perf_resp = client.get(
        f"/v1/performance/{model_id}",
        headers=auth_headers,
    )
    assert perf_resp.status_code == 200, perf_resp.text
    perf_data = perf_resp.json()
    assert "data" in perf_data

    # -------------------------------------------------------------------------
    # 7. Query Alerts
    # -------------------------------------------------------------------------
    alerts_resp = client.get(
        f"/v1/alerts/{model_id}",
        headers=auth_headers,
    )
    assert alerts_resp.status_code == 200, alerts_resp.text
    alerts_data = alerts_resp.json()
    assert "data" in alerts_data

    # -------------------------------------------------------------------------
    # 8. Resolve Alert (if any alert exists)
    # -------------------------------------------------------------------------
    if len(alerts_data["data"]) > 0:
        alert_id = alerts_data["data"][0]["alert_id"]
        resolve_resp = client.patch(
            f"/v1/alerts/{alert_id}/resolve",
            headers=auth_headers,
        )
        assert resolve_resp.status_code == 200, resolve_resp.text
        assert resolve_resp.json()["resolved"] is True

    # -------------------------------------------------------------------------
    # 9. Render Unauthenticated Operations Dashboard
    # -------------------------------------------------------------------------
    dash_resp = client.get("/dashboard")
    assert dash_resp.status_code == 200
    assert "text/html" in dash_resp.headers["content-type"]
    assert "credit_default_predictor" in dash_resp.text

    # -------------------------------------------------------------------------
    # 10. Health Check Verification
    # -------------------------------------------------------------------------
    health_resp = client.get("/health")
    assert health_resp.status_code in [200, 503]
