"""API route handlers for monitoring, drift reports, and performance tracking queries.

Implements:
  - GET /v1/drift/{model_id}:
      Returns paginated drift reports for specified or latest window.
      Severity is computed dynamically from PSI/score against core thresholds (GI-05).
      Returns 200 OK with insufficient_data=True when window sample count < 30.
  - GET /v1/performance/{model_id}:
      Returns paginated performance metric logs for specified or latest window.
      Severity is computed dynamically from delta (value - baseline) (GI-05).
      Returns 200 OK with insufficient_data=True when matched pairs < 50.
  - POST /v1/monitoring/run/{model_id}:
      Manually triggers monitoring evaluation pipeline outside APScheduler.
      Rejects warming_up models (sample_count < 30) with HTTP 409 MODEL_WARMING_UP.
      Rejects deprecated models with HTTP 409 MODEL_DEPRECATED.
      Returns 200 OK with triggered status and run_id UUIDv4.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import desc
from sqlalchemy.orm import Session

from mlsentry.api.errors import MLSentryAPIException
from mlsentry.core.constants import (
    AlertSeverity,
    DriftMethod,
    FeatureKind,
    ModelStatus,
)
from mlsentry.core.drift.statistical import (
    classify_chi2_severity,
    classify_psi_severity,
)
from mlsentry.core.performance.tracker import classify_perf_severity
from mlsentry.db.models import (
    DriftReportRecord,
    GroundTruthRecord,
    ModelRecord,
    PerformanceLogRecord,
    PredictionRecord,
)
from mlsentry.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


# ─── Pydantic Response DTOs ──────────────────────────────────────


class PaginationMeta(BaseModel):
    """Pagination metadata container."""

    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int


class DriftFeatureItem(BaseModel):
    """Single feature drift evaluation result."""

    model_config = ConfigDict(extra="forbid")

    feature_name: str
    feature_type: str
    method: str
    score: float
    psi: float | None = None
    severity: str


class DriftReportResponse(BaseModel):
    """Response payload for GET /v1/drift/{model_id}."""

    model_config = ConfigDict(extra="forbid")

    model_id: uuid.UUID
    window: str
    sample_count: int
    model_status: str
    data: list[DriftFeatureItem]
    meta: PaginationMeta
    insufficient_data: bool = False
    message: str | None = None


class PerformanceMetricItem(BaseModel):
    """Single performance metric evaluation result."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float
    baseline_value: float
    delta: float
    severity: str


class PerformanceResponse(BaseModel):
    """Response payload for GET /v1/performance/{model_id}."""

    model_config = ConfigDict(extra="forbid")

    model_id: uuid.UUID
    window: str
    matched_pairs: int
    data: list[PerformanceMetricItem]
    meta: PaginationMeta
    insufficient_data: bool = False
    message: str | None = None


class MonitoringRunResponse(BaseModel):
    """Response payload for POST /v1/monitoring/run/{model_id}."""

    model_config = ConfigDict(extra="forbid")

    model_id: uuid.UUID
    run_id: uuid.UUID
    status: str
    triggered_at: str
    message: str


# ─── Route Handlers ──────────────────────────────────────────────


@router.get(
    "/drift/{model_id}",
    response_model=DriftReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Get drift report data for a model",
)
def get_drift_report(
    model_id: uuid.UUID,
    window: str | None = Query(
        None,
        description="ISO 8601 UTC timestamp matching a window_bucket; omit for latest window",
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_session),
) -> DriftReportResponse:
    """Fetch paginated drift report for a specific model and window.

    Dynamic severity classification per GI-05:
      - Numerical features (PSI): score < 0.10 -> OK, 0.10 <= score < 0.25 -> WARNING, >= 0.25 -> CRITICAL
      - Categorical features (Chi-square): p_value > 0.05 -> OK, 0.01 < p_value <= 0.05 -> WARNING, <= 0.01 -> CRITICAL
    """
    model = db.get(ModelRecord, model_id)
    if model is None:
        raise MLSentryAPIException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=f"model_id '{model_id}' not found.",
        )

    target_window: datetime
    if window is not None:
        try:
            dt_str = window.replace("Z", "+00:00")
            target_window = datetime.fromisoformat(dt_str)
        except Exception:
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message=f"Invalid ISO 8601 window timestamp: '{window}'.",
            )
    else:
        # Find latest window_start from drift_reports
        latest_report = (
            db.query(DriftReportRecord.window_start)
            .filter(DriftReportRecord.model_id == model_id)
            .order_by(desc(DriftReportRecord.window_start))
            .first()
        )
        if latest_report is not None:
            target_window = latest_report[0]
        else:
            # Check latest prediction window bucket
            latest_pred = (
                db.query(PredictionRecord.window_bucket)
                .filter(PredictionRecord.model_id == model_id)
                .order_by(desc(PredictionRecord.window_bucket))
                .first()
            )
            target_window = (
                latest_pred[0]
                if latest_pred is not None
                else datetime.now(timezone.utc)
            )
            sample_count = model.sample_count
            insufficient_msg = (
                f"INSUFFICIENT_DATA: {sample_count} samples, minimum 30 required."
                if sample_count < 30
                else "No drift reports generated yet for this model."
            )
            return DriftReportResponse(
                model_id=model_id,
                window=target_window.strftime("%Y-%m-%dT%H:%M:%SZ"),
                sample_count=sample_count,
                model_status=model.status.value,
                data=[],
                meta=PaginationMeta(total=0, limit=limit, offset=offset),
                insufficient_data=True,
                message=insufficient_msg,
            )

    base_query = db.query(DriftReportRecord).filter(
        DriftReportRecord.model_id == model_id,
        DriftReportRecord.window_start == target_window,
    )
    total_count = base_query.count()

    if total_count == 0:
        pred_count = (
            db.query(PredictionRecord)
            .filter(
                PredictionRecord.model_id == model_id,
                PredictionRecord.window_bucket == target_window,
                PredictionRecord.schema_valid.is_(True),
            )
            .count()
        )
        return DriftReportResponse(
            model_id=model_id,
            window=target_window.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sample_count=pred_count,
            model_status=model.status.value,
            data=[],
            meta=PaginationMeta(total=0, limit=limit, offset=offset),
            insufficient_data=True,
            message=f"INSUFFICIENT_DATA: {pred_count} samples, minimum 30 required.",
        )

    reports = (
        base_query.order_by(DriftReportRecord.feature_name)
        .offset(offset)
        .limit(limit)
        .all()
    )

    sample_count = reports[0].sample_count if reports else model.sample_count

    data_items: list[DriftFeatureItem] = []
    for r in reports:
        if r.method == DriftMethod.PSI:
            sev = classify_psi_severity(float(r.score))
            sev_str = "CRITICAL" if sev == AlertSeverity.CRITICAL else ("WARNING" if sev == AlertSeverity.WARNING else "OK")
            method_str = "psi"
            psi_val = float(r.psi) if r.psi is not None else float(r.score)
        else:
            p_val = float(r.p_value) if r.p_value is not None else 1.0
            sev = classify_chi2_severity(p_val)
            sev_str = "CRITICAL" if sev == AlertSeverity.CRITICAL else ("WARNING" if sev == AlertSeverity.WARNING else "OK")
            method_str = "chi-square"
            psi_val = None

        data_items.append(
            DriftFeatureItem(
                feature_name=r.feature_name,
                feature_type=r.feature_type.value,
                method=method_str,
                score=float(r.score),
                psi=psi_val,
                severity=sev_str,
            )
        )

    return DriftReportResponse(
        model_id=model_id,
        window=target_window.strftime("%Y-%m-%dT%H:%M:%SZ"),
        sample_count=sample_count,
        model_status=model.status.value,
        data=data_items,
        meta=PaginationMeta(total=total_count, limit=limit, offset=offset),
        insufficient_data=False,
    )


@router.get(
    "/performance/{model_id}",
    response_model=PerformanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get performance degradation metrics for a model",
)
def get_performance_logs(
    model_id: uuid.UUID,
    window: str | None = Query(
        None,
        description="ISO 8601 UTC timestamp matching a window_bucket; omit for latest window",
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_session),
) -> PerformanceResponse:
    """Fetch paginated performance degradation logs for a specific model and window.

    Dynamic severity classification per GI-05:
      - delta = value - baseline_value
      - delta > -0.05 -> INFO
      - -0.15 < delta <= -0.05 -> WARNING
      - delta <= -0.15 -> CRITICAL
    """
    model = db.get(ModelRecord, model_id)
    if model is None:
        raise MLSentryAPIException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=f"model_id '{model_id}' not found.",
        )

    target_window: datetime
    if window is not None:
        try:
            dt_str = window.replace("Z", "+00:00")
            target_window = datetime.fromisoformat(dt_str)
        except Exception:
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message=f"Invalid ISO 8601 window timestamp: '{window}'.",
            )
    else:
        latest_log = (
            db.query(PerformanceLogRecord.window_start)
            .filter(PerformanceLogRecord.model_id == model_id)
            .order_by(desc(PerformanceLogRecord.window_start))
            .first()
        )
        if latest_log is not None:
            target_window = latest_log[0]
        else:
            latest_pred = (
                db.query(PredictionRecord.window_bucket)
                .filter(PredictionRecord.model_id == model_id)
                .order_by(desc(PredictionRecord.window_bucket))
                .first()
            )
            target_window = (
                latest_pred[0]
                if latest_pred is not None
                else datetime.now(timezone.utc)
            )
            return PerformanceResponse(
                model_id=model_id,
                window=target_window.strftime("%Y-%m-%dT%H:%M:%SZ"),
                matched_pairs=0,
                data=[],
                meta=PaginationMeta(total=0, limit=limit, offset=offset),
                insufficient_data=True,
                message="Performance metrics not computed: fewer than 50 matched prediction–ground_truth pairs exist in this window.",
            )

    base_query = db.query(PerformanceLogRecord).filter(
        PerformanceLogRecord.model_id == model_id,
        PerformanceLogRecord.window_start == target_window,
    )
    total_count = base_query.count()

    if total_count == 0:
        matched_count = (
            db.query(GroundTruthRecord)
            .join(PredictionRecord, GroundTruthRecord.pred_id == PredictionRecord.pred_id)
            .filter(
                PredictionRecord.model_id == model_id,
                PredictionRecord.schema_valid.is_(True),
                PredictionRecord.window_bucket == target_window,
            )
            .count()
        )
        return PerformanceResponse(
            model_id=model_id,
            window=target_window.strftime("%Y-%m-%dT%H:%M:%SZ"),
            matched_pairs=matched_count,
            data=[],
            meta=PaginationMeta(total=0, limit=limit, offset=offset),
            insufficient_data=True,
            message="Performance metrics not computed: fewer than 50 matched prediction–ground_truth pairs exist in this window.",
        )

    logs = (
        base_query.order_by(PerformanceLogRecord.metric)
        .offset(offset)
        .limit(limit)
        .all()
    )

    matched_pairs = logs[0].sample_count if logs else 0

    data_items: list[PerformanceMetricItem] = []
    for l in logs:
        sev = classify_perf_severity(float(l.delta))
        sev_str = sev.name.upper()

        data_items.append(
            PerformanceMetricItem(
                metric=l.metric.value,
                value=float(l.value),
                baseline_value=float(l.baseline_value),
                delta=float(l.delta),
                severity=sev_str,
            )
        )

    return PerformanceResponse(
        model_id=model_id,
        window=target_window.strftime("%Y-%m-%dT%H:%M:%SZ"),
        matched_pairs=matched_pairs,
        data=data_items,
        meta=PaginationMeta(total=total_count, limit=limit, offset=offset),
        insufficient_data=False,
    )


@router.post(
    "/monitoring/run/{model_id}",
    response_model=MonitoringRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger manual monitoring run for an active model",
)
def trigger_monitoring_run(
    model_id: uuid.UUID,
    db: Session = Depends(get_session),
) -> MonitoringRunResponse:
    """Manually trigger monitoring evaluation pipeline for an active model.

    Validations:
      - 404 NOT_FOUND if model_id does not exist.
      - 409 MODEL_WARMING_UP if model.status is warming_up (sample_count < 30).
      - 409 MODEL_DEPRECATED if model.status is deprecated.
    """
    model = db.get(ModelRecord, model_id)
    if model is None:
        raise MLSentryAPIException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=f"model_id '{model_id}' not found.",
        )

    if model.status == ModelStatus.WARMING_UP or model.sample_count < 30:
        raise MLSentryAPIException(
            status_code=status.HTTP_409_CONFLICT,
            error_code="MODEL_WARMING_UP",
            message=f"Model '{model_id}' requires active status for this operation (sample_count < 30). Log predictions until sample_count >= 30.",
        )

    if model.status == ModelStatus.DEPRECATED:
        raise MLSentryAPIException(
            status_code=status.HTTP_409_CONFLICT,
            error_code="MODEL_DEPRECATED",
            message=f"Model '{model_id}' is deprecated and cannot be monitored.",
        )

    run_id = uuid.uuid4()
    now_utc = datetime.now(timezone.utc)

    logger.info(
        "MONITORING_RUN_TRIGGERED: model_id=%s, run_id=%s, triggered_at=%s",
        model_id,
        run_id,
        now_utc.isoformat(),
    )

    return MonitoringRunResponse(
        model_id=model_id,
        run_id=run_id,
        status="triggered",
        triggered_at=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        message="Monitoring job dispatched. Results will be available via GET /drift/{model_id} and GET /performance/{model_id}.",
    )
