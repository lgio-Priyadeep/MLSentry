"""API route handlers for live prediction ingestion and delayed ground truth labeling.

Implements:
  - POST /v1/predictions/log:
      * Validates feature payload envelope and dtype constraints (hard violations short-circuit with 422).
      * Evaluates soft violations (missing required, bounds, extra features) and emits alerts with cooldown.
      * Computes 15-minute window_bucket at application layer.
      * Atomic insert of PredictionRecord + sample_count increment (promotes warming_up to active at 30).
      * Zero raw features returned or exposed in logs.
  - POST /v1/ground_truth/log:
      * Matches ground truth label to prediction by pred_id (sole join key).
      * Enforces 72-hour cutoff window (rejects > 72h arrivals with 422 LATE_LABEL_REJECTED).
      * Validates optional model_id against prediction.model_id (422 MODEL_ID_MISMATCH).
      * Enforces write-once immutability (409 CONFLICT on duplicate pred_id).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from mlsentry.api.errors import MLSentryAPIException
from mlsentry.core.constants import (
    ALERT_COOLDOWN_SCHEMA_VIOLATION_MINUTES,
    ALERT_COOLDOWN_SCHEMA_WARNING_MINUTES,
    GROUND_TRUTH_CUTOFF_HOURS,
    WARM_UP_THRESHOLD,
    AlertSeverity,
    AlertType,
    ModelStatus,
)
from mlsentry.core.schema.validator import (
    FeatureSchema,
    ViolationType,
    validate_features,
)
from mlsentry.db.models import (
    AlertRecord,
    GroundTruthRecord,
    ModelRecord,
    ModelSchemaRecord,
    PredictionRecord,
)
from mlsentry.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


class PredictionLogRequest(BaseModel):
    """Payload for POST /v1/predictions/log."""

    model_id: uuid.UUID
    features_json: dict[str, Any]
    prediction_label: Any
    confidence: float
    timestamp: datetime

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message="confidence must be in range [0.0, 1.0].",
            )
        return v

    @field_validator("prediction_label")
    @classmethod
    def validate_prediction_label(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and len(v.strip()) == 0):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message="prediction_label must be a non-empty scalar value.",
            )
        return str(v)[:50]


class PredictionLogResponse(BaseModel):
    """Response payload for POST /v1/predictions/log."""

    pred_id: uuid.UUID


class GroundTruthLogRequest(BaseModel):
    """Payload for POST /v1/ground_truth/log."""

    pred_id: uuid.UUID
    label: Any
    model_id: uuid.UUID | None = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and len(v.strip()) == 0):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message="label must be a non-empty scalar value.",
            )
        return str(v)[:50]


class GroundTruthLogResponse(BaseModel):
    """Response payload for POST /v1/ground_truth/log."""

    gt_id: uuid.UUID
    pred_id: uuid.UUID
    received_at: str


@router.post(
    "/predictions/log",
    status_code=status.HTTP_200_OK,
    response_model=PredictionLogResponse,
)
async def log_prediction(
    payload: PredictionLogRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> Any:
    """Log an inference prediction event and validate schema constraints."""
    # 1. Lookup model
    model = (
        session.query(ModelRecord)
        .filter(ModelRecord.model_id == payload.model_id)
        .first()
    )
    if model is None:
        raise MLSentryAPIException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=f"model_id '{payload.model_id}' not found.",
        )

    if model.status == ModelStatus.DEPRECATED:
        raise MLSentryAPIException(
            status_code=status.HTTP_409_CONFLICT,
            error_code="MODEL_DEPRECATED",
            message=(
                f"Model '{model.name}' v{model.version} is deprecated. "
                f"No new predictions, ground truth, or log classifications are accepted."
            ),
        )

    # 2. Fetch schema rules
    schema_records = (
        session.query(ModelSchemaRecord)
        .filter(ModelSchemaRecord.model_id == payload.model_id)
        .all()
    )
    feature_schemas = [
        FeatureSchema(
            feature_name=s.feature_name,
            dtype=s.dtype,
            required=s.required,
            min_value=float(s.min_value) if s.min_value is not None else None,
            max_value=float(s.max_value) if s.max_value is not None else None,
            allowed_values=s.allowed_values,
        )
        for s in schema_records
    ]

    # 3. Validate features
    val_result = validate_features(payload.features_json, feature_schemas)
    if val_result.is_hard_violation:
        for hv in val_result.hard_violations:
            if hv.violation_type == ViolationType.HARD_TOO_LARGE:
                raise MLSentryAPIException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    error_code="FEATURES_JSON_TOO_LARGE",
                    message="features_json exceeds the maximum of 64 KB or 200 keys.",
                )
            elif hv.violation_type == ViolationType.HARD_NESTED:
                raise MLSentryAPIException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    error_code="FEATURES_JSON_NESTED",
                    message="features_json contains nested objects or arrays; only scalar values allowed.",
                )
            elif hv.violation_type == ViolationType.HARD_WRONG_DTYPE:
                raise MLSentryAPIException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    error_code="SCHEMA_VALIDATION_FAILED",
                    message="One or more features violate registered data type constraints.",
                )

    now = datetime.now(timezone.utc)

    # 4. Handle soft violations and alert emission with cooldown deduplication
    for sv in val_result.soft_violations:
        if sv.violation_type in (
            ViolationType.SOFT_MISSING_REQUIRED,
            ViolationType.SOFT_OUT_OF_BOUNDS,
        ):
            alert_type = AlertType.SCHEMA_VIOLATION
            cooldown_mins = ALERT_COOLDOWN_SCHEMA_VIOLATION_MINUTES
        else:
            alert_type = AlertType.SCHEMA_WARNING
            cooldown_mins = ALERT_COOLDOWN_SCHEMA_WARNING_MINUTES

        # Check for active unresolved alert within cooldown
        existing_alert = (
            session.query(AlertRecord)
            .filter(
                AlertRecord.model_id == payload.model_id,
                AlertRecord.type == alert_type,
                AlertRecord.feature_name == sv.feature_name,
                AlertRecord.resolved.is_(False),
                AlertRecord.cooldown_until > now,
            )
            .first()
        )

        if existing_alert is None:
            cooldown_until = now + timedelta(minutes=cooldown_mins)
            alert = AlertRecord(
                alert_id=uuid.uuid4(),
                model_id=payload.model_id,
                type=alert_type,
                severity=AlertSeverity.WARNING,
                feature_name=sv.feature_name,
                message=sv.message,
                triggered_at=now,
                cooldown_until=cooldown_until,
            )
            session.add(alert)

    # 5. Compute 15-minute window bucket
    ts = (
        payload.timestamp
        if payload.timestamp.tzinfo
        else payload.timestamp.replace(tzinfo=timezone.utc)
    )
    bucket_minute = (ts.minute // 15) * 15
    window_bucket = ts.replace(minute=bucket_minute, second=0, microsecond=0)

    # 6. Insert prediction and update model sample count atomically
    pred_id = uuid.uuid4()
    prediction_rec = PredictionRecord(
        pred_id=pred_id,
        model_id=payload.model_id,
        model_version=model.version,
        features_json=payload.features_json,
        prediction_label=str(payload.prediction_label),
        confidence=payload.confidence,
        schema_valid=val_result.schema_valid,
        window_bucket=window_bucket,
        logged_at=ts,
    )
    session.add(prediction_rec)

    # Increment sample count and check warm-up threshold
    model.sample_count = model.sample_count + 1
    if (
        model.status == ModelStatus.WARMING_UP
        and model.sample_count >= WARM_UP_THRESHOLD
    ):
        model.status = ModelStatus.ACTIVE
        logger.info(
            "MODEL_PROMOTED_ACTIVE: model_id=%s, sample_count=%d",
            model.model_id,
            model.sample_count,
        )

    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("PREDICTION_LOG_DB_FAILED: %s", exc)
        raise MLSentryAPIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_ERROR",
            message="Failed to record prediction.",
        ) from exc

    return {"pred_id": pred_id}


@router.post(
    "/ground_truth/log",
    status_code=status.HTTP_201_CREATED,
    response_model=GroundTruthLogResponse,
)
async def log_ground_truth(
    payload: GroundTruthLogRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> Any:
    """Associate a delayed ground truth label with a prior inference prediction."""
    # 1. Lookup prediction by pred_id
    pred = (
        session.query(PredictionRecord)
        .filter(PredictionRecord.pred_id == payload.pred_id)
        .first()
    )
    if pred is None:
        raise MLSentryAPIException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=f"pred_id '{payload.pred_id}' not found in predictions.",
        )

    # 2. Check model_id consistency if provided
    if payload.model_id is not None and payload.model_id != pred.model_id:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="MODEL_ID_MISMATCH",
            message="Provided model_id does not match prediction's model_id.",
        )

    # 3. Check model deprecation status (GI-03 & F-15)
    model = (
        session.query(ModelRecord)
        .filter(ModelRecord.model_id == pred.model_id)
        .first()
    )
    if model is not None and model.status == ModelStatus.DEPRECATED:
        raise MLSentryAPIException(
            status_code=status.HTTP_409_CONFLICT,
            error_code="MODEL_DEPRECATED",
            message=(
                f"Model '{model.name}' v{model.version} is deprecated. "
                f"No new predictions, ground truth, or log classifications are accepted."
            ),
        )

    # 4. Check for duplicate ground truth
    existing_gt = (
        session.query(GroundTruthRecord)
        .filter(GroundTruthRecord.pred_id == payload.pred_id)
        .first()
    )
    if existing_gt is not None:
        raise MLSentryAPIException(
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
            message=(
                f"pred_id '{payload.pred_id}' already has a ground truth label. "
                f"Ground truth is immutable once set."
            ),
        )

    # 5. Check 72-hour cutoff rule
    now = datetime.now(timezone.utc)
    pred_ts = (
        pred.logged_at
        if pred.logged_at.tzinfo
        else pred.logged_at.replace(tzinfo=timezone.utc)
    )
    latency_hours = (now - pred_ts).total_seconds() / 3600.0
    if latency_hours > GROUND_TRUTH_CUTOFF_HOURS:
        logger.warning(
            "LATE_LABEL_DROPPED: pred_id=%s, latency_hours=%.2f",
            pred.pred_id,
            latency_hours,
        )
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="LATE_LABEL_REJECTED",
            message=f"Ground truth label arrived > {GROUND_TRUTH_CUTOFF_HOURS} hours post-inference and was rejected.",
        )

    # 6. Insert GroundTruthRecord
    gt_id = uuid.uuid4()
    gt_rec = GroundTruthRecord(
        gt_id=gt_id,
        pred_id=payload.pred_id,
        model_id=pred.model_id,
        true_label=str(payload.label),
        latency_hours=round(max(0.0, latency_hours), 4),
        received_at=now,
    )
    session.add(gt_rec)

    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("GROUND_TRUTH_LOG_DB_FAILED: %s", exc)
        raise MLSentryAPIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_ERROR",
            message="Failed to record ground truth label.",
        ) from exc

    return {
        "gt_id": gt_id,
        "pred_id": payload.pred_id,
        "received_at": now.isoformat(),
    }
