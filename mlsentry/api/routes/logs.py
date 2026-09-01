"""API route handler for NLP log anomaly classification.

Implements POST /v1/logs/classify:
  - Validates non-empty log_line (fast-fails with 422 EMPTY_LOG_LINE without model invocation).
  - Verifies model_id exists and is not deprecated.
  - Delegates CPU-bound DistilBERT inference to thread pool executor.
  - Handles circuit breaker OPEN state with fast-fail HTTP 503 DISTILBERT_UNAVAILABLE.
  - Handles invalid log length with HTTP 422 INVALID_LOG_LINE.
  - Persists classification record and triggers LOG_ANOMALY alert when confidence >= 0.85.
  - Strict zero-raw-data policy: raw log strings are never exposed in logs or error traces.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from mlsentry.api.errors import MLSentryAPIException
from mlsentry.core.anomaly.log_classifier import (
    CircuitBreakerOpenError,
    DistilBERTUnavailableError,
    LogClassifier,
)
from mlsentry.core.constants import (
    LOG_ANOMALY_CONFIDENCE_THRESHOLD,
    AlertSeverity,
    AlertType,
    LogLabel,
    ModelStatus,
)
from mlsentry.db.models import AlertRecord, LogClassificationRecord, ModelRecord
from mlsentry.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["logs"])


class LogClassifyRequest(BaseModel):
    """Payload for POST /v1/logs/classify."""

    model_id: uuid.UUID
    log_line: str


class LogClassifyResponse(BaseModel):
    """Response payload for POST /v1/logs/classify."""

    label: str
    confidence: float


@router.post(
    "/logs/classify",
    status_code=status.HTTP_200_OK,
    response_model=LogClassifyResponse,
)
async def classify_log(
    payload: LogClassifyRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> Any:
    """Classify an unstructured pipeline log line as anomalous or normal."""
    # 1. Reject empty log line before model or DB invocation
    if not payload.log_line or len(payload.log_line.strip()) == 0:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="EMPTY_LOG_LINE",
            message="log_line must not be empty.",
        )

    # 2. Lookup model and status
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

    # 3. Retrieve classifier instance from application state
    classifier: LogClassifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        logger.warning(
            "DISTILBERT_UNINITIALIZED: model_id=%s",
            payload.model_id,
        )
        raise MLSentryAPIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DISTILBERT_UNAVAILABLE",
            message="DistilBERT log anomaly classifier is not initialized.",
        )

    if classifier.circuit_breaker.is_open:
        logger.warning(
            "DISTILBERT_CIRCUIT_FAST_FAIL: model_id=%s",
            payload.model_id,
        )
        raise MLSentryAPIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DISTILBERT_UNAVAILABLE",
            message="DistilBERT log anomaly classifier is temporarily unavailable (circuit breaker OPEN).",
        )

    # 4. Offload CPU-bound inference to thread pool
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            classifier.classify_log,
            payload.log_line,
        )
    except ValueError as exc:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="INVALID_LOG_LINE",
            message=str(exc),
        ) from exc
    except CircuitBreakerOpenError as exc:
        logger.warning(
            "DISTILBERT_CIRCUIT_TRIPPED: model_id=%s",
            payload.model_id,
        )
        raise MLSentryAPIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DISTILBERT_UNAVAILABLE",
            message="DistilBERT log anomaly classifier is temporarily unavailable (circuit breaker OPEN).",
        ) from exc
    except DistilBERTUnavailableError as exc:
        logger.error(
            "DISTILBERT_INFERENCE_ERROR: model_id=%s",
            payload.model_id,
        )
        raise MLSentryAPIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="DISTILBERT_UNAVAILABLE",
            message="DistilBERT log anomaly inference failed.",
        ) from exc

    # 5. Persist classification record & create alert if anomalous + confidence >= 0.85
    now = datetime.now(timezone.utc)
    alert_created = (
        result.label == LogLabel.ANOMALOUS
        and result.confidence >= LOG_ANOMALY_CONFIDENCE_THRESHOLD
    )

    if alert_created:
        alert = AlertRecord(
            alert_id=uuid.uuid4(),
            model_id=payload.model_id,
            type=AlertType.LOG_ANOMALY,
            severity=AlertSeverity.WARNING,
            feature_name=None,
            message=f"Log anomaly detected with confidence {result.confidence:.4f}",
            context_json={
                "log_line_hash": result.log_line_hash,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
            },
            triggered_at=now,
            cooldown_until=None,  # No cooldown for log anomalies
        )
        session.add(alert)

    log_rec = LogClassificationRecord(
        log_id=uuid.uuid4(),
        model_id=payload.model_id,
        log_line=payload.log_line.strip()[:2000],
        label=result.label,
        confidence=result.confidence,
        model_checkpoint=result.model_checkpoint,
        alert_created=alert_created,
        classified_at=now,
    )
    session.add(log_rec)

    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("LOG_CLASSIFY_DB_FAILED: %s", exc)
        raise MLSentryAPIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_ERROR",
            message="Failed to persist log classification record.",
        ) from exc

    return {
        "label": result.label.value,
        "confidence": result.confidence,
    }
