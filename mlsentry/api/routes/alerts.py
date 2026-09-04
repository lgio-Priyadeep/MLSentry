"""API route handlers for alert management, querying, and manual resolution.

Implements:
  - GET /v1/alerts/{model_id}:
      Returns paginated list of alerts for a specific model.
      Supports filtering by status ("active", "resolved") with default "active".
      Supports filtering by severity ("INFO", "WARNING", "CRITICAL").
      Invalid query parameters trigger HTTP 422 SCHEMA_VALIDATION_FAILED.
      Explicitly serializes null values for feature_name and resolved_at.
  - PATCH /v1/alerts/{alert_id}/resolve:
      Manually resolves an active alert by setting resolved=true and resolved_at=now().
      Alert resolution is write-once and immutable: already resolved alerts return HTTP 409 CONFLICT.
      Alert cooldown (cooldown_until) is NOT reset on resolution.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import case, desc
from sqlalchemy.orm import Session

from mlsentry.api.errors import MLSentryAPIException
from mlsentry.core.constants import AlertSeverity, AlertType
from mlsentry.db.models import AlertRecord, ModelRecord
from mlsentry.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["alerts"])


# ─── Pydantic Response DTOs ──────────────────────────────────────


class PaginationMeta(BaseModel):
    """Pagination metadata container."""

    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int


class AlertItem(BaseModel):
    """Single alert record representation."""

    model_config = ConfigDict(extra="forbid")

    alert_id: uuid.UUID
    type: str
    feature_name: str | None = None
    severity: str
    message: str
    triggered_at: str
    resolved_at: str | None = None


class AlertListResponse(BaseModel):
    """Response payload for GET /v1/alerts/{model_id}."""

    model_config = ConfigDict(extra="forbid")

    model_id: uuid.UUID
    data: list[AlertItem]
    meta: PaginationMeta


class AlertResolveResponse(BaseModel):
    """Response payload for PATCH /v1/alerts/{alert_id}/resolve."""

    model_config = ConfigDict(extra="forbid")

    alert_id: uuid.UUID
    resolved: bool
    resolved_at: str


# ─── Route Handlers ──────────────────────────────────────────────


@router.get(
    "/alerts/{model_id}",
    response_model=AlertListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get alerts for a model with status and severity filters",
)
def get_alerts(
    model_id: uuid.UUID,
    status_filter: str = Query(
        "active",
        alias="status",
        description="Filter alerts by status: 'active' or 'resolved'",
    ),
    severity_filter: str | None = Query(
        None,
        alias="severity",
        description="Filter alerts by severity: 'INFO', 'WARNING', 'CRITICAL'",
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_session),
) -> AlertListResponse:
    """Fetch paginated alerts for a specific model with status and severity filtering.

    Validations:
      - 404 NOT_FOUND if model_id does not exist.
      - 422 SCHEMA_VALIDATION_FAILED if status is not 'active' or 'resolved'.
      - 422 SCHEMA_VALIDATION_FAILED if severity is not 'INFO', 'WARNING', or 'CRITICAL'.
    """
    model = db.get(ModelRecord, model_id)
    if model is None:
        raise MLSentryAPIException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=f"model_id '{model_id}' not found.",
        )

    if status_filter not in ("active", "resolved"):
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="SCHEMA_VALIDATION_FAILED",
            message=f"Invalid status filter '{status_filter}'. Accepted values: 'active', 'resolved'.",
        )

    parsed_severity: AlertSeverity | None = None
    if severity_filter is not None:
        sev_upper = severity_filter.strip().upper()
        if sev_upper not in ("INFO", "WARNING", "CRITICAL"):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="SCHEMA_VALIDATION_FAILED",
                message=f"Invalid severity filter '{severity_filter}'. Accepted values: 'INFO', 'WARNING', 'CRITICAL'.",
            )
        parsed_severity = AlertSeverity(sev_upper.lower())

    query = db.query(AlertRecord).filter(AlertRecord.model_id == model_id)

    if status_filter == "active":
        query = query.filter(AlertRecord.resolved.is_(False))
    elif status_filter == "resolved":
        query = query.filter(AlertRecord.resolved.is_(True))

    if parsed_severity is not None:
        query = query.filter(AlertRecord.severity == parsed_severity)

    total_count = query.count()
    severity_priority = case(
        (AlertRecord.severity == AlertSeverity.CRITICAL, 3),
        (AlertRecord.severity == AlertSeverity.WARNING, 2),
        (AlertRecord.severity == AlertSeverity.INFO, 1),
        else_=0,
    )
    alerts = (
        query.order_by(desc(severity_priority), desc(AlertRecord.triggered_at))
        .offset(offset)
        .limit(limit)
        .all()
    )

    data_items: list[AlertItem] = []
    for a in alerts:
        data_items.append(
            AlertItem(
                alert_id=a.alert_id,
                type=a.type.value,
                feature_name=a.feature_name,
                severity=a.severity.value.upper(),
                message=a.message,
                triggered_at=a.triggered_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                resolved_at=a.resolved_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                if a.resolved_at is not None
                else None,
            )
        )

    return AlertListResponse(
        model_id=model_id,
        data=data_items,
        meta=PaginationMeta(total=total_count, limit=limit, offset=offset),
    )


@router.patch(
    "/alerts/{alert_id}/resolve",
    response_model=AlertResolveResponse,
    status_code=status.HTTP_200_OK,
    summary="Manually resolve an active alert",
)
def resolve_alert(
    alert_id: uuid.UUID,
    db: Session = Depends(get_session),
) -> AlertResolveResponse:
    """Manually resolve an active alert.

    Validations:
      - 404 NOT_FOUND if alert_id does not exist.
      - 409 CONFLICT if alert is already resolved.
    """
    alert = db.get(AlertRecord, alert_id)
    if alert is None:
        raise MLSentryAPIException(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=f"alert_id '{alert_id}' not found.",
        )

    if alert.resolved or alert.resolved_at is not None:
        resolved_ts = (
            alert.resolved_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            if alert.resolved_at is not None
            else "already"
        )
        raise MLSentryAPIException(
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
            message=f"Alert '{alert_id}' is already resolved at {resolved_ts}.",
        )

    now_utc = datetime.now(timezone.utc)
    alert.resolved = True
    alert.resolved_at = now_utc

    # Note: cooldown_until is NOT reset per specification
    db.commit()
    db.refresh(alert)

    logger.info(
        "ALERT_RESOLVED: alert_id=%s, model_id=%s, type=%s, resolved_at=%s",
        alert.alert_id,
        alert.model_id,
        alert.type.value,
        now_utc.isoformat(),
    )

    return AlertResolveResponse(
        alert_id=alert.alert_id,
        resolved=True,
        resolved_at=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
