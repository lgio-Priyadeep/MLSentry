"""API route handlers and request/response validation schemas for model registration.

Implements POST /v1/models/register:
  - Pydantic validation of model metadata, prediction types, and feature schemas.
  - Baseline reference statistics validation (histogram bins for numerical, frequency map for categorical).
  - Uniqueness enforcement for (name, version) pairs.
  - In-process MLflow integration with circuit breaker fallback (503 MLFLOW_REGISTRATION_FAILED).
  - Atomic transaction writing to models, model_schemas, and reference_stats tables.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from mlsentry.api.errors import MLSentryAPIException, get_request_id
from mlsentry.core.constants import (
    FeatureDtype,
    ModelStatus,
    PredictionType,
    StatType,
    WARM_UP_THRESHOLD,
)
from mlsentry.db.models import ModelRecord, ModelSchemaRecord, ReferenceStatRecord
from mlsentry.db.session import get_session
from mlsentry.integrations.mlflow_client import (
    MLflowCircuitBreakerOpenError,
    MLflowClient,
    MLflowUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

NAME_REGEX = re.compile(r"^[a-z0-9_-]+$")
VERSION_REGEX = re.compile(r"^\d+\.\d+(\.\d+)?$")
FEATURE_NAME_REGEX = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


class FeatureSchemaInput(BaseModel):
    """Pydantic schema definition for a single feature during registration."""

    feature_name: str
    dtype: FeatureDtype
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    max_length: int | None = None
    baseline_stats: dict[str, Any] = Field(default_factory=dict)

    @field_validator("feature_name")
    @classmethod
    def validate_feature_name(cls, v: str) -> str:
        if not FEATURE_NAME_REGEX.match(v):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message=f"feature_name '{v}' does not match pattern ^[a-zA-Z][a-zA-Z0-9_]*$",
            )
        return v

    @model_validator(mode="after")
    def validate_feature_constraints(self) -> "FeatureSchemaInput":
        # Numerical constraints check
        if self.dtype in (FeatureDtype.BOOL, FeatureDtype.STRING, FeatureDtype.CATEGORY):
            if self.min_value is not None or self.max_value is not None:
                raise MLSentryAPIException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    error_code="VALIDATION_ERROR",
                    message=(
                        f"min_value and max_value are only valid for 'float' and 'int' dtypes, "
                        f"but received for feature '{self.feature_name}' with dtype '{self.dtype.value}'."
                    ),
                )

        if self.min_value is not None and self.max_value is not None:
            if self.min_value >= self.max_value:
                raise MLSentryAPIException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    error_code="VALIDATION_ERROR",
                    message=(
                        f"min_value ({self.min_value}) must be strictly less than "
                        f"max_value ({self.max_value}) for feature '{self.feature_name}'."
                    ),
                )

        # max_length validation (string only)
        if self.dtype != FeatureDtype.STRING and self.max_length is not None:
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message=(
                    f"max_length is valid only for 'string' dtype, "
                    f"but received for feature '{self.feature_name}' with dtype '{self.dtype.value}'."
                ),
            )

        if self.dtype == FeatureDtype.STRING and self.max_length is not None:
            if not (1 <= self.max_length <= 1000):
                raise MLSentryAPIException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    error_code="VALIDATION_ERROR",
                    message=f"max_length for string feature '{self.feature_name}' must be between 1 and 1000.",
                )

        # Categorical baseline_stats frequency_map validation
        if self.dtype == FeatureDtype.CATEGORY:
            freq_map = self.baseline_stats.get("frequency_map")
            if freq_map is None or not isinstance(freq_map, dict) or len(freq_map) == 0:
                raise MLSentryAPIException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    error_code="FREQUENCY_MAP_INVALID",
                    message=f"Categorical feature '{self.feature_name}' requires a non-empty 'frequency_map' in baseline_stats.",
                )
            total_prop = sum(float(v) for v in freq_map.values())
            if abs(total_prop - 1.0) > 0.001:
                raise MLSentryAPIException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    error_code="FREQUENCY_MAP_INVALID",
                    message=(
                        f"Categorical baseline frequency_map for feature '{self.feature_name}' "
                        f"must sum to 1.0 ±0.001 (actual sum: {total_prop:.6f})."
                    ),
                )

        # Numerical baseline_stats histogram validation
        if self.dtype in (FeatureDtype.FLOAT, FeatureDtype.INT):
            edges = self.baseline_stats.get("histogram_bin_edges")
            counts = self.baseline_stats.get("histogram_counts")
            if edges is not None or counts is not None:
                if not isinstance(edges, list) or len(edges) < 2:
                    raise MLSentryAPIException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        error_code="INVALID_BASELINE_STATS",
                        message=f"Numerical feature '{self.feature_name}' histogram_bin_edges must be a list of at least 2 numbers.",
                    )
                if not isinstance(counts, list) or len(counts) != len(edges) - 1:
                    raise MLSentryAPIException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        error_code="INVALID_BASELINE_STATS",
                        message=(
                            f"Numerical feature '{self.feature_name}' histogram_counts length "
                            f"({len(counts) if isinstance(counts, list) else 0}) must equal len(histogram_bin_edges) - 1 ({len(edges) - 1})."
                        ),
                    )
                if any(float(edges[i]) >= float(edges[i + 1]) for i in range(len(edges) - 1)):
                    raise MLSentryAPIException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        error_code="INVALID_BASELINE_STATS",
                        message=f"Numerical feature '{self.feature_name}' histogram_bin_edges must be strictly monotonically increasing.",
                    )
                if any(int(c) < 0 for c in counts):
                    raise MLSentryAPIException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        error_code="INVALID_BASELINE_STATS",
                        message=f"Numerical feature '{self.feature_name}' histogram_counts must all be non-negative integers.",
                    )

        return self


class ModelRegisterRequest(BaseModel):
    """Request payload for POST /v1/models/register."""

    name: str
    version: str
    prediction_type: PredictionType
    features: list[FeatureSchemaInput]
    baseline_f1: float
    baseline_auc: float | None = None
    description: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not NAME_REGEX.match(v):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="INVALID_MODEL_NAME",
                message=f"name '{v}' does not match required pattern ^[a-z0-9_-]+$",
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if not VERSION_REGEX.match(v):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="INVALID_MODEL_VERSION",
                message=f"version '{v}' does not match required pattern ^\\d+\\.\\d+(\\.\\d+)?$",
            )
        return v

    @field_validator("baseline_f1")
    @classmethod
    def validate_baseline_f1(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message="baseline_f1 must be within range [0.0, 1.0].",
            )
        return v

    @field_validator("baseline_auc")
    @classmethod
    def validate_baseline_auc(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message="baseline_auc must be within range [0.0, 1.0].",
            )
        return v

    @model_validator(mode="after")
    def validate_features_list(self) -> "ModelRegisterRequest":
        if not self.features or len(self.features) == 0:
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message="features list must contain at least one feature definition.",
            )

        names = [f.feature_name for f in self.features]
        if len(names) != len(set(names)):
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message="features list contains duplicate feature_name entries.",
            )

        return self


class ModelRegisterResponse(BaseModel):
    """Response payload for successful POST /v1/models/register."""

    model_id: uuid.UUID
    status: str
    registered_at: str
    sample_count: int
    warm_up_threshold: int


@router.post(
    "/models/register",
    status_code=status.HTTP_201_CREATED,
    response_model=ModelRegisterResponse,
)
async def register_model(
    payload: ModelRegisterRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> Any:
    """Register a new model version and freeze reference baseline statistics."""
    # 1. Uniqueness check for (name, version)
    existing = (
        session.query(ModelRecord)
        .filter(
            ModelRecord.name == payload.name,
            ModelRecord.version == payload.version,
        )
        .first()
    )
    if existing is not None:
        raise MLSentryAPIException(
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
            message=(
                f"Model '{payload.name}' version '{payload.version}' is already registered. "
                f"Use a new version string or deprecate the existing registration."
            ),
        )

    # 2. In-process MLflow registry validation & circuit breaker check (per F-01 & Test Oracle Feature 1)
    req_id = get_request_id(request)
    settings = getattr(request.app.state, "settings", None)
    mlflow_client = getattr(request.app.state, "mlflow_client", None)
    if mlflow_client is None:
        mlflow_client = MLflowClient.from_settings(settings) if settings else MLflowClient()

    try:
        mlflow_client.fetch_model_metadata(
            name=payload.name,
            version=payload.version,
            request_id=req_id,
        )
    except (MLflowUnavailableError, MLflowCircuitBreakerOpenError) as exc:
        logger.error(
            "MLFLOW_ERROR: model_name=%s, version=%s, request_id=%s, error_type=%s",
            payload.name,
            payload.version,
            req_id,
            type(exc).__name__,
        )
        raise MLSentryAPIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MLFLOW_REGISTRATION_FAILED",
            message="Model registration failed — MLflow unreachable. Retry when MLflow is available.",
        ) from exc

    # 3. Build and persist records atomically
    model_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    description = (
        payload.description.strip()[:500] if payload.description else None
    )
    model = ModelRecord(
        model_id=model_id,
        name=payload.name,
        version=payload.version,
        status=ModelStatus.WARMING_UP,
        baseline_f1=payload.baseline_f1,
        baseline_auc=payload.baseline_auc,
        description=description,
        prediction_type=payload.prediction_type,
        sample_count=0,
        registered_at=now,
    )
    session.add(model)

    for feat in payload.features:
        allowed_vals = None
        if feat.dtype == FeatureDtype.CATEGORY and "frequency_map" in feat.baseline_stats:
            allowed_vals = list(feat.baseline_stats["frequency_map"].keys())

        schema_rec = ModelSchemaRecord(
            schema_id=uuid.uuid4(),
            model_id=model_id,
            feature_name=feat.feature_name,
            dtype=feat.dtype,
            required=feat.required,
            min_value=feat.min_value,
            max_value=feat.max_value,
            max_length=feat.max_length,
            allowed_values=allowed_vals,
            created_at=now,
        )
        session.add(schema_rec)

        # Populate baseline reference statistics
        stats = feat.baseline_stats
        sample_cnt = max(30, int(stats.get("sample_count", 500)))

        if feat.dtype in (FeatureDtype.FLOAT, FeatureDtype.INT):
            # Numerical scalar metrics
            for st_name in ("mean", "std", "min", "max", "p25", "p50", "p75", "p95"):
                if st_name in stats:
                    val = float(stats[st_name])
                    session.add(
                        ReferenceStatRecord(
                            stat_id=uuid.uuid4(),
                            model_id=model_id,
                            feature_name=feat.feature_name,
                            stat_type=StatType(st_name),
                            stat_value=val,
                            frequency_map=None,
                            histogram_data=None,
                            sample_count=sample_cnt,
                            computed_at=now,
                        )
                    )
            # Numerical histogram bins and counts
            if "histogram_bin_edges" in stats:
                session.add(
                    ReferenceStatRecord(
                        stat_id=uuid.uuid4(),
                        model_id=model_id,
                        feature_name=feat.feature_name,
                        stat_type=StatType.HISTOGRAM_BIN_EDGES,
                        stat_value=None,
                        frequency_map=None,
                        histogram_data=stats["histogram_bin_edges"],
                        sample_count=sample_cnt,
                        computed_at=now,
                    )
                )
            if "histogram_counts" in stats:
                session.add(
                    ReferenceStatRecord(
                        stat_id=uuid.uuid4(),
                        model_id=model_id,
                        feature_name=feat.feature_name,
                        stat_type=StatType.HISTOGRAM_COUNTS,
                        stat_value=None,
                        frequency_map=None,
                        histogram_data=stats["histogram_counts"],
                        sample_count=sample_cnt,
                        computed_at=now,
                    )
                )
        elif feat.dtype == FeatureDtype.CATEGORY:
            if "frequency_map" in stats:
                session.add(
                    ReferenceStatRecord(
                        stat_id=uuid.uuid4(),
                        model_id=model_id,
                        feature_name=feat.feature_name,
                        stat_type=StatType.FREQUENCY_MAP,
                        stat_value=None,
                        frequency_map=stats["frequency_map"],
                        histogram_data=None,
                        sample_count=sample_cnt,
                        computed_at=now,
                    )
                )

    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("REGISTRATION_DB_FAILED: %s", exc)
        raise MLSentryAPIException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="REGISTRATION_FAILED",
            message="Model registration transaction failed.",
        ) from exc

    return {
        "model_id": model_id,
        "status": "warming_up",
        "registered_at": now.isoformat(),
        "sample_count": 0,
        "warm_up_threshold": WARM_UP_THRESHOLD,
    }
