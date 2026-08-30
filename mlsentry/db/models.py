"""SQLAlchemy ORM models for all 10 MLSentry PostgreSQL tables.

Entity hierarchy (9 FK relationships):
    models (root)
    ├── model_schemas       (model_id FK, CASCADE)
    ├── predictions         (model_id FK, RESTRICT)
    │   └── ground_truth    (pred_id FK, RESTRICT; model_id denormalized)
    ├── reference_stats     (model_id FK, CASCADE)
    ├── drift_reports       (model_id FK, CASCADE)
    ├── performance_logs    (model_id FK, CASCADE)
    ├── alerts              (model_id FK, RESTRICT)
    ├── trigger_events      (model_id FK, RESTRICT)
    └── log_classifications (model_id FK, CASCADE)

FK summary:
    CASCADE off models.model_id (5): model_schemas, reference_stats,
        drift_reports, performance_logs, log_classifications
    RESTRICT off models.model_id (3): predictions, alerts, trigger_events
    RESTRICT on predictions.pred_id (1): ground_truth
"""
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    desc,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# JSON type variant: PostgreSQL uses binary JSONB, SQLite (and other test backends) uses standard JSON
JSON_TYPE = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


def enum_col(enum_cls: Any, name: str) -> SAEnum:
    """Helper for standardizing SAEnum with value-based string serialization."""
    return SAEnum(
        enum_cls,
        values_callable=lambda x: [e.value for e in x],
        name=name,
        create_type=True,
    )

from mlsentry.core.constants import (
    AlertSeverity,
    AlertType,
    DriftMethod,
    FeatureDtype,
    FeatureKind,
    LogLabel,
    ModelStatus,
    PerformanceMetric,
    PredictionType,
    StatType,
    TriggerStatus,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


# ─── Entity 1: models ─────────────────────────────────────────────


class ModelRecord(Base):
    """Root entity. Every other table keys off model_id."""

    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_models_name_version"),
        CheckConstraint(
            "baseline_f1 IS NULL OR (baseline_f1 >= 0 AND baseline_f1 <= 1)",
            name="ck_models_baseline_f1_range",
        ),
        CheckConstraint(
            "baseline_auc IS NULL OR (baseline_auc >= 0 AND baseline_auc <= 1)",
            name="ck_models_baseline_auc_range",
        ),
        CheckConstraint(
            "sample_count >= 0",
            name="ck_models_sample_count_non_negative",
        ),
        CheckConstraint(
            "deprecated_at IS NULL OR deprecated_at >= registered_at",
            name="ck_models_deprecated_after_registered",
        ),
        CheckConstraint(
            "status != 'deprecated' OR deprecated_at IS NOT NULL",
            name="ck_models_deprecated_has_timestamp",
        ),
    )

    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[ModelStatus] = mapped_column(
        enum_col(ModelStatus, "model_status_enum"),
        nullable=False,
        server_default="warming_up",
    )
    baseline_f1: Mapped[float | None] = mapped_column(
        Numeric(5, 4), nullable=True,
    )
    baseline_auc: Mapped[float | None] = mapped_column(
        Numeric(5, 4), nullable=True,
    )
    description: Mapped[str | None] = mapped_column(
        String(500), nullable=True,
    )
    prediction_type: Mapped[PredictionType] = mapped_column(
        enum_col(PredictionType, "prediction_type_enum"),
        nullable=False,
    )
    sample_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"),
    )
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    deprecated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships — CASCADE children
    schemas: Mapped[list["ModelSchemaRecord"]] = relationship(
        back_populates="model", cascade="all, delete-orphan",
    )
    reference_stats: Mapped[list["ReferenceStatRecord"]] = relationship(
        back_populates="model", cascade="all, delete-orphan",
    )
    drift_reports: Mapped[list["DriftReportRecord"]] = relationship(
        back_populates="model", cascade="all, delete-orphan",
    )
    performance_logs: Mapped[list["PerformanceLogRecord"]] = relationship(
        back_populates="model", cascade="all, delete-orphan",
    )
    log_classifications: Mapped[list["LogClassificationRecord"]] = (
        relationship(
            back_populates="model", cascade="all, delete-orphan",
        )
    )
    # Relationships — RESTRICT children (no cascade)
    predictions: Mapped[list["PredictionRecord"]] = relationship(
        back_populates="model",
    )
    alerts: Mapped[list["AlertRecord"]] = relationship(
        back_populates="model",
    )
    trigger_events: Mapped[list["TriggerEventRecord"]] = relationship(
        back_populates="model",
    )


# ─── Entity 2: model_schemas ──────────────────────────────────────


class ModelSchemaRecord(Base):
    """Write-once feature schema rules. Immutable after registration."""

    __tablename__ = "model_schemas"
    __table_args__ = (
        UniqueConstraint(
            "model_id", "feature_name",
            name="uq_model_schemas_model_feature",
        ),
        CheckConstraint(
            "max_length IS NULL OR (max_length >= 1 AND max_length <= 1000)",
            name="ck_model_schemas_max_length_range",
        ),
        CheckConstraint(
            "min_value IS NULL OR max_value IS NULL OR min_value < max_value",
            name="ck_model_schemas_min_lt_max",
        ),
        CheckConstraint(
            "(dtype IN ('float', 'int') AND allowed_values IS NULL AND max_length IS NULL) OR "
            "(dtype = 'category' AND max_length IS NULL) OR "
            "(dtype = 'string') OR "
            "(dtype = 'bool' AND max_length IS NULL AND min_value IS NULL AND max_value IS NULL)",
            name="ck_model_schemas_dtype_constraints",
        ),
    )

    schema_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    dtype: Mapped[FeatureDtype] = mapped_column(
        enum_col(FeatureDtype, "feature_dtype_enum"),
        nullable=False,
    )
    required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true"),
    )
    min_value: Mapped[float | None] = mapped_column(
        Numeric(15, 6), nullable=True,
    )
    max_value: Mapped[float | None] = mapped_column(
        Numeric(15, 6), nullable=True,
    )
    max_length: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True,
    )
    allowed_values: Mapped[list[str] | None] = mapped_column(
        JSON_TYPE, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    model: Mapped["ModelRecord"] = relationship(back_populates="schemas")


# ─── Entity 3: predictions ────────────────────────────────────────


class PredictionRecord(Base):
    """Core ingestion table. High write volume."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index(
            "idx_predictions_model_window",
            "model_id", desc("window_bucket"),
        ),
        Index(
            "idx_predictions_model_logged",
            "model_id", desc("logged_at"),
        ),
        Index(
            "idx_predictions_schema_valid",
            "model_id", "schema_valid",
            postgresql_where=text("schema_valid = false"),
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_predictions_confidence_range",
        ),
        CheckConstraint(
            "length(prediction_label) >= 1 AND length(prediction_label) <= 50",
            name="ck_predictions_label_length",
        ),
    )

    pred_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="RESTRICT"),
        nullable=False,
    )
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    features_json: Mapped[dict] = mapped_column(JSON_TYPE, nullable=False)
    prediction_label: Mapped[str] = mapped_column(
        String(50), nullable=False,
    )
    confidence: Mapped[float | None] = mapped_column(
        Numeric(5, 4), nullable=True,
    )
    schema_valid: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true"),
    )
    window_bucket: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    model: Mapped["ModelRecord"] = relationship(
        back_populates="predictions",
    )
    ground_truth: Mapped["GroundTruthRecord | None"] = relationship(
        back_populates="prediction", uselist=False,
    )


# ─── Entity 4: ground_truth ───────────────────────────────────────


class GroundTruthRecord(Base):
    """Matches a label to a prediction. May arrive up to 72h after prediction."""

    __tablename__ = "ground_truth"
    __table_args__ = (
        Index(
            "idx_gt_model_received",
            "model_id", desc("received_at"),
        ),
        CheckConstraint(
            "latency_hours >= 0",
            name="ck_ground_truth_latency_non_negative",
        ),
        CheckConstraint(
            "length(true_label) >= 1 AND length(true_label) <= 50",
            name="ck_ground_truth_label_length",
        ),
    )

    gt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    pred_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.pred_id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    # Denormalized model reference for query speed.
    # App-layer consistency only — NOT a DB-level FK.
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False,
    )
    true_label: Mapped[str] = mapped_column(String(50), nullable=False)
    latency_hours: Mapped[float] = mapped_column(
        Numeric(8, 4), nullable=False,
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    prediction: Mapped["PredictionRecord"] = relationship(
        back_populates="ground_truth",
    )


# ─── Entity 5: reference_stats ────────────────────────────────────


class ReferenceStatRecord(Base):
    """Baseline feature statistics frozen at model registration."""

    __tablename__ = "reference_stats"
    __table_args__ = (
        UniqueConstraint(
            "model_id", "feature_name", "stat_type",
            name="uq_reference_stats_model_feature_stat",
        ),
        CheckConstraint(
            "sample_count >= 30",
            name="ck_reference_stats_sample_count_min",
        ),
        CheckConstraint(
            "(stat_type = 'frequency_map' "
            "AND frequency_map IS NOT NULL AND stat_value IS NULL AND histogram_data IS NULL) OR "
            "(stat_type IN ('histogram_bin_edges', 'histogram_counts') "
            "AND histogram_data IS NOT NULL AND stat_value IS NULL AND frequency_map IS NULL) OR "
            "(stat_type NOT IN ('frequency_map', 'histogram_bin_edges', 'histogram_counts') "
            "AND stat_value IS NOT NULL AND frequency_map IS NULL AND histogram_data IS NULL)",
            name="ck_reference_stats_stat_type_exclusivity",
        ),
    )

    stat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    stat_type: Mapped[StatType] = mapped_column(
        enum_col(StatType, "stat_type_enum"),
        nullable=False,
    )
    stat_value: Mapped[float | None] = mapped_column(
        Numeric(15, 6), nullable=True,
    )
    frequency_map: Mapped[dict | None] = mapped_column(
        JSON_TYPE, nullable=True,
    )
    histogram_data: Mapped[list[Any] | None] = mapped_column(
        JSON_TYPE, nullable=True,
    )
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    model: Mapped["ModelRecord"] = relationship(
        back_populates="reference_stats",
    )


# ─── Entity 6: drift_reports ──────────────────────────────────────


class DriftReportRecord(Base):
    """One row per (model, feature, method, window).

    NOTE: severity column is DROPPED per GI-05 / F-005 Conflict #3.
    Severity is computed dynamically in route handlers from score/psi
    values and config thresholds in core/constants.py.
    """

    __tablename__ = "drift_reports"
    __table_args__ = (
        Index(
            "idx_drift_model_computed",
            "model_id", desc("computed_at"),
        ),
        CheckConstraint(
            "window_start < window_end",
            name="ck_drift_reports_window_order",
        ),
        CheckConstraint(
            "score >= 0",
            name="ck_drift_reports_score_non_negative",
        ),
        CheckConstraint(
            "p_value IS NULL OR (p_value >= 0 AND p_value <= 1)",
            name="ck_drift_reports_p_value_range",
        ),
        CheckConstraint(
            "sample_count >= 30",
            name="ck_drift_reports_sample_count_min",
        ),
        CheckConstraint(
            "(method = 'chi_square' AND p_value IS NOT NULL) OR "
            "(method = 'psi' AND p_value IS NULL)",
            name="ck_drift_reports_method_p_value",
        ),
    )

    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    method: Mapped[DriftMethod] = mapped_column(
        enum_col(DriftMethod, "drift_method_enum"),
        nullable=False,
    )
    score: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    p_value: Mapped[float | None] = mapped_column(
        Numeric(10, 8), nullable=True,
    )
    feature_type: Mapped[FeatureKind] = mapped_column(
        enum_col(FeatureKind, "feature_type_enum"),
        nullable=False,
    )
    psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    model: Mapped["ModelRecord"] = relationship(
        back_populates="drift_reports",
    )


# ─── Entity 7: performance_logs ───────────────────────────────────


class PerformanceLogRecord(Base):
    """Computed when >= 50 matched prediction-label pairs exist in window."""

    __tablename__ = "performance_logs"
    __table_args__ = (
        CheckConstraint(
            "window_start < window_end",
            name="ck_perf_logs_window_order",
        ),
        CheckConstraint(
            "value >= 0 AND value <= 1",
            name="ck_perf_logs_value_range",
        ),
        CheckConstraint(
            "baseline_value >= 0 AND baseline_value <= 1",
            name="ck_perf_logs_baseline_range",
        ),
        CheckConstraint(
            "delta >= -1 AND delta <= 1",
            name="ck_perf_logs_delta_range",
        ),
        CheckConstraint(
            "sample_count >= 50",
            name="ck_perf_logs_sample_count_min",
        ),
    )

    perf_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="CASCADE"),
        nullable=False,
    )
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    metric: Mapped[PerformanceMetric] = mapped_column(
        enum_col(PerformanceMetric, "performance_metric_enum"),
        nullable=False,
    )
    value: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    baseline_value: Mapped[float] = mapped_column(
        Numeric(5, 4), nullable=False,
    )
    delta: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    model: Mapped["ModelRecord"] = relationship(
        back_populates="performance_logs",
    )


# ─── Entity 8: alerts ─────────────────────────────────────────────


class AlertRecord(Base):
    """Central alert store. All alert types land here."""

    __tablename__ = "alerts"
    __table_args__ = (
        Index(
            "idx_alerts_model_unresolved",
            "model_id", desc("severity"), desc("triggered_at"),
            postgresql_where=text("resolved = false"),
        ),
        Index(
            "idx_alerts_cooldown",
            "model_id", "type", "feature_name", "cooldown_until",
            postgresql_where=text("resolved = false"),
        ),
        CheckConstraint(
            "(resolved = false AND resolved_at IS NULL) OR "
            "(resolved = true AND resolved_at IS NOT NULL)",
            name="ck_alerts_resolved_consistency",
        ),
        CheckConstraint(
            "resolved_at IS NULL OR resolved_at >= triggered_at",
            name="ck_alerts_resolved_after_triggered",
        ),
        CheckConstraint(
            "cooldown_until IS NULL OR cooldown_until > triggered_at",
            name="ck_alerts_cooldown_after_triggered",
        ),
    )

    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="RESTRICT"),
        nullable=False,
    )
    type: Mapped[AlertType] = mapped_column(
        enum_col(AlertType, "alert_type_enum"),
        nullable=False,
    )
    severity: Mapped[AlertSeverity] = mapped_column(
        enum_col(AlertSeverity, "alert_severity_enum"),
        nullable=False,
    )
    feature_name: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
    )
    message: Mapped[str] = mapped_column(String(1000), nullable=False)
    context_json: Mapped[dict | None] = mapped_column(
        JSON_TYPE, nullable=True,
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    resolved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"),
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    cooldown_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    model: Mapped["ModelRecord"] = relationship(back_populates="alerts")


# ─── Entity 9: trigger_events ─────────────────────────────────────


class TriggerEventRecord(Base):
    """Audit log for every retraining dispatch attempt. Immutable after write."""

    __tablename__ = "trigger_events"
    __table_args__ = (
        Index(
            "idx_trigger_model_cooldown",
            "model_id", desc("next_allowed_at"),
        ),
        CheckConstraint(
            "next_allowed_at > triggered_at",
            name="ck_trigger_next_after_triggered",
        ),
        CheckConstraint(
            "(status = 'success' AND github_response_code = 204 "
            "AND error_message IS NULL AND error_type IS NULL) OR "
            "(status = 'failed' AND github_response_code IS NOT NULL "
            "AND error_message IS NOT NULL AND error_type IS NULL) OR "
            "(status = 'failed' AND github_response_code IS NULL "
            "AND error_message IS NOT NULL "
            "AND error_type IN ('NETWORK_TIMEOUT', 'DNS_FAILURE', 'CONNECTION_REFUSED')) OR "
            "(status = 'suppressed' AND github_response_code IS NULL "
            "AND error_type IS NULL)",
            name="ck_trigger_status_consistency",
        ),
    )

    trigger_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[TriggerStatus] = mapped_column(
        enum_col(TriggerStatus, "trigger_status_enum"),
        nullable=False,
    )
    drift_report_summary: Mapped[dict] = mapped_column(
        JSON_TYPE, nullable=False,
    )
    github_response_code: Mapped[int | None] = mapped_column(
        SmallInteger, nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        String(500), nullable=True,
    )
    error_type: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    next_allowed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    model: Mapped["ModelRecord"] = relationship(
        back_populates="trigger_events",
    )


# ─── Entity 10: log_classifications ───────────────────────────────


class LogClassificationRecord(Base):
    """Results from DistilBERT log anomaly detection."""

    __tablename__ = "log_classifications"
    __table_args__ = (
        Index(
            "idx_logclass_model_anomalous",
            "model_id", desc("classified_at"),
            postgresql_where=text("label = 'anomalous'"),
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_log_classifications_confidence_range",
        ),
        CheckConstraint(
            "alert_created = false OR label = 'anomalous'",
            name="ck_log_classifications_alert_only_anomalous",
        ),
        CheckConstraint(
            "length(log_line) >= 10 AND length(log_line) <= 2000",
            name="ck_log_classifications_log_line_length",
        ),
    )

    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.model_id", ondelete="CASCADE"),
        nullable=False,
    )
    log_line: Mapped[str] = mapped_column(String(2000), nullable=False)
    label: Mapped[LogLabel] = mapped_column(
        enum_col(LogLabel, "log_label_enum"),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(
        Numeric(5, 4), nullable=False,
    )
    model_checkpoint: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )
    alert_created: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"),
    )
    classified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )

    model: Mapped["ModelRecord"] = relationship(
        back_populates="log_classifications",
    )
