"""Named constants and domain enums for MLSentry.

All magic numbers are defined here. No hardcoded values elsewhere.
Constants are grouped by domain concern.
"""
import enum
from datetime import timedelta


# ─── Model Lifecycle ───────────────────────────────────────────────
WARM_UP_THRESHOLD: int = 30

# ─── Monitoring Cadence ────────────────────────────────────────────
MONITORING_INTERVAL_MINUTES: int = 15

# ─── Ground Truth ──────────────────────────────────────────────────
GROUND_TRUTH_CUTOFF_HOURS: int = 72

# ─── Features JSON Limits ─────────────────────────────────────────
FEATURES_JSON_MAX_KEYS: int = 200
FEATURES_JSON_MAX_BYTES: int = 65_536  # 64 KB

# ─── Frequency Map Tolerance ──────────────────────────────────────
FREQUENCY_MAP_TOLERANCE: float = 0.001  # ±0.001 from 1.0

# ─── Statistical Drift Thresholds (PSI — Numerical) ──────────────
PSI_THRESHOLD_WARNING: float = 0.1
PSI_THRESHOLD_CRITICAL: float = 0.25

# ─── Statistical Drift Thresholds (Chi-square — Categorical) ─────
CHI2_P_VALUE_WARNING: float = 0.05
CHI2_P_VALUE_CRITICAL: float = 0.01

# ─── Performance Degradation Thresholds ───────────────────────────
PERF_DELTA_WARNING: float = -0.05
PERF_DELTA_CRITICAL: float = -0.15

# ─── Performance Tracking ─────────────────────────────────────────
MIN_MATCHED_PAIRS: int = 50

# ─── Log Anomaly Detection ────────────────────────────────────────
LOG_ANOMALY_CONFIDENCE_THRESHOLD: float = 0.85
DISTILBERT_SLA_MS: int = 500

# ─── DistilBERT Circuit Breaker ───────────────────────────────────
DISTILBERT_CB_FAILURE_THRESHOLD: int = 5
DISTILBERT_CB_RESET_SECONDS: int = 60

# ─── MLflow Circuit Breaker ───────────────────────────────────────
MLFLOW_CB_FAILURE_THRESHOLD: int = 3
MLFLOW_CB_WINDOW_SECONDS: int = 60
MLFLOW_CB_RESET_SECONDS: int = 30

# ─── MLflow Client ────────────────────────────────────────────────
# Fixed connect timeout per ND-01 (read timeout is configurable via
# MLFLOW_TIMEOUT_MS env var in settings.py).
MLFLOW_CONNECT_TIMEOUT_SECONDS: int = 3

# ─── GitHub Actions ───────────────────────────────────────────────
# Fixed connect timeout per ND-03 (read timeout is configurable via
# GITHUB_API_TIMEOUT_MS env var in settings.py).
GITHUB_CONNECT_TIMEOUT_SECONDS: int = 5
GITHUB_CONSECUTIVE_FAILURE_ALERT: int = 3
GITHUB_DISPATCH_SUSPENDED_HOURS: int = 24

# ─── Retraining Dispatch Cooldown ─────────────────────────────────
RETRAINING_COOLDOWN_HOURS: int = 6

# ─── Alert Deduplication Cooldowns (minutes) ──────────────────────
ALERT_COOLDOWN_DRIFT_MINUTES: int = 30
ALERT_COOLDOWN_PERF_DEG_MINUTES: int = 60
ALERT_COOLDOWN_SCHEMA_VIOLATION_MINUTES: int = 5
ALERT_COOLDOWN_SCHEMA_WARNING_MINUTES: int = 15
ALERT_COOLDOWN_LOG_ANOMALY_MINUTES: int = 0
ALERT_COOLDOWN_TRIGGER_FAILURE_MINUTES: int = 0
ALERT_COOLDOWN_MONITORING_ENGINE_FAILURE_MINUTES: int = 0

# ─── Database Connection Pool ─────────────────────────────────────
DB_POOL_SIZE: int = 10
DB_MAX_OVERFLOW: int = 5
DB_POOL_TIMEOUT: int = 3
DB_POOL_RECYCLE: int = 1800

# ─── Thread Pool ──────────────────────────────────────────────────
THREAD_POOL_MAX_WORKERS: int = 4

# ─── Pagination ───────────────────────────────────────────────────
PAGINATION_DEFAULT_LIMIT: int = 50
PAGINATION_MAX_LIMIT: int = 200

# ─── Graceful Shutdown ────────────────────────────────────────────
SHUTDOWN_DRAIN_TIMEOUT_SECONDS: int = 10

# ─── Health Check ─────────────────────────────────────────────────
SCHEDULER_HEARTBEAT_STALENESS_MINUTES: int = 30

# ─── Log Line Limits ─────────────────────────────────────────────
LOG_LINE_MIN_LENGTH: int = 10
LOG_LINE_MAX_LENGTH: int = 2000

# ─── Trigger Events Sentinel ─────────────────────────────────────
TRIGGER_FAILURE_SENTINEL_SECONDS: int = 1

# ─── Webhook SSRF ────────────────────────────────────────────────
WEBHOOK_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


# ═══ Domain Enums ═════════════════════════════════════════════════


class ModelStatus(str, enum.Enum):
    """Model lifecycle status."""

    WARMING_UP = "warming_up"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class PredictionType(str, enum.Enum):
    """Model prediction output type."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class FeatureDtype(str, enum.Enum):
    """Registered feature data type."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    CATEGORY = "category"


class StatType(str, enum.Enum):
    """Reference statistics type."""

    MEAN = "mean"
    STD = "std"
    MIN = "min"
    MAX = "max"
    P25 = "p25"
    P50 = "p50"
    P75 = "p75"
    P95 = "p95"
    FREQUENCY_MAP = "frequency_map"
    HISTOGRAM_BIN_EDGES = "histogram_bin_edges"
    HISTOGRAM_COUNTS = "histogram_counts"


class DriftMethod(str, enum.Enum):
    """Statistical drift detection method."""

    PSI = "psi"
    CHI_SQUARE = "chi_square"


class FeatureKind(str, enum.Enum):
    """Feature data category for drift computation."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class PerformanceMetric(str, enum.Enum):
    """Performance evaluation metric."""

    F1 = "f1"
    AUC = "auc"


class AlertType(str, enum.Enum):
    """Alert classification type."""

    DRIFT = "DRIFT"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    SCHEMA_WARNING = "SCHEMA_WARNING"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    LOG_ANOMALY = "LOG_ANOMALY"
    TRIGGER_FAILURE = "TRIGGER_FAILURE"
    MONITORING_ENGINE_FAILURE = "MONITORING_ENGINE_FAILURE"


class AlertSeverity(str, enum.Enum):
    """Alert severity level.

    Stored as lowercase in DB; serialized as uppercase in API DTOs.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class LogLabel(str, enum.Enum):
    """DistilBERT log classification label."""

    ANOMALOUS = "anomalous"
    NORMAL = "normal"


class TriggerStatus(str, enum.Enum):
    """GitHub Actions dispatch status."""

    SUCCESS = "success"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


# ─── Alert Cooldown Mapping ──────────────────────────────────────

ALERT_COOLDOWN_MAP: dict[AlertType, timedelta] = {
    AlertType.DRIFT: timedelta(minutes=ALERT_COOLDOWN_DRIFT_MINUTES),
    AlertType.SCHEMA_VIOLATION: timedelta(
        minutes=ALERT_COOLDOWN_SCHEMA_VIOLATION_MINUTES,
    ),
    AlertType.SCHEMA_WARNING: timedelta(
        minutes=ALERT_COOLDOWN_SCHEMA_WARNING_MINUTES,
    ),
    AlertType.PERFORMANCE_DEGRADATION: timedelta(
        minutes=ALERT_COOLDOWN_PERF_DEG_MINUTES,
    ),
    AlertType.LOG_ANOMALY: timedelta(
        minutes=ALERT_COOLDOWN_LOG_ANOMALY_MINUTES,
    ),
    AlertType.TRIGGER_FAILURE: timedelta(
        minutes=ALERT_COOLDOWN_TRIGGER_FAILURE_MINUTES,
    ),
    AlertType.MONITORING_ENGINE_FAILURE: timedelta(
        minutes=ALERT_COOLDOWN_MONITORING_ENGINE_FAILURE_MINUTES,
    ),
}
