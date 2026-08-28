"""MLflow Model Registry integration client with in-process circuit breaker.

Provides model metadata fetching, baseline reference statistics extraction,
connection timeout enforcement, and circuit breaker resilience.

Architectural rules & constraints:
  - Client SDK: Uses mlflow-skinny / HTTP client against MLflow Tracking Server.
  - Timeouts (ND-01): Connect timeout = 3s (MLFLOW_CONNECT_TIMEOUT_SECONDS),
    read timeout configurable via MLFLOW_TIMEOUT_MS in settings (default 5s).
  - Circuit Breaker (ND-02): 3 consecutive failures within 60s
    (MLFLOW_CB_WINDOW_SECONDS) trips circuit OPEN for 30s
    (MLFLOW_CB_RESET_SECONDS). While OPEN, calls immediately raise
    MLflowCircuitBreakerOpenError without invoking the remote MLflow server.
  - Zero Raw Data: Feature schema details, baseline stats values, and raw
    features must NEVER appear in error log lines or exception traces.
  - Fallback: Fail closed — registration rejects entirely on MLflow outage;
    no partial or degraded registration mode permitted.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from mlsentry.config.settings import Settings
from mlsentry.core.constants import (
    MLFLOW_CB_FAILURE_THRESHOLD,
    MLFLOW_CB_RESET_SECONDS,
    MLFLOW_CB_WINDOW_SECONDS,
    MLFLOW_CONNECT_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class MLflowIntegrationError(Exception):
    """Base exception for MLflow integration failures."""


class MLflowUnavailableError(MLflowIntegrationError):
    """Raised when the MLflow server is unreachable or returns non-2xx."""


class MLflowCircuitBreakerOpenError(MLflowUnavailableError):
    """Raised when the MLflow circuit breaker is in OPEN state."""


@dataclass(frozen=True)
class MLflowModelMetadata:
    """Metadata extracted from MLflow Model Registry.

    Attributes:
        name: Registered model name.
        version: Registered model version.
        run_id: MLflow run ID associated with the model version.
        description: Model description from registry (if present).
        tags: Dictionary of string tags associated with the model version.
        baseline_stats: Extracted baseline statistics dictionary, or None.
    """

    name: str
    version: str
    run_id: str
    description: str | None
    tags: dict[str, str]
    baseline_stats: dict[str, Any] | None


class MLflowCircuitBreaker:
    """Thread-safe in-process circuit breaker for MLflow service interactions.

    State transitions:
      - CLOSED: Normal operation. Failures are recorded with timestamps.
        If >= 3 failures occur within 60s, state transitions to OPEN.
      - OPEN: Rejects all calls immediately with MLflowCircuitBreakerOpenError.
        Transitions to HALF_OPEN after 30s reset window.
      - HALF_OPEN: Next call acts as a probe. Success transitions to CLOSED;
        failure extends OPEN state for another 30s.
    """

    def __init__(
        self,
        failure_threshold: int = MLFLOW_CB_FAILURE_THRESHOLD,
        window_seconds: float = MLFLOW_CB_WINDOW_SECONDS,
        reset_seconds: float = MLFLOW_CB_RESET_SECONDS,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.reset_seconds = reset_seconds
        self._failures: list[float] = []
        self._open_until: float = 0.0
        self._state: str = "CLOSED"
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Current state of the circuit breaker ('CLOSED', 'OPEN', 'HALF_OPEN')."""
        with self._lock:
            self._evaluate_state()
            return self._state

    @property
    def is_open(self) -> bool:
        """True if calls should be rejected without calling MLflow."""
        with self._lock:
            self._evaluate_state()
            return self._state == "OPEN"

    def _evaluate_state(self) -> None:
        now = time.monotonic()
        if self._state == "OPEN":
            if now >= self._open_until:
                self._state = "HALF_OPEN"
                logger.info(
                    "CIRCUIT_HALF_OPEN: service=mlflow, probe_allowed=true"
                )

    def record_success(self) -> None:
        """Record a successful call to MLflow, resetting the circuit."""
        with self._lock:
            self._failures.clear()
            self._open_until = 0.0
            if self._state != "CLOSED":
                logger.info("CIRCUIT_CLOSED: service=mlflow, recovered=true")
            self._state = "CLOSED"

    def record_failure(self) -> None:
        """Record a failed call to MLflow, updating circuit breaker state."""
        now = time.monotonic()
        with self._lock:
            # Prune failures outside the sliding window
            cutoff = now - self.window_seconds
            self._failures = [t for t in self._failures if t >= cutoff]
            self._failures.append(now)

            if self._state == "HALF_OPEN" or len(self._failures) >= self.failure_threshold:
                self._state = "OPEN"
                self._open_until = now + self.reset_seconds
                logger.warning(
                    "CIRCUIT_OPEN: service=mlflow, failures=%d, open_until=%.2f",
                    len(self._failures),
                    self._open_until,
                )

    def reset(self) -> None:
        """Explicitly reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._failures.clear()
            self._open_until = 0.0
            self._state = "CLOSED"


class MLflowClient:
    """Client for interacting with the remote MLflow Model Registry service."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        timeout_ms: int = 5000,
        circuit_breaker: MLflowCircuitBreaker | None = None,
    ) -> None:
        self.tracking_uri = (tracking_uri or "http://localhost:5000").rstrip("/")
        self.connect_timeout = MLFLOW_CONNECT_TIMEOUT_SECONDS
        self.read_timeout = max(1.0, timeout_ms / 1000.0)
        self._circuit_breaker = circuit_breaker or MLflowCircuitBreaker()

    @classmethod
    def from_settings(cls, settings: Settings) -> "MLflowClient":
        """Instantiate MLflowClient from application settings."""
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            timeout_ms=settings.mlflow_timeout_ms,
        )

    @property
    def circuit_breaker(self) -> MLflowCircuitBreaker:
        """Underlying circuit breaker instance."""
        return self._circuit_breaker

    def _get_timeout_tuple(self) -> tuple[float, float]:
        return (float(self.connect_timeout), float(self.read_timeout))

    def fetch_model_metadata(
        self,
        name: str,
        version: str,
        request_id: str | None = None,
    ) -> MLflowModelMetadata:
        """Fetch model version metadata from MLflow Model Registry.

        Args:
            name: Registered model name.
            version: Registered model version string.
            request_id: Tracing UUID for request correlation.

        Returns:
            MLflowModelMetadata object with model details.

        Raises:
            MLflowCircuitBreakerOpenError: When circuit breaker is OPEN.
            MLflowUnavailableError: When MLflow is unreachable or returns error.
        """
        if self._circuit_breaker.is_open:
            logger.warning(
                "MLFLOW_CIRCUIT_REJECTED: model_name=%s, version=%s, request_id=%s",
                name,
                version,
                request_id or "NONE",
            )
            raise MLflowCircuitBreakerOpenError(
                "MLflow circuit breaker is OPEN. Fast-failing registration request."
            )

        url = f"{self.tracking_uri}/api/2.0/mlflow/model-versions/get"
        params = {"name": name, "version": str(version)}

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self._get_timeout_tuple(),
                headers={"Accept": "application/json"},
            )
            if response.status_code == 200:
                self._circuit_breaker.record_success()
                data = response.json().get("model_version", {})
                return MLflowModelMetadata(
                    name=data.get("name", name),
                    version=str(data.get("version", version)),
                    run_id=data.get("run_id", ""),
                    description=data.get("description"),
                    tags={t["key"]: t["value"] for t in data.get("tags", []) if "key" in t and "value" in t},
                    baseline_stats=None,
                )
            else:
                self._circuit_breaker.record_failure()
                logger.error(
                    "MLFLOW_ERROR: model_name=%s, version=%s, request_id=%s, http_status=%d",
                    name,
                    version,
                    request_id or "NONE",
                    response.status_code,
                )
                raise MLflowUnavailableError(
                    f"MLflow returned non-200 status code: {response.status_code}"
                )
        except requests.RequestException as exc:
            self._circuit_breaker.record_failure()
            logger.error(
                "MLFLOW_ERROR: model_name=%s, version=%s, request_id=%s, error_type=%s",
                name,
                version,
                request_id or "NONE",
                exc.__class__.__name__,
            )
            raise MLflowUnavailableError(
                f"Failed to connect to MLflow server: {exc.__class__.__name__}"
            ) from exc

    def check_health(self) -> bool:
        """Check if MLflow tracking server is reachable.

        Returns:
            True if MLflow responds with HTTP 200 within timeout, False otherwise.
        """
        if self._circuit_breaker.is_open:
            return False
        url = f"{self.tracking_uri}/health"
        try:
            response = requests.get(
                url,
                timeout=self._get_timeout_tuple(),
            )
            if response.status_code == 200:
                self._circuit_breaker.record_success()
                return True
            self._circuit_breaker.record_failure()
            return False
        except requests.RequestException:
            self._circuit_breaker.record_failure()
            return False
