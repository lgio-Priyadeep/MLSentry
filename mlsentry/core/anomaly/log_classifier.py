"""NLP log anomaly classifier runtime using HuggingFace DistilBERT.

Provides text classification for pipeline log strings, SLA latency tracking,
zero-raw-data PII hashing, and a thread-safe in-process circuit breaker.

Architectural rules & constraints:
  - Model: HuggingFace text-classification pipeline (distilbert-base-uncased).
  - SLA: P95 <= 500ms on CPU (DISTILBERT_SLA_MS). Latency > 500ms triggers
    structured SLA_BREACH log event without request abort (log-and-continue).
  - Circuit Breaker (ND-05): 5 consecutive inference failures trips circuit OPEN
    for 60 seconds (DISTILBERT_CB_RESET_SECONDS). While OPEN, calls immediately
    raise CircuitBreakerOpenError without invoking the model pipeline.
  - Zero Raw Data: Raw log_line is never logged or exposed in error messages;
    only SHA256 log_line_hash and classification metadata are recorded.
  - Input limits: 10 <= len(log_line) <= 2000 characters.

This is a pure domain/runtime module:
  - No database persistence (persistence performed by caller)
  - No direct FastAPI HTTP routing
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from mlsentry.core.constants import (
    DISTILBERT_CB_FAILURE_THRESHOLD,
    DISTILBERT_CB_RESET_SECONDS,
    DISTILBERT_SLA_MS,
    LOG_ANOMALY_CONFIDENCE_THRESHOLD,
    LOG_LINE_MAX_LENGTH,
    LOG_LINE_MIN_LENGTH,
    LogLabel,
)

logger = logging.getLogger(__name__)


class DistilBERTUnavailableError(Exception):
    """Raised when the DistilBERT model pipeline fails to load or execute."""


class CircuitBreakerOpenError(DistilBERTUnavailableError):
    """Raised when DistilBERT circuit breaker is in OPEN state."""


@dataclass(frozen=True)
class LogClassificationResult:
    """Result of classifying a single log line string.

    Attributes:
        log_line_hash: SHA256 hex digest of the raw log string.
        label: ANOMALOUS or NORMAL.
        confidence: Classification confidence score in [0.0, 1.0].
        alert_created: True if label is ANOMALOUS and confidence >= 0.85.
        latency_ms: Inference execution time in milliseconds.
        sla_breached: True if latency_ms > 500ms.
        model_checkpoint: Name/path of the active model checkpoint.
    """

    log_line_hash: str
    label: LogLabel
    confidence: float
    alert_created: bool
    latency_ms: float
    sla_breached: bool
    model_checkpoint: str


class DistilBERTCircuitBreaker:
    """Thread-safe in-process circuit breaker for DistilBERT inference.

    States:
      - CLOSED: Normal operation. Consecutive failures increment counter.
      - OPEN: Trips when consecutive failures reach threshold (5).
              Rejects requests immediately for reset window (60s).
      - HALF_OPEN: Probe request allowed after reset window. Success closes
                   circuit; failure extends OPEN state.
    """

    def __init__(
        self,
        failure_threshold: int = DISTILBERT_CB_FAILURE_THRESHOLD,
        reset_seconds: int = DISTILBERT_CB_RESET_SECONDS,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is currently in OPEN state."""
        with self._lock:
            if self._open_until == 0.0:
                return False
            now = time.monotonic()
            if now < self._open_until:
                return True
            # Transition from OPEN to HALF_OPEN (permit probe)
            return False

    def record_success(self) -> None:
        """Record successful execution, resetting circuit to CLOSED."""
        with self._lock:
            self._consecutive_failures = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        """Record failed execution. Trips circuit OPEN if threshold reached."""
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                now = time.monotonic()
                self._open_until = now + self.reset_seconds
                logger.error(
                    "CIRCUIT_OPEN: service=distilbert, failures=%d, open_until=%.2f",
                    self._consecutive_failures,
                    self._open_until,
                )


class LogClassifier:
    """DistilBERT log anomaly classification engine."""

    def __init__(
        self,
        checkpoint: str = "distilbert-base-uncased",
        pipeline_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        self._circuit_breaker = DistilBERTCircuitBreaker()
        self._lock = threading.Lock()

    @property
    def circuit_breaker(self) -> DistilBERTCircuitBreaker:
        """Underlying circuit breaker instance."""
        return self._circuit_breaker

    def load_model(self) -> None:
        """Eagerly load the HuggingFace text-classification pipeline."""
        with self._lock:
            if self._pipeline is not None:
                return
            try:
                if self._pipeline_factory is not None:
                    self._pipeline = self._pipeline_factory(self.checkpoint)
                else:
                    from transformers import pipeline

                    self._pipeline = pipeline(
                        "text-classification",
                        model=self.checkpoint,
                        device=-1,  # CPU inference
                    )
            except Exception as exc:
                logger.error(
                    "DISTILBERT_ERROR: exception_type=%s, checkpoint=%s",
                    type(exc).__name__,
                    self.checkpoint,
                )
                raise DistilBERTUnavailableError(
                    f"Failed to load DistilBERT checkpoint '{self.checkpoint}': {exc}"
                ) from exc

    def classify_log(
        self,
        log_line: str,
        confidence_threshold: float = LOG_ANOMALY_CONFIDENCE_THRESHOLD,
    ) -> LogClassificationResult:
        """Classify a log string as anomalous or normal.

        Args:
            log_line: Raw input log line string.
            confidence_threshold: Threshold to trigger LOG_ANOMALY alert (default 0.85).

        Returns:
            LogClassificationResult: Result with label, confidence, and SLA status.

        Raises:
            ValueError: If log_line is empty, whitespace, or outside length limits.
            CircuitBreakerOpenError: If circuit breaker is OPEN (tripped by consecutive failures).
            DistilBERTUnavailableError: If model inference fails.
        """
        # Defensive input validation
        cleaned = log_line.strip()
        if not cleaned:
            raise ValueError("log_line must not be empty or whitespace-only.")
        if len(cleaned) < LOG_LINE_MIN_LENGTH or len(cleaned) > LOG_LINE_MAX_LENGTH:
            raise ValueError(
                f"log_line length ({len(cleaned)}) must be between "
                f"{LOG_LINE_MIN_LENGTH} and {LOG_LINE_MAX_LENGTH} characters."
            )

        # Zero-raw-data hash for privacy-safe logging
        log_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        # Check circuit breaker before attempting inference
        if self._circuit_breaker.is_open:
            raise CircuitBreakerOpenError(
                "DistilBERT circuit breaker is OPEN due to consecutive failures."
            )

        # Ensure pipeline is loaded
        if self._pipeline is None:
            self.load_model()

        # Execute CPU inference with latency SLA timer wrapper
        start_time = time.monotonic()
        try:
            raw_output = self._pipeline(cleaned)
            self._circuit_breaker.record_success()
        except Exception as exc:
            self._circuit_breaker.record_failure()
            logger.error(
                "DISTILBERT_ERROR: exception_type=%s, log_line_hash=%s",
                type(exc).__name__,
                log_hash,
            )
            raise DistilBERTUnavailableError(
                f"DistilBERT inference failure: {exc}"
            ) from exc

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        sla_breached = elapsed_ms > DISTILBERT_SLA_MS

        if sla_breached:
            logger.warning(
                "SLA_BREACH: service=distilbert, latency_ms=%.2f, threshold_ms=%d, log_line_hash=%s",
                elapsed_ms,
                DISTILBERT_SLA_MS,
                log_hash,
            )

        # Parse pipeline output
        # Standard HF text-classification returns: [{'label': '...', 'score': 0.95}]
        item = raw_output[0] if isinstance(raw_output, list) else raw_output
        out_label = str(item.get("label", "")).strip().lower()
        score = float(item.get("score", 0.0))

        if "anom" in out_label or out_label in ("label_1", "1"):
            label = LogLabel.ANOMALOUS
        else:
            label = LogLabel.NORMAL

        confidence = round(float(min(max(score, 0.0), 1.0)), 4)
        alert_created = (
            label == LogLabel.ANOMALOUS and confidence >= confidence_threshold
        )

        return LogClassificationResult(
            log_line_hash=log_hash,
            label=label,
            confidence=confidence,
            alert_created=alert_created,
            latency_ms=round(elapsed_ms, 2),
            sla_breached=sla_breached,
            model_checkpoint=self.checkpoint,
        )
