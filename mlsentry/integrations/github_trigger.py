"""GitHub Actions workflow_dispatch integration for automated retraining triggers.

Dispatches retraining workflows when simultaneous CRITICAL drift and CRITICAL
performance degradation conditions are detected for a model.

Architectural rules & constraints:
  - Trigger condition: Dispatched ONLY when dual CRITICAL drift (PSI >= 0.25)
    AND CRITICAL performance degradation (F1Δ < -0.15 / AUCΔ < -0.15) occur.
  - Cooldown (GI-09): 6 hours enforced for successful dispatches (status='success').
    Query filters WHERE status = 'success'.
  - Sentinel (GI-09): Failed or suppressed dispatches write next_allowed_at =
    triggered_at + INTERVAL '1 second'.
  - Failure Escalation (ND-04): 3 consecutive 'failed' trigger_events rows for the
    same model_id triggers a 24-hour terminal suspension state ('DISPATCH_SUSPENDED')
    and emits a MONITORING_ENGINE_FAILURE alert (severity=CRITICAL, cooldown=none).
  - Timeouts (ND-03): Connect timeout = 5s (GITHUB_CONNECT_TIMEOUT_SECONDS),
    read timeout configurable via GITHUB_API_TIMEOUT_MS (default 10s).
  - Zero Token Leakage (NFR-12): GITHUB_TOKEN and Authorization header values must
    NEVER be logged, printed, or included in error traces.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from mlsentry.config.settings import Settings
from mlsentry.core.constants import (
    GITHUB_CONNECT_TIMEOUT_SECONDS,
    GITHUB_CONSECUTIVE_FAILURE_ALERT,
    GITHUB_DISPATCH_SUSPENDED_HOURS,
    RETRAINING_COOLDOWN_HOURS,
    TRIGGER_FAILURE_SENTINEL_SECONDS,
    AlertSeverity,
    AlertType,
    TriggerStatus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrainingDispatchResult:
    """Result of a GitHub Actions workflow_dispatch trigger attempt.

    Attributes:
        status: SUCCESS, FAILED, or SUPPRESSED.
        github_response_code: HTTP status returned by GitHub (204 on success, or None on network error).
        error_message: Error description on failure, or None.
        error_type: Canonical network error type ('NETWORK_TIMEOUT', 'DNS_FAILURE', 'CONNECTION_REFUSED'), or None.
        next_allowed_at: Timestamp calculated for next allowable dispatch attempt.
        alert_type_to_emit: AlertType to create (TRIGGER_FAILURE or MONITORING_ENGINE_FAILURE), or None.
        alert_severity: Severity of alert to emit (WARNING or CRITICAL), or None.
        is_suspended: True if 24-hour terminal backoff was activated due to 3 consecutive failures.
    """

    status: TriggerStatus
    github_response_code: int | None
    error_message: str | None
    error_type: str | None
    next_allowed_at: datetime
    alert_type_to_emit: AlertType | None
    alert_severity: AlertSeverity | None
    is_suspended: bool


class GitHubTrigger:
    """GitHub Actions workflow_dispatch trigger client and policy engine."""

    def __init__(
        self,
        token: str = "",
        repo_owner: str = "",
        repo_name: str = "",
        workflow_id: str = "retrain.yml",
        timeout_ms: int = 10000,
    ) -> None:
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.workflow_id = workflow_id
        self.connect_timeout = GITHUB_CONNECT_TIMEOUT_SECONDS
        self.read_timeout = max(1.0, timeout_ms / 1000.0)

    @classmethod
    def from_settings(cls, settings: Settings) -> "GitHubTrigger":
        """Instantiate GitHubTrigger from application settings."""
        return cls(
            token=settings.github_token,
            repo_owner=settings.github_repo_owner,
            repo_name=settings.github_repo_name,
            workflow_id=settings.github_workflow_id,
            timeout_ms=settings.github_api_timeout_ms,
        )

    def is_configured(self) -> bool:
        """True if required GitHub credentials and repository details are present."""
        return bool(self.token and self.repo_owner and self.repo_name)

    def is_cooldown_active(
        self,
        last_successful_trigger_at: datetime | None,
        current_time: datetime | None = None,
    ) -> bool:
        """Check if 6-hour retraining cooldown is active for successful dispatches.

        Args:
            last_successful_trigger_at: Timestamp of most recent trigger_events row with status='success'.
            current_time: Reference timestamp (defaults to UTC now).

        Returns:
            True if within 6-hour cooldown window; False if cooldown has expired or no prior success.
        """
        if last_successful_trigger_at is None:
            return False
        if last_successful_trigger_at.tzinfo is None:
            last_successful_trigger_at = last_successful_trigger_at.replace(
                tzinfo=timezone.utc
            )
        now = current_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cooldown_delta = timedelta(hours=RETRAINING_COOLDOWN_HOURS)
        return (now - last_successful_trigger_at) < cooldown_delta

    def evaluate_consecutive_failures(
        self,
        recent_statuses: list[TriggerStatus | str],
    ) -> bool:
        """Check if last 3 trigger events were all 'failed' (ND-04 escalation).

        Args:
            recent_statuses: List of recent trigger_events statuses in descending order of time.

        Returns:
            True if the 3 most recent events all have status == 'failed'.
        """
        if len(recent_statuses) < GITHUB_CONSECUTIVE_FAILURE_ALERT:
            return False
        first_three = recent_statuses[:GITHUB_CONSECUTIVE_FAILURE_ALERT]
        return all(
            (s == TriggerStatus.FAILED or s == "failed")
            for s in first_three
        )

    def dispatch_workflow(
        self,
        model_id: uuid.UUID | str,
        model_name: str,
        drift_report_summary: dict[str, Any],
        recent_failure_count: int = 0,
        triggered_at: datetime | None = None,
    ) -> RetrainingDispatchResult:
        """Dispatch GitHub Actions workflow_dispatch event for retraining.

        Args:
            model_id: UUID of model triggering retraining.
            model_name: Name of registered model.
            drift_report_summary: Summary dict containing drift/performance metrics.
            recent_failure_count: Number of consecutive failures prior to this attempt (for escalation).
            triggered_at: Event timestamp (defaults to UTC now).

        Returns:
            RetrainingDispatchResult containing status, response code, sentinel, and alert flags.
        """
        now = triggered_at or datetime.now(timezone.utc)
        model_id_str = str(model_id)

        if not self.is_configured():
            logger.warning(
                "GITHUB_TRIGGER_UNCONFIGURED: model_id=%s, status=suppressed",
                model_id_str,
            )
            return RetrainingDispatchResult(
                status=TriggerStatus.SUPPRESSED,
                github_response_code=None,
                error_message="GitHub trigger unconfigured (missing token or repository)",
                error_type=None,
                next_allowed_at=now + timedelta(seconds=TRIGGER_FAILURE_SENTINEL_SECONDS),
                alert_type_to_emit=None,
                alert_severity=None,
                is_suspended=False,
            )

        url = (
            f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            f"/actions/workflows/{self.workflow_id}/dispatches"
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload = {
            "ref": "main",
            "inputs": {
                "model_id": model_id_str,
                "model_name": model_name,
                "drift_summary": json.dumps(drift_report_summary),
                "trigger_reason": "CRITICAL_DRIFT_AND_PERFORMANCE",
            },
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(float(self.connect_timeout), float(self.read_timeout)),
            )
            if response.status_code == 204:
                logger.info(
                    "GITHUB_DISPATCH_SUCCESS: model_id=%s, status=204",
                    model_id_str,
                )
                return RetrainingDispatchResult(
                    status=TriggerStatus.SUCCESS,
                    github_response_code=204,
                    error_message=None,
                    error_type=None,
                    next_allowed_at=now + timedelta(hours=RETRAINING_COOLDOWN_HOURS),
                    alert_type_to_emit=None,
                    alert_severity=None,
                    is_suspended=False,
                )
            else:
                # HTTP Non-204 error response
                logger.error(
                    "GITHUB_DISPATCH_FAILED: http_status=%d, model_id=%s",
                    response.status_code,
                    model_id_str,
                )
                return self._handle_failure(
                    now=now,
                    model_id_str=model_id_str,
                    http_status=response.status_code,
                    error_msg=f"GitHub API returned HTTP {response.status_code}",
                    error_type=None,
                    recent_failure_count=recent_failure_count,
                )
        except requests.Timeout as exc:
            logger.error(
                "GITHUB_DISPATCH_FAILED: model_id=%s, error_type=NETWORK_TIMEOUT",
                model_id_str,
            )
            return self._handle_failure(
                now=now,
                model_id_str=model_id_str,
                http_status=None,
                error_msg="Network timeout connecting to GitHub Actions API",
                error_type="NETWORK_TIMEOUT",
                recent_failure_count=recent_failure_count,
            )
        except requests.ConnectionError as exc:
            err_str = str(exc)
            err_type = "DNS_FAILURE" if "NameResolutionError" in err_str or "getaddrinfo" in err_str else "CONNECTION_REFUSED"
            logger.error(
                "GITHUB_DISPATCH_FAILED: model_id=%s, error_type=%s",
                model_id_str,
                err_type,
            )
            return self._handle_failure(
                now=now,
                model_id_str=model_id_str,
                http_status=None,
                error_msg=f"Connection failure to GitHub API: {err_type}",
                error_type=err_type,
                recent_failure_count=recent_failure_count,
            )
        except requests.RequestException as exc:
            logger.error(
                "GITHUB_DISPATCH_FAILED: model_id=%s, error_type=REQUEST_FAILED",
                model_id_str,
            )
            return self._handle_failure(
                now=now,
                model_id_str=model_id_str,
                http_status=None,
                error_msg=f"Unexpected request failure: {exc.__class__.__name__}",
                error_type="CONNECTION_REFUSED",
                recent_failure_count=recent_failure_count,
            )

    def _handle_failure(
        self,
        now: datetime,
        model_id_str: str,
        http_status: int | None,
        error_msg: str,
        error_type: str | None,
        recent_failure_count: int,
    ) -> RetrainingDispatchResult:
        total_failures = recent_failure_count + 1
        if total_failures >= GITHUB_CONSECUTIVE_FAILURE_ALERT:
            logger.critical(
                "GITHUB_DISPATCH_SUSPENDED: model_id=%s, consecutive_failures=%d, suspended_hours=24",
                model_id_str,
                total_failures,
            )
            return RetrainingDispatchResult(
                status=TriggerStatus.FAILED,
                github_response_code=http_status,
                error_message=error_msg,
                error_type=error_type,
                next_allowed_at=now + timedelta(hours=GITHUB_DISPATCH_SUSPENDED_HOURS),
                alert_type_to_emit=AlertType.MONITORING_ENGINE_FAILURE,
                alert_severity=AlertSeverity.CRITICAL,
                is_suspended=True,
            )
        return RetrainingDispatchResult(
            status=TriggerStatus.FAILED,
            github_response_code=http_status,
            error_message=error_msg,
            error_type=error_type,
            next_allowed_at=now + timedelta(seconds=TRIGGER_FAILURE_SENTINEL_SECONDS),
            alert_type_to_emit=AlertType.TRIGGER_FAILURE,
            alert_severity=AlertSeverity.WARNING,
            is_suspended=False,
        )
