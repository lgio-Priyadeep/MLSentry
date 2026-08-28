"""Unit tests for GitHub Actions retraining workflow dispatch integration."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from mlsentry.config.settings import Settings
from mlsentry.core.constants import AlertSeverity, AlertType, TriggerStatus
from mlsentry.integrations.github_trigger import GitHubTrigger, RetrainingDispatchResult


@pytest.fixture
def configured_trigger() -> GitHubTrigger:
    return GitHubTrigger(
        token="ghp_test_token_secret_12345",
        repo_owner="acme-corp",
        repo_name="ml-models",
        workflow_id="retrain.yml",
        timeout_ms=5000,
    )


class TestGitHubTrigger:
    """Test suite for GitHubActions trigger dispatching and cooldown policy."""

    def test_unconfigured_trigger_returns_suppressed(self) -> None:
        unconfigured = GitHubTrigger()
        assert unconfigured.is_configured() is False

        model_id = uuid.uuid4()
        res = unconfigured.dispatch_workflow(
            model_id=model_id,
            model_name="churn_model",
            drift_report_summary={"psi": 0.35, "f1_delta": -0.22},
        )
        assert res.status == TriggerStatus.SUPPRESSED
        assert res.github_response_code is None
        assert res.alert_type_to_emit is None

    def test_cooldown_active_within_6_hours(
        self, configured_trigger: GitHubTrigger
    ) -> None:
        now = datetime.now(timezone.utc)
        recent_success = now - timedelta(hours=5, minutes=59)
        assert configured_trigger.is_cooldown_active(recent_success, now) is True

    def test_cooldown_expired_after_6_hours(
        self, configured_trigger: GitHubTrigger
    ) -> None:
        now = datetime.now(timezone.utc)
        old_success = now - timedelta(hours=6, minutes=1)
        assert configured_trigger.is_cooldown_active(old_success, now) is False
        assert configured_trigger.is_cooldown_active(None, now) is False

    @patch("mlsentry.integrations.github_trigger.requests.post")
    def test_dispatch_workflow_success_204(
        self, mock_post: MagicMock, configured_trigger: GitHubTrigger
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_post.return_value = mock_resp

        model_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        res = configured_trigger.dispatch_workflow(
            model_id=model_id,
            model_name="churn_model",
            drift_report_summary={"psi": 0.40, "f1_delta": -0.20},
            triggered_at=now,
        )

        assert res.status == TriggerStatus.SUCCESS
        assert res.github_response_code == 204
        assert res.error_message is None
        assert res.error_type is None
        assert res.alert_type_to_emit is None
        assert res.next_allowed_at == now + timedelta(hours=6)

        # Verify Authorization header used Bearer token
        called_headers = mock_post.call_args.kwargs["headers"]
        assert called_headers["Authorization"] == "Bearer ghp_test_token_secret_12345"

    @patch("mlsentry.integrations.github_trigger.requests.post")
    def test_dispatch_workflow_http_error_emits_trigger_failure(
        self, mock_post: MagicMock, configured_trigger: GitHubTrigger
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        model_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        res = configured_trigger.dispatch_workflow(
            model_id=model_id,
            model_name="churn_model",
            drift_report_summary={"psi": 0.40},
            recent_failure_count=0,
            triggered_at=now,
        )

        assert res.status == TriggerStatus.FAILED
        assert res.github_response_code == 500
        assert res.error_type is None
        assert res.alert_type_to_emit == AlertType.TRIGGER_FAILURE
        assert res.alert_severity == AlertSeverity.WARNING
        assert res.next_allowed_at == now + timedelta(seconds=1)
        assert res.is_suspended is False

    @patch("mlsentry.integrations.github_trigger.requests.post")
    def test_dispatch_workflow_network_timeout(
        self, mock_post: MagicMock, configured_trigger: GitHubTrigger
    ) -> None:
        mock_post.side_effect = requests.Timeout("Network timed out")

        model_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        res = configured_trigger.dispatch_workflow(
            model_id=model_id,
            model_name="churn_model",
            drift_report_summary={"psi": 0.40},
            recent_failure_count=1,
            triggered_at=now,
        )

        assert res.status == TriggerStatus.FAILED
        assert res.github_response_code is None
        assert res.error_type == "NETWORK_TIMEOUT"
        assert res.alert_type_to_emit == AlertType.TRIGGER_FAILURE
        assert res.next_allowed_at == now + timedelta(seconds=1)

    @patch("mlsentry.integrations.github_trigger.requests.post")
    def test_three_consecutive_failures_escalates_to_terminal_suspension(
        self, mock_post: MagicMock, configured_trigger: GitHubTrigger
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_post.return_value = mock_resp

        model_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        # 3rd consecutive failure (recent_failure_count=2 -> total 3)
        res = configured_trigger.dispatch_workflow(
            model_id=model_id,
            model_name="churn_model",
            drift_report_summary={"psi": 0.40},
            recent_failure_count=2,
            triggered_at=now,
        )

        assert res.status == TriggerStatus.FAILED
        assert res.github_response_code == 503
        assert res.is_suspended is True
        assert res.alert_type_to_emit == AlertType.MONITORING_ENGINE_FAILURE
        assert res.alert_severity == AlertSeverity.CRITICAL
        assert res.next_allowed_at == now + timedelta(hours=24)

    def test_from_settings(self) -> None:
        settings = Settings(
            mlsentry_api_key="secret-key",
            database_url="postgresql://user:pass@localhost:5432/db",
            github_token="ghp_test_token_secret_12345",
            github_repo_owner="acme-corp",
            github_repo_name="ml-models",
            github_workflow_id="custom-retrain.yml",
            github_api_timeout_ms=8000,
        )
        trigger = GitHubTrigger.from_settings(settings)
        assert trigger.token == "ghp_test_token_secret_12345"
        assert trigger.repo_owner == "acme-corp"
        assert trigger.repo_name == "ml-models"
        assert trigger.workflow_id == "custom-retrain.yml"
        assert trigger.read_timeout == 8.0

    def test_cooldown_with_naive_datetime(
        self, configured_trigger: GitHubTrigger
    ) -> None:
        now = datetime.now(timezone.utc)
        naive_recent = (now - timedelta(hours=2)).replace(tzinfo=None)
        assert configured_trigger.is_cooldown_active(naive_recent, now) is True
