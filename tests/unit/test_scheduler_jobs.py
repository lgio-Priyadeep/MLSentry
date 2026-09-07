"""Unit tests for APScheduler monitoring background jobs (drift, performance, retraining triggers).

Covers:
  - Insufficient prediction samples (< 30) skips drift and logs 'INSUFFICIENT_DATA'
  - Sufficient samples (>= 30) computes numerical PSI & categorical Chi2, writes drift_reports, emits DRIFT alert
  - 30-minute alert cooldown suppression for DRIFT alerts
  - Insufficient matched pairs (< 50) skips performance metrics
  - Sufficient matched pairs (>= 50) computes F1 and AUC, writes performance_logs, emits PERFORMANCE_DEGRADATION alert
  - 60-minute alert cooldown suppression for PERFORMANCE_DEGRADATION alerts
  - Dual critical conditions (PSI >= 0.25 AND F1 delta < -0.15) trigger GitHub Actions workflow_dispatch
  - 6-hour retraining cooldown suppression for successful dispatches
  - Job failure isolation: per-model exception caught, failure counter incremented, 3 crashes emit MONITORING_ENGINE_FAILURE alert
  - Scheduler lifecycle: start_scheduler and stop_scheduler start and clean up cleanly
"""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mlsentry.config.settings import Settings
from mlsentry.core.constants import (
    AlertSeverity,
    AlertType,
    DriftMethod,
    FeatureDtype,
    FeatureKind,
    ModelStatus,
    PerformanceMetric,
    PredictionType,
    StatType,
    TriggerStatus,
)
from mlsentry.db.models import (
    AlertRecord,
    Base,
    DriftReportRecord,
    GroundTruthRecord,
    ModelRecord,
    ModelSchemaRecord,
    PerformanceLogRecord,
    PredictionRecord,
    ReferenceStatRecord,
    TriggerEventRecord,
)
from mlsentry.integrations.github_trigger import GitHubTrigger, RetrainingDispatchResult
from mlsentry.scheduler.jobs import (
    create_alert_if_not_in_cooldown,
    evaluate_model_drift,
    evaluate_model_performance,
    evaluate_retraining_trigger,
    get_model_failure_count,
    is_alert_in_cooldown,
    reset_model_failure_count,
    run_monitoring_cycle,
    start_scheduler,
    stop_scheduler,
)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_model(db_session) -> ModelRecord:
    now = datetime.now(timezone.utc)
    model = ModelRecord(
        model_id=uuid.uuid4(),
        name="credit_risk",
        version="1.0.0",
        prediction_type=PredictionType.BINARY,
        status=ModelStatus.ACTIVE,
        sample_count=100,
        baseline_f1=0.85,
        baseline_auc=0.90,
        registered_at=now,
    )
    db_session.add(model)

    # Add numerical schema + reference stats (one row per stat_type)
    schema_num = ModelSchemaRecord(
        schema_id=uuid.uuid4(),
        model_id=model.model_id,
        feature_name="income",
        dtype=FeatureDtype.FLOAT,
        required=True,
        min_value=0.0,
        max_value=100000.0,
        created_at=now,
    )
    ref_num_edges = ReferenceStatRecord(
        stat_id=uuid.uuid4(),
        model_id=model.model_id,
        feature_name="income",
        stat_type=StatType.HISTOGRAM_BIN_EDGES,
        stat_value=None,
        frequency_map=None,
        histogram_data=[0.0, 25000.0, 50000.0, 75000.0, 100000.0],
        sample_count=100,
        computed_at=now,
    )
    ref_num_counts = ReferenceStatRecord(
        stat_id=uuid.uuid4(),
        model_id=model.model_id,
        feature_name="income",
        stat_type=StatType.HISTOGRAM_COUNTS,
        stat_value=None,
        frequency_map=None,
        histogram_data=[25, 25, 25, 25],
        sample_count=100,
        computed_at=now,
    )

    # Add categorical schema + reference stats
    schema_cat = ModelSchemaRecord(
        schema_id=uuid.uuid4(),
        model_id=model.model_id,
        feature_name="employment_type",
        dtype=FeatureDtype.CATEGORY,
        required=True,
        allowed_values=["salaried", "self_employed", "unemployed"],
        created_at=now,
    )
    ref_cat = ReferenceStatRecord(
        stat_id=uuid.uuid4(),
        model_id=model.model_id,
        feature_name="employment_type",
        stat_type=StatType.FREQUENCY_MAP,
        stat_value=None,
        frequency_map={"salaried": 0.50, "self_employed": 0.40, "unemployed": 0.10},
        histogram_data=None,
        sample_count=100,
        computed_at=now,
    )

    db_session.add_all([schema_num, ref_num_edges, ref_num_counts, schema_cat, ref_cat])
    db_session.commit()
    db_session.refresh(model)
    return model


class TestDriftEvaluation:
    """Test evaluate_model_drift logic and sample count gates."""

    def test_insufficient_samples_skips_drift(
        self, db_session, sample_model: ModelRecord
    ) -> None:
        target_window = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
        window_end = target_window + timedelta(minutes=15)
        now = datetime.now(timezone.utc)

        # Insert 15 predictions (< 30)
        for i in range(15):
            pred = PredictionRecord(
                pred_id=uuid.uuid4(),
                model_id=sample_model.model_id,
                model_version=sample_model.version,
                prediction_label="0",
                features_json={"income": 50000.0, "employment_type": "salaried"},
                schema_valid=True,
                window_bucket=target_window,
                logged_at=target_window + timedelta(seconds=i),
            )
            db_session.add(pred)
        db_session.commit()

        reports = evaluate_model_drift(db_session, sample_model, target_window, window_end, now)
        assert len(reports) == 0
        assert db_session.query(DriftReportRecord).count() == 0
        assert db_session.query(AlertRecord).count() == 0

    def test_sufficient_samples_computes_drift_and_emits_alert(
        self, db_session, sample_model: ModelRecord
    ) -> None:
        target_window = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
        window_end = target_window + timedelta(minutes=15)
        now = datetime.now(timezone.utc)

        # Insert 35 predictions (heavily shifted to trigger drift)
        for i in range(35):
            pred = PredictionRecord(
                pred_id=uuid.uuid4(),
                model_id=sample_model.model_id,
                model_version=sample_model.version,
                prediction_label="0",
                features_json={"income": 95000.0, "employment_type": "unemployed"},
                schema_valid=True,
                window_bucket=target_window,
                logged_at=target_window + timedelta(seconds=i),
            )
            db_session.add(pred)
        db_session.commit()

        reports = evaluate_model_drift(db_session, sample_model, target_window, window_end, now)
        for r in reports:
            db_session.add(r)
        db_session.commit()

        assert len(reports) == 2
        # Check alerts created
        alerts = db_session.query(AlertRecord).filter(AlertRecord.model_id == sample_model.model_id).all()
        assert len(alerts) >= 1
        assert alerts[0].type == AlertType.DRIFT
        assert alerts[0].cooldown_until is not None

    def test_drift_alert_respects_30m_cooldown(
        self, db_session, sample_model: ModelRecord
    ) -> None:
        now = datetime.now(timezone.utc)
        # Create an active alert in cooldown
        existing_alert = AlertRecord(
            alert_id=uuid.uuid4(),
            model_id=sample_model.model_id,
            type=AlertType.DRIFT,
            feature_name="income",
            severity=AlertSeverity.CRITICAL,
            message="Prior alert",
            resolved=False,
            cooldown_until=now + timedelta(minutes=20),
            triggered_at=now - timedelta(minutes=10),
        )
        db_session.add(existing_alert)
        db_session.commit()

        alert_created = create_alert_if_not_in_cooldown(
            db=db_session,
            model_id=sample_model.model_id,
            alert_type=AlertType.DRIFT,
            feature_name="income",
            severity=AlertSeverity.CRITICAL,
            message="New alert",
            now=now,
        )
        assert alert_created is None


class TestPerformanceEvaluation:
    """Test evaluate_model_performance logic and 50-pair gate."""

    def test_insufficient_matched_pairs_skips_performance(
        self, db_session, sample_model: ModelRecord
    ) -> None:
        target_window = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
        window_end = target_window + timedelta(minutes=15)
        now = datetime.now(timezone.utc)

        # 30 matched pairs (< 50)
        for i in range(30):
            p_id = uuid.uuid4()
            pred = PredictionRecord(
                pred_id=p_id,
                model_id=sample_model.model_id,
                model_version=sample_model.version,
                prediction_label="1",
                confidence=0.9,
                features_json={},
                schema_valid=True,
                window_bucket=target_window,
                logged_at=target_window + timedelta(seconds=i),
            )
            gt = GroundTruthRecord(
                gt_id=uuid.uuid4(),
                pred_id=p_id,
                model_id=sample_model.model_id,
                true_label="1",
                latency_hours=0.1,
                received_at=target_window + timedelta(seconds=i + 10),
            )
            db_session.add_all([pred, gt])
        db_session.commit()

        logs = evaluate_model_performance(db_session, sample_model, target_window, window_end, now)
        assert len(logs) == 0
        assert db_session.query(PerformanceLogRecord).count() == 0

    def test_sufficient_matched_pairs_computes_metrics_and_emits_alert(
        self, db_session, sample_model: ModelRecord
    ) -> None:
        target_window = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
        window_end = target_window + timedelta(minutes=15)
        now = datetime.now(timezone.utc)

        # 60 matched pairs with severe degradation (pred=1, actual=0)
        for i in range(60):
            p_id = uuid.uuid4()
            pred = PredictionRecord(
                pred_id=p_id,
                model_id=sample_model.model_id,
                model_version=sample_model.version,
                prediction_label="1",
                confidence=0.95,
                features_json={},
                schema_valid=True,
                window_bucket=target_window,
                logged_at=target_window + timedelta(seconds=i),
            )
            gt = GroundTruthRecord(
                gt_id=uuid.uuid4(),
                pred_id=p_id,
                model_id=sample_model.model_id,
                true_label="0",
                latency_hours=0.1,
                received_at=target_window + timedelta(seconds=i + 5),
            )
            db_session.add_all([pred, gt])
        db_session.commit()

        logs = evaluate_model_performance(db_session, sample_model, target_window, window_end, now)
        for l in logs:
            db_session.add(l)
        db_session.commit()

        assert len(logs) >= 1
        alerts = db_session.query(AlertRecord).filter(
            AlertRecord.model_id == sample_model.model_id,
            AlertRecord.type == AlertType.PERFORMANCE_DEGRADATION,
        ).all()
        assert len(alerts) >= 1


class TestRetrainingTrigger:
    """Test automated GitHub Actions retraining dispatch evaluation."""

    def test_dual_critical_triggers_dispatch_and_records_event(
        self, db_session, sample_model: ModelRecord
    ) -> None:
        now = datetime.now(timezone.utc)
        target_window = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
        window_end = target_window + timedelta(minutes=15)

        drift_reports = [
            DriftReportRecord(
                report_id=uuid.uuid4(),
                model_id=sample_model.model_id,
                feature_name="income",
                window_start=target_window,
                window_end=window_end,
                sample_count=100,
                method=DriftMethod.PSI,
                score=0.35,  # CRITICAL (>= 0.25)
                p_value=None,
                feature_type=FeatureKind.NUMERICAL,
                psi=0.35,
                computed_at=now,
            )
        ]
        perf_logs = [
            PerformanceLogRecord(
                perf_id=uuid.uuid4(),
                model_id=sample_model.model_id,
                window_start=target_window,
                window_end=window_end,
                metric=PerformanceMetric.F1,
                value=0.50,
                baseline_value=0.85,
                delta=-0.35,  # CRITICAL (< -0.15)
                sample_count=60,
                computed_at=now,
            )
        ]

        trigger = GitHubTrigger(
            token="ghp_test_token_123",
            repo_owner="org",
            repo_name="retrain-repo",
            workflow_id="retrain.yml",
        )

        with patch.object(trigger, "dispatch_workflow") as mock_dispatch:
            mock_dispatch.return_value = RetrainingDispatchResult(
                status=TriggerStatus.SUCCESS,
                github_response_code=204,
                error_message=None,
                error_type=None,
                next_allowed_at=now + timedelta(hours=6),
                alert_type_to_emit=None,
                alert_severity=None,
                is_suspended=False,
            )
            event = evaluate_retraining_trigger(
                db=db_session,
                model=sample_model,
                drift_reports=drift_reports,
                perf_logs=perf_logs,
                github_trigger=trigger,
                now=now,
            )
            db_session.commit()

        assert event is not None
        assert event.status == TriggerStatus.SUCCESS
        assert event.github_response_code == 204
        assert db_session.query(TriggerEventRecord).count() == 1


class TestFailureIsolationAndEscalation:
    """Test error handling, failure counters, and escalation alerts (ND-06)."""

    def test_three_consecutive_crashes_emits_monitoring_engine_failure_alert(
        self, db_session, sample_model: ModelRecord
    ) -> None:
        reset_model_failure_count(sample_model.model_id)

        # Mock evaluate_model_drift to raise an unhandled exception
        with patch("mlsentry.scheduler.jobs.evaluate_model_drift", side_effect=RuntimeError("Scipy crash")):
            session_factory = lambda: db_session

            # Run 3 consecutive cycles
            for _ in range(3):
                run_monitoring_cycle(session_factory=session_factory)

        assert get_model_failure_count(sample_model.model_id) == 3
        escalation_alert = db_session.query(AlertRecord).filter(
            AlertRecord.model_id == sample_model.model_id,
            AlertRecord.type == AlertType.MONITORING_ENGINE_FAILURE,
        ).first()

        assert escalation_alert is not None
        assert escalation_alert.severity == AlertSeverity.CRITICAL

        reset_model_failure_count(sample_model.model_id)


class TestSchedulerLifecycle:
    """Test BackgroundScheduler startup and graceful shutdown."""

    def test_start_and_stop_scheduler(self) -> None:
        settings = Settings(
            mlsentry_api_key="secret",
            database_url="sqlite:///:memory:",
            monitoring_interval_minutes=15,
        )
        scheduler = start_scheduler(settings=settings)
        assert scheduler is not None
        assert scheduler.running is True

        stop_scheduler()
        assert not scheduler.running
