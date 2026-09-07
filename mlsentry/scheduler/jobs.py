"""Background APScheduler monitoring worker for drift detection, performance tracking, and retraining triggers.

Implements:
  - run_monitoring_cycle:
      Main 15-minute execution loop (MONITORING_INTERVAL_MINUTES=15).
      Iterates over all ACTIVE models (or models with sample_count >= 30) for the current closed 15m window.
      1. Drift Evaluation:
         - Filters predictions WHERE schema_valid = true in window_bucket.
         - If sample count < 30: logs 'INSUFFICIENT_DATA: {count} samples, minimum 30 required' and skips drift computation.
         - If sample count >= 30: computes numerical PSI and categorical Chi-square against reference_stats,
           writes drift_reports rows, creates DRIFT alerts (PSI >= 0.10 or p <= 0.05) with 30m cooldown.
      2. Performance Evaluation:
         - Joins predictions (WHERE schema_valid = true) and ground_truth on pred_id for window_bucket.
         - If matched pairs < 50: logs skip reason and skips performance computation.
         - If matched pairs >= 50: calculates F1 score and AUC delta against baseline, writes performance_logs rows,
           creates PERFORMANCE_DEGRADATION alerts (delta < -0.05) with 60m cooldown.
      3. Retraining Trigger Evaluation:
         - Fires GitHub Actions workflow_dispatch ONLY upon simultaneous CRITICAL drift (PSI >= 0.25 or Chi2 p <= 0.01)
           AND CRITICAL performance degradation (delta < -0.15).
         - Enforces 6-hour retraining cooldown (WHERE status = 'success').
         - Writes trigger_events rows with status, response code, error_type, and sentinel next_allowed_at.
      4. Failure Isolation & Escalation (ND-06):
         - Catches exceptions per model without crashing the monitoring loop or FastAPI server.
         - Tracks in-memory consecutive failure counter per model_id.
         - Emits MONITORING_ENGINE_FAILURE alert (severity=CRITICAL, cooldown=none) after 3 consecutive crashes.
         - Updates scheduler heartbeat on cycle completion.
  - start_scheduler / stop_scheduler:
      Lifecycle helpers managing in-process BackgroundScheduler worker.
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import desc
from sqlalchemy.orm import Session

from mlsentry.config.settings import Settings
from mlsentry.core.constants import (
    ALERT_COOLDOWN_MAP,
    GITHUB_CONSECUTIVE_FAILURE_ALERT,
    MONITORING_INTERVAL_MINUTES,
    PERF_DELTA_CRITICAL,
    AlertSeverity,
    AlertType,
    DriftMethod,
    FeatureDtype,
    FeatureKind,
    ModelStatus,
    PerformanceMetric,
    StatType,
    TriggerStatus,
)
from mlsentry.core.drift.statistical import (
    calculate_chi_square,
    calculate_psi,
    classify_chi2_severity,
    classify_psi_severity,
    compute_histogram_counts,
    detect_categorical_drift,
    detect_numerical_drift,
)
from mlsentry.core.performance.tracker import (
    calculate_auc_score,
    calculate_f1_score,
    calculate_metric_delta,
    classify_perf_severity,
    evaluate_performance,
)
from mlsentry.db.models import (
    AlertRecord,
    DriftReportRecord,
    GroundTruthRecord,
    ModelRecord,
    ModelSchemaRecord,
    PerformanceLogRecord,
    PredictionRecord,
    ReferenceStatRecord,
    TriggerEventRecord,
)
from mlsentry.db.session import SessionLocal
from mlsentry.integrations.github_trigger import GitHubTrigger

logger = logging.getLogger(__name__)

# Module-level state
_scheduler: BackgroundScheduler | None = None
_scheduler_lock = threading.Lock()

_model_consecutive_failures: dict[uuid.UUID, int] = {}
_failure_lock = threading.Lock()


# ─── In-Memory Failure Tracking (ND-06) ──────────────────────────


def get_model_failure_count(model_id: uuid.UUID) -> int:
    """Retrieve current in-memory consecutive failure count for a model."""
    with _failure_lock:
        return _model_consecutive_failures.get(model_id, 0)


def increment_model_failure_count(model_id: uuid.UUID) -> int:
    """Increment and return in-memory consecutive failure count for a model."""
    with _failure_lock:
        cnt = _model_consecutive_failures.get(model_id, 0) + 1
        _model_consecutive_failures[model_id] = cnt
        return cnt


def reset_model_failure_count(model_id: uuid.UUID) -> None:
    """Reset in-memory consecutive failure count for a model to 0."""
    with _failure_lock:
        _model_consecutive_failures[model_id] = 0


# ─── Alert Cooldown Helpers ──────────────────────────────────────


def is_alert_in_cooldown(
    db: Session,
    model_id: uuid.UUID,
    alert_type: AlertType,
    feature_name: str | None,
    now: datetime,
) -> bool:
    """Check if an active cooldown exists for the given (model_id, feature_name, alert_type) tuple."""
    query = db.query(AlertRecord).filter(
        AlertRecord.model_id == model_id,
        AlertRecord.type == alert_type,
        AlertRecord.cooldown_until.isnot(None),
        AlertRecord.cooldown_until > now,
    )
    if feature_name is not None:
        query = query.filter(AlertRecord.feature_name == feature_name)
    else:
        query = query.filter(AlertRecord.feature_name.is_(None))

    return query.first() is not None


def create_alert_if_not_in_cooldown(
    db: Session,
    model_id: uuid.UUID,
    alert_type: AlertType,
    feature_name: str | None,
    severity: AlertSeverity,
    message: str,
    now: datetime,
) -> AlertRecord | None:
    """Create and persist an alert record if not currently suppressed by cooldown."""
    cooldown_delta = ALERT_COOLDOWN_MAP.get(alert_type, timedelta(minutes=0))
    cooldown_seconds = cooldown_delta.total_seconds()

    if cooldown_seconds > 0:
        if is_alert_in_cooldown(db, model_id, alert_type, feature_name, now):
            logger.info(
                "ALERT_SUPPRESSED_BY_COOLDOWN: model_id=%s, type=%s, feature=%s",
                model_id,
                alert_type.value,
                feature_name,
            )
            return None
        cooldown_until = now + cooldown_delta
    else:
        cooldown_until = None

    alert = AlertRecord(
        alert_id=uuid.uuid4(),
        model_id=model_id,
        type=alert_type,
        feature_name=feature_name,
        severity=severity,
        message=message,
        resolved=False,
        resolved_at=None,
        cooldown_until=cooldown_until,
        triggered_at=now,
    )
    db.add(alert)
    logger.warning(
        "ALERT_CREATED: alert_id=%s, model_id=%s, type=%s, severity=%s, feature=%s",
        alert.alert_id,
        model_id,
        alert_type.value,
        severity.value.upper(),
        feature_name,
    )
    return alert


# ─── Domain Evaluation Steps ─────────────────────────────────────


def evaluate_model_drift(
    db: Session,
    model: ModelRecord,
    window_start: datetime,
    window_end: datetime,
    now: datetime,
) -> list[DriftReportRecord]:
    """Evaluate statistical drift across all registered features for a model in target window."""
    valid_preds = (
        db.query(PredictionRecord)
        .filter(
            PredictionRecord.model_id == model.model_id,
            PredictionRecord.schema_valid.is_(True),
            PredictionRecord.window_bucket == window_start,
        )
        .all()
    )
    sample_count = len(valid_preds)

    if sample_count < 30:
        logger.info(
            "INSUFFICIENT_DATA: %d samples, minimum 30 required | model_id=%s window=%s",
            sample_count,
            model.model_id,
            window_start.isoformat(),
        )
        return []

    reference_stats = (
        db.query(ReferenceStatRecord)
        .filter(ReferenceStatRecord.model_id == model.model_id)
        .all()
    )
    schemas = (
        db.query(ModelSchemaRecord)
        .filter(ModelSchemaRecord.model_id == model.model_id)
        .all()
    )

    ref_map = {(r.feature_name, r.stat_type): r for r in reference_stats}
    reports: list[DriftReportRecord] = []

    for s in schemas:
        feat_name = s.feature_name
        raw_values = [
            p.features_json.get(feat_name)
            for p in valid_preds
            if p.features_json and feat_name in p.features_json and p.features_json[feat_name] is not None
        ]
        if not raw_values:
            continue

        if s.dtype in (FeatureDtype.FLOAT, FeatureDtype.INT):
            bin_stat = ref_map.get((feat_name, StatType.HISTOGRAM_BIN_EDGES))
            cnt_stat = ref_map.get((feat_name, StatType.HISTOGRAM_COUNTS))
            if not bin_stat or not cnt_stat or not bin_stat.histogram_data or not cnt_stat.histogram_data:
                continue

            sample_numbers: list[float] = []
            for v in raw_values:
                try:
                    sample_numbers.append(float(v))
                except (ValueError, TypeError):
                    pass

            if not sample_numbers:
                continue

            drift_res = detect_numerical_drift(
                feature_name=feat_name,
                current_values=sample_numbers,
                bin_edges=bin_stat.histogram_data,
                reference_counts=cnt_stat.histogram_data,
            )

            report = DriftReportRecord(
                report_id=uuid.uuid4(),
                model_id=model.model_id,
                feature_name=feat_name,
                window_start=window_start,
                window_end=window_end,
                sample_count=sample_count,
                method=DriftMethod.PSI,
                score=drift_res.score,
                p_value=None,
                feature_type=FeatureKind.NUMERICAL,
                psi=drift_res.psi,
                computed_at=now,
            )
            reports.append(report)

            if drift_res.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
                create_alert_if_not_in_cooldown(
                    db=db,
                    model_id=model.model_id,
                    alert_type=AlertType.DRIFT,
                    feature_name=feat_name,
                    severity=drift_res.severity,
                    message=f"Numerical feature '{feat_name}' drift detected: PSI = {drift_res.score:.4f} ({drift_res.severity.value.upper()})",
                    now=now,
                )

        elif s.dtype == FeatureDtype.CATEGORY:
            freq_stat = ref_map.get((feat_name, StatType.FREQUENCY_MAP))
            if not freq_stat or not freq_stat.frequency_map:
                continue

            sample_counts: dict[str, int] = {}
            for v in raw_values:
                k = str(v)
                sample_counts[k] = sample_counts.get(k, 0) + 1

            drift_res = detect_categorical_drift(
                feature_name=feat_name,
                current_values=sample_counts,
                frequency_map=freq_stat.frequency_map,
            )

            report = DriftReportRecord(
                report_id=uuid.uuid4(),
                model_id=model.model_id,
                feature_name=feat_name,
                window_start=window_start,
                window_end=window_end,
                sample_count=sample_count,
                method=DriftMethod.CHI_SQUARE,
                score=drift_res.score,
                p_value=drift_res.p_value,
                feature_type=FeatureKind.CATEGORICAL,
                psi=None,
                computed_at=now,
            )
            reports.append(report)

            if drift_res.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
                p_val_display = drift_res.p_value if drift_res.p_value is not None else 0.0
                create_alert_if_not_in_cooldown(
                    db=db,
                    model_id=model.model_id,
                    alert_type=AlertType.DRIFT,
                    feature_name=feat_name,
                    severity=drift_res.severity,
                    message=f"Categorical feature '{feat_name}' drift detected: Chi2 p-value = {p_val_display:.6f} ({drift_res.severity.value.upper()})",
                    now=now,
                )

    return reports


def evaluate_model_performance(
    db: Session,
    model: ModelRecord,
    window_start: datetime,
    window_end: datetime,
    now: datetime,
) -> list[PerformanceLogRecord]:
    """Evaluate performance metrics (F1 and AUC) for a model in target window."""
    matched_pairs = (
        db.query(PredictionRecord, GroundTruthRecord)
        .join(GroundTruthRecord, PredictionRecord.pred_id == GroundTruthRecord.pred_id)
        .filter(
            PredictionRecord.model_id == model.model_id,
            PredictionRecord.schema_valid.is_(True),
            PredictionRecord.window_bucket == window_start,
        )
        .all()
    )
    pair_count = len(matched_pairs)

    if pair_count < 50:
        logger.info(
            "INSUFFICIENT_MATCHED_PAIRS: %d matched pairs, minimum 50 required | model_id=%s window=%s",
            pair_count,
            model.model_id,
            window_start.isoformat(),
        )
        return []

    y_true: list[Any] = [gt.true_label for _, gt in matched_pairs]
    y_pred: list[Any] = [p.prediction_label for p, _ in matched_pairs]
    y_prob: list[float] = [
        float(p.confidence)
        for p, _ in matched_pairs
        if p.confidence is not None
    ]

    logs: list[PerformanceLogRecord] = []

    if model.baseline_f1 is not None:
        f1_value = calculate_f1_score(y_true, y_pred)
        delta_f1 = calculate_metric_delta(f1_value, float(model.baseline_f1))
        severity_f1 = classify_perf_severity(delta_f1)

        perf_log_f1 = PerformanceLogRecord(
            perf_id=uuid.uuid4(),
            model_id=model.model_id,
            window_start=window_start,
            window_end=window_end,
            metric=PerformanceMetric.F1,
            value=f1_value,
            baseline_value=float(model.baseline_f1),
            delta=delta_f1,
            sample_count=pair_count,
            computed_at=now,
        )
        logs.append(perf_log_f1)

        if severity_f1 in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
            create_alert_if_not_in_cooldown(
                db=db,
                model_id=model.model_id,
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                feature_name=None,
                severity=severity_f1,
                message=f"Model performance degradation: F1 delta = {delta_f1:.4f} ({severity_f1.value.upper()})",
                now=now,
            )

    if model.baseline_auc is not None and len(y_prob) == pair_count:
        try:
            auc_value = calculate_auc_score(y_true, y_prob)
            if auc_value is not None:
                delta_auc = calculate_metric_delta(auc_value, float(model.baseline_auc))
                severity_auc = classify_perf_severity(delta_auc)

                perf_log_auc = PerformanceLogRecord(
                    perf_id=uuid.uuid4(),
                    model_id=model.model_id,
                    window_start=window_start,
                    window_end=window_end,
                    metric=PerformanceMetric.AUC,
                    value=auc_value,
                    baseline_value=float(model.baseline_auc),
                    delta=delta_auc,
                    sample_count=pair_count,
                    computed_at=now,
                )
                logs.append(perf_log_auc)

                if severity_auc in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
                    create_alert_if_not_in_cooldown(
                        db=db,
                        model_id=model.model_id,
                        alert_type=AlertType.PERFORMANCE_DEGRADATION,
                        feature_name=None,
                        severity=severity_auc,
                        message=f"Model performance degradation: AUC delta = {delta_auc:.4f} ({severity_auc.value.upper()})",
                        now=now,
                    )
        except Exception as exc:
            logger.warning("AUC_COMPUTATION_SKIPPED: model_id=%s, reason=%s", model.model_id, exc)

    return logs


def evaluate_retraining_trigger(
    db: Session,
    model: ModelRecord,
    drift_reports: list[DriftReportRecord],
    perf_logs: list[PerformanceLogRecord],
    github_trigger: GitHubTrigger,
    now: datetime,
) -> TriggerEventRecord | None:
    """Evaluate conditions and dispatch automated retraining workflow via GitHub Actions."""
    has_critical_drift = any(
        (r.method == DriftMethod.PSI and r.score >= 0.25)
        or (r.method == DriftMethod.CHI_SQUARE and r.p_value is not None and r.p_value <= 0.01)
        for r in drift_reports
    )
    has_critical_perf = any(
        p.delta <= PERF_DELTA_CRITICAL
        for p in perf_logs
    )

    if not (has_critical_drift and has_critical_perf):
        return None

    last_success = (
        db.query(TriggerEventRecord)
        .filter(
            TriggerEventRecord.model_id == model.model_id,
            TriggerEventRecord.status == TriggerStatus.SUCCESS,
        )
        .order_by(desc(TriggerEventRecord.triggered_at))
        .first()
    )
    last_success_at = last_success.triggered_at if last_success else None

    if github_trigger.is_cooldown_active(last_success_at, now):
        return None

    recent_events = (
        db.query(TriggerEventRecord.status)
        .filter(TriggerEventRecord.model_id == model.model_id)
        .order_by(desc(TriggerEventRecord.triggered_at))
        .limit(3)
        .all()
    )
    recent_statuses = [r[0] for r in recent_events]
    if github_trigger.evaluate_consecutive_failures(recent_statuses):
        return None

    drift_summary = {
        "model_id": str(model.model_id),
        "model_name": model.name,
        "model_version": model.version,
        "triggered_at": now.isoformat(),
        "critical_features": [
            d.feature_name
            for d in drift_reports
            if (d.method == DriftMethod.PSI and float(d.score) >= 0.25)
            or (d.method == DriftMethod.CHI_SQUARE and d.p_value is not None and float(d.p_value) <= 0.01)
        ],
        "drift_reports": [
            {
                "feature": d.feature_name,
                "score": float(d.score),
                "p_value": float(d.p_value) if d.p_value is not None else None,
                "method": d.method.value,
            }
            for d in drift_reports
        ],
        "performance_logs": [
            {
                "metric": p.metric.value,
                "delta": float(p.delta),
                "value": float(p.value),
            }
            for p in perf_logs
        ],
    }

    result = github_trigger.dispatch_workflow(
        model_id=model.model_id,
        model_name=model.name,
        drift_report_summary=drift_summary,
        recent_failure_count=len([s for s in recent_statuses if s in (TriggerStatus.FAILED, "failed")]),
        triggered_at=now,
    )

    trigger_event = TriggerEventRecord(
        trigger_id=uuid.uuid4(),
        model_id=model.model_id,
        status=result.status,
        drift_report_summary=drift_summary,
        github_response_code=result.github_response_code,
        error_message=result.error_message,
        error_type=result.error_type,
        next_allowed_at=result.next_allowed_at,
        triggered_at=now,
    )
    db.add(trigger_event)

    if result.alert_type_to_emit is not None:
        create_alert_if_not_in_cooldown(
            db=db,
            model_id=model.model_id,
            alert_type=result.alert_type_to_emit,
            feature_name=None,
            severity=result.alert_severity or AlertSeverity.CRITICAL,
            message=f"Retraining dispatch failed for model '{model.name}': {result.error_message}",
            now=now,
        )

    return trigger_event


# ─── Cycle Execution & Lifecycle ─────────────────────────────────


def run_monitoring_cycle(
    session_factory: Any = None,
    settings: Settings | None = None,
) -> None:
    """Execute a complete 15-minute monitoring cycle across all active models."""
    from mlsentry.api.main import set_scheduler_heartbeat

    app_settings = settings or Settings()
    github_trigger = GitHubTrigger.from_settings(app_settings)
    session_maker = session_factory or SessionLocal

    now = datetime.now(timezone.utc)
    interval = app_settings.monitoring_interval_minutes or MONITORING_INTERVAL_MINUTES
    minute_bucket = (now.minute // interval) * interval
    window_start = now.replace(minute=minute_bucket, second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=interval)

    db: Session = session_maker()
    try:
        models = (
            db.query(ModelRecord)
            .filter(ModelRecord.status == ModelStatus.ACTIVE)
            .all()
        )

        for model in models:
            try:
                drift_reports = evaluate_model_drift(db, model, window_start, window_end, now)
                for r in drift_reports:
                    db.add(r)

                perf_logs = evaluate_model_performance(db, model, window_start, window_end, now)
                for p in perf_logs:
                    db.add(p)

                evaluate_retraining_trigger(db, model, drift_reports, perf_logs, github_trigger, now)

                db.commit()
                reset_model_failure_count(model.model_id)

            except Exception as exc:
                db.rollback()
                fail_cnt = increment_model_failure_count(model.model_id)
                tb = traceback.format_exc()
                logger.error(
                    "MONITORING_JOB_CRASH: model_id=%s, window=%s, exception_type=%s, failure_count=%d\n%s",
                    model.model_id,
                    window_start.isoformat(),
                    exc.__class__.__name__,
                    fail_cnt,
                    tb,
                )
                if fail_cnt >= GITHUB_CONSECUTIVE_FAILURE_ALERT:
                    logger.critical(
                        "ESCALATION_ALERT: model_id=%s, consecutive_failures=%d, window=%s",
                        model.model_id,
                        fail_cnt,
                        window_start.isoformat(),
                    )
                    try:
                        create_alert_if_not_in_cooldown(
                            db=db,
                            model_id=model.model_id,
                            alert_type=AlertType.MONITORING_ENGINE_FAILURE,
                            feature_name=None,
                            severity=AlertSeverity.CRITICAL,
                            message=f"Monitoring engine encountered {fail_cnt} consecutive crashes for model '{model.name}': {exc.__class__.__name__}",
                            now=now,
                        )
                        db.commit()
                    except Exception as alert_exc:
                        db.rollback()
                        logger.error("Failed to write escalation alert: %s", alert_exc)

        set_scheduler_heartbeat(now)
        logger.info(
            "MONITORING_CYCLE_COMPLETED: models_evaluated=%d, window=%s",
            len(models),
            window_start.isoformat(),
        )

    except Exception as outer_exc:
        db.rollback()
        logger.error("MONITORING_CYCLE_OUTER_CRASH: %s", outer_exc)
    finally:
        db.close()


def start_scheduler(
    settings: Settings | None = None,
    session_factory: Any = None,
) -> BackgroundScheduler:
    """Initialize and start background APScheduler worker."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            return _scheduler

        app_settings = settings or Settings()
        interval = app_settings.monitoring_interval_minutes or MONITORING_INTERVAL_MINUTES

        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=run_monitoring_cycle,
            trigger="interval",
            minutes=interval,
            id="mlsentry_monitoring_job",
            name="MLSentry 15-Minute Monitoring Cycle",
            misfire_grace_time=300,
            coalesce=True,
            max_instances=1,
            kwargs={"session_factory": session_factory, "settings": app_settings},
        )
        scheduler.start()
        _scheduler = scheduler
        logger.info("APScheduler background monitoring worker started with interval=%d minutes.", interval)
        return _scheduler


def stop_scheduler() -> None:
    """Gracefully stop background APScheduler worker."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            _scheduler.shutdown(wait=True)
            logger.info("APScheduler background monitoring worker shutdown gracefully.")
        _scheduler = None


def get_scheduler() -> BackgroundScheduler | None:
    """Retrieve global BackgroundScheduler instance."""
    return _scheduler
