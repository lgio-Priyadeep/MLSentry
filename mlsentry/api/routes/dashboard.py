"""FastAPI route handler for server-rendered HTML operations dashboard (GET /dashboard)."""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mlsentry.config.settings import Settings
from mlsentry.core.constants import (
    ALERT_COOLDOWN_MAP,
    RETRAINING_COOLDOWN_HOURS,
    WARM_UP_THRESHOLD,
    AlertSeverity,
    AlertType,
    DriftMethod,
    FeatureKind,
    ModelStatus,
    PerformanceMetric,
    TriggerStatus,
)
from mlsentry.core.drift.statistical import classify_chi2_severity, classify_psi_severity
from mlsentry.db.models import (
    AlertRecord,
    DriftReportRecord,
    ModelRecord,
    PerformanceLogRecord,
    ReferenceStatRecord,
    TriggerEventRecord,
)
from mlsentry.db.session import get_session

logger = logging.getLogger(__name__)

# Template configuration
_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "templates"
)
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(
    request: Request,
    model_id: str | None = Query(default=None, description="Optional UUID to filter by single model"),
    db: Session = Depends(get_session),
) -> HTMLResponse:
    """Render 3-section operations dashboard (Model Summary, Drift & Alerts, Performance Trends).
    
    Zero raw features_json and zero raw log_line data rendered to guarantee strict privacy.
    Unauthenticated route for local operations and management dashboards.
    """
    try:
        # 1. Fetch all models for selector dropdown
        all_models_query = select(ModelRecord).order_by(ModelRecord.name, ModelRecord.version)
        all_models = db.execute(all_models_query).scalars().all()

        # 2. Filter target models for summary table
        if model_id:
            try:
                target_uuid = uuid.UUID(model_id)
                target_models = [m for m in all_models if m.model_id == target_uuid]
            except ValueError:
                target_models = []
        else:
            target_models = [
                m for m in all_models
                if m.status not in (ModelStatus.DEPRECATED, ModelStatus.DEPRECATED.value)
            ]

        # Assemble Model Summary data
        model_summaries: list[dict[str, Any]] = []
        model_names_map: dict[uuid.UUID, str] = {m.model_id: f"{m.name} (v{m.version})" for m in all_models}

        for m in target_models:
            # Query last monitoring run timestamp from drift_reports
            last_drift = (
                db.execute(
                    select(DriftReportRecord.computed_at)
                    .filter(DriftReportRecord.model_id == m.model_id)
                    .order_by(desc(DriftReportRecord.computed_at))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            last_run_str = last_drift.strftime("%Y-%m-%d %H:%M:%S UTC") if last_drift else None

            model_summaries.append({
                "model_id": str(m.model_id),
                "name": m.name,
                "version": m.version,
                "prediction_type": m.prediction_type,
                "status": m.status,
                "sample_count": m.sample_count,
                "warm_up_threshold": WARM_UP_THRESHOLD,
                "baseline_f1": m.baseline_f1,
                "baseline_auc": m.baseline_auc,
                "last_run_at": last_run_str,
            })

        # 3. Fetch latest Drift Reports (compute dynamic severity per GI-05)
        target_uuids = [m.model_id for m in target_models]
        drift_rows = []
        if target_uuids:
            drift_query = (
                select(DriftReportRecord)
                .filter(DriftReportRecord.model_id.in_(target_uuids))
                .order_by(desc(DriftReportRecord.window_start), desc(DriftReportRecord.computed_at))
                .limit(50)
            )
            raw_drifts = db.execute(drift_query).scalars().all()

            for d in raw_drifts:
                # Dynamic severity classification
                method_val = d.method.value if hasattr(d.method, "value") else str(d.method)
                if method_val.lower() == "psi":
                    sev = classify_psi_severity(float(d.score))
                    thresh_summary = "WARN ≥ 0.10, CRIT ≥ 0.25"
                    method_name = "PSI"
                else:
                    p_val = float(d.p_value) if d.p_value is not None else 1.0
                    sev = classify_chi2_severity(p_val)
                    thresh_summary = "WARN p ≤ 0.05, CRIT p ≤ 0.01"
                    method_name = "CHI-SQUARE"

                drift_rows.append({
                    "model_name": model_names_map.get(d.model_id, str(d.model_id)[:8]),
                    "feature_name": d.feature_name,
                    "drift_method": method_name,
                    "drift_score": float(d.score),
                    "threshold_summary": thresh_summary,
                    "severity": sev.name.upper(),
                    "window_bucket": d.window_start.strftime("%Y-%m-%d %H:%M UTC") if d.window_start else "—",
                })

        # 4. Fetch Active Alerts (resolved IS FALSE)
        alert_rows = []
        active_alerts_query = select(AlertRecord).filter(AlertRecord.resolved.is_(False))
        if target_uuids:
            active_alerts_query = active_alerts_query.filter(AlertRecord.model_id.in_(target_uuids))
        active_alerts_query = active_alerts_query.order_by(desc(AlertRecord.triggered_at)).limit(50)
        raw_alerts = db.execute(active_alerts_query).scalars().all()

        total_critical = 0
        total_warning = 0
        for a in raw_alerts:
            sev_str = a.severity.value if hasattr(a.severity, "value") else str(a.severity)
            if sev_str.lower() == "critical":
                total_critical += 1
            elif sev_str.lower() == "warning":
                total_warning += 1

            type_str = a.type.value if hasattr(a.type, "value") else str(a.type)
            alert_rows.append({
                "alert_id": str(a.alert_id),
                "severity": sev_str.upper(),
                "alert_type": type_str,
                "model_name": model_names_map.get(a.model_id, str(a.model_id)[:8]),
                "feature_name": a.feature_name,
                "message": a.message,
                "triggered_at": a.triggered_at.strftime("%Y-%m-%d %H:%M:%S UTC") if a.triggered_at else "—",
            })

        # 5. Fetch Performance Trends (last 5 windows for single model or first target model)
        perf_labels: list[str] = []
        perf_f1_deltas: list[float | None] = []
        perf_auc_deltas: list[float | None] = []

        primary_model_uuid = target_models[0].model_id if target_models else None
        if primary_model_uuid:
            perf_query = (
                select(PerformanceLogRecord)
                .filter(PerformanceLogRecord.model_id == primary_model_uuid)
                .order_by(desc(PerformanceLogRecord.window_start), PerformanceLogRecord.metric)
                .limit(10)
            )
            raw_perf = list(db.execute(perf_query).scalars().all())

            # Group by window_start
            windows_dict: dict[datetime, dict[str, float | None]] = {}
            for p in raw_perf:
                w = p.window_start
                if w not in windows_dict:
                    windows_dict[w] = {"f1": None, "auc": None}
                metric_str = p.metric.value if hasattr(p.metric, "value") else str(p.metric)
                if metric_str.lower() == "f1":
                    windows_dict[w]["f1"] = round(float(p.delta), 4) if p.delta is not None else None
                elif metric_str.lower() == "auc":
                    windows_dict[w]["auc"] = round(float(p.delta), 4) if p.delta is not None else None

            # Sort chronologically (earliest to latest) for line chart display
            sorted_windows = sorted(windows_dict.keys())[-5:]
            for w in sorted_windows:
                lbl = w.strftime("%H:%M") if w else "W"
                perf_labels.append(lbl)
                perf_f1_deltas.append(windows_dict[w]["f1"])
                perf_auc_deltas.append(windows_dict[w]["auc"])

        # 6. Retraining Cooldown Status
        retrain_status = "ready"
        next_allowed_str = "None"
        if primary_model_uuid:
            last_trigger = (
                db.execute(
                    select(TriggerEventRecord)
                    .filter(
                        TriggerEventRecord.model_id == primary_model_uuid,
                        TriggerEventRecord.status == TriggerStatus.SUCCESS.value,
                    )
                    .order_by(desc(TriggerEventRecord.triggered_at))
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if last_trigger and last_trigger.next_allowed_at:
                now_utc = datetime.now(timezone.utc)
                if now_utc < last_trigger.next_allowed_at:
                    retrain_status = "cooling_down"
                    next_allowed_str = last_trigger.next_allowed_at.strftime("%Y-%m-%d %H:%M:%S UTC")

        context = {
            "request": request,
            "all_models": all_models,
            "selected_model_id": model_id,
            "models": model_summaries,
            "drift_reports": drift_rows,
            "active_alerts": alert_rows,
            "total_critical_alerts": total_critical,
            "total_warning_alerts": total_warning,
            "perf_chart_data": {
                "labels": perf_labels,
                "f1_deltas": perf_f1_deltas,
                "auc_deltas": perf_auc_deltas,
            },
            "retraining_status": retrain_status,
            "next_allowed_retrain": next_allowed_str,
        }

        return templates.TemplateResponse("dashboard.html", context)

    except Exception as exc:
        logger.error("DASHBOARD_RENDER_ERROR: %s", exc)
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": "Unable to render dashboard due to database connectivity issue."},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
