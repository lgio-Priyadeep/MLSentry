"""FastAPI application factory, lifespan management, and core diagnostic routes.

Lifespan startup sequence:
  1. Fail-fast validation of MLSENTRY_API_KEY.
  2. Database connection pool initialization via init_engine().
  3. Pre-load DistilBERT log classifier model checkpoint.
  4. Mount diagnostic and authenticated API routers.

Lifespan shutdown sequence:
  1. Dispose database connection pool via dispose_engine().
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from mlsentry.api.errors import (
    MLSentryAPIException,
    create_error_response,
    generic_exception_handler,
    get_request_id,
    http_exception_handler,
    mlsentry_api_exception_handler,
    validation_exception_handler,
)
from mlsentry.api.middleware import RequestIDMiddleware, get_api_key_dependency
from mlsentry.config.settings import Settings
from mlsentry.core.anomaly.log_classifier import DistilBERTUnavailableError, LogClassifier
from mlsentry.core.constants import SCHEDULER_HEARTBEAT_STALENESS_MINUTES
from mlsentry.db.session import dispose_engine, get_session, init_engine
from mlsentry.integrations.mlflow_client import MLflowClient

logger = logging.getLogger(__name__)

# Global runtime state for scheduler liveness tracking
_last_scheduler_cycle_at: datetime | None = None
_distilbert_classifier: LogClassifier | None = None


def set_scheduler_heartbeat(timestamp: datetime | None = None) -> None:
    """Record a scheduler execution heartbeat."""
    global _last_scheduler_cycle_at
    _last_scheduler_cycle_at = timestamp or datetime.now(timezone.utc)


def get_scheduler_heartbeat() -> datetime | None:
    """Retrieve last recorded scheduler heartbeat timestamp."""
    return _last_scheduler_cycle_at


def get_log_classifier() -> LogClassifier | None:
    """Retrieve global LogClassifier instance."""
    return _distilbert_classifier


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context managing startup validation and shutdown resource release."""
    global _distilbert_classifier

    settings: Settings = app.state.settings

    # 1. Fail-fast check for API Key
    if not settings.mlsentry_api_key or len(settings.mlsentry_api_key.strip()) == 0:
        logger.critical("STARTUP_FAILED: MLSENTRY_API_KEY is not configured")
        raise RuntimeError("MLSENTRY_API_KEY environment variable must be set.")

    # 2. Initialize database connection pool
    logger.info("Initializing database connection pool...")
    init_engine(settings.database_url)

    # 3. Pre-load DistilBERT model checkpoint
    logger.info("Pre-loading DistilBERT log classifier checkpoint...")
    _distilbert_classifier = LogClassifier(checkpoint=settings.distilbert_checkpoint)
    try:
        _distilbert_classifier.load_model()
        logger.info("DistilBERT model loaded successfully.")
    except (DistilBERTUnavailableError, Exception) as exc:
        logger.warning(
            "DistilBERT model failed to load at startup: %s. Classification route will return 503 until ready.",
            exc,
        )

    # 4. Attach shared classifier and MLflow client to app state
    if getattr(app.state, "classifier", None) is None:
        app.state.classifier = _distilbert_classifier
    if getattr(app.state, "mlflow_client", None) is None:
        app.state.mlflow_client = MLflowClient.from_settings(settings)

    # Initialize heartbeat for local startup readiness
    set_scheduler_heartbeat(datetime.now(timezone.utc))
    logger.info("MLSentry startup sequence completed.")

    yield

    # Shutdown sequence
    logger.info("Executing graceful shutdown...")
    dispose_engine()
    logger.info("Database connection pool disposed.")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app_settings = settings or Settings()

    app = FastAPI(
        title="MLSentry API",
        version="1.0.0",
        description="Machine Learning Monitoring, Statistical Drift & Performance Tracking Engine",
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    # Middleware
    app.add_middleware(RequestIDMiddleware)

    # Exception Handlers
    app.add_exception_handler(MLSentryAPIException, mlsentry_api_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Unauthenticated Health Route
    @app.get("/health", status_code=status.HTTP_200_OK)
    async def health_check(request: Request) -> Any:
        """System health check endpoint verifying database, DistilBERT, and scheduler liveness."""
        req_id = get_request_id(request)
        db_ok = False
        classifier_ok = False
        scheduler_ok = False

        # 1. Check Database connectivity
        try:
            for session in get_session():
                session.execute(text("SELECT 1"))
                db_ok = True
                break
        except Exception as exc:
            logger.warning("HEALTH_CHECK_DB_FAILED: %s", exc)
            db_ok = False

        # 2. Check Classifier readiness
        classifier = getattr(request.app.state, "classifier", None) or _distilbert_classifier
        cb = getattr(classifier, "_circuit_breaker", None) or getattr(classifier, "circuit_breaker", None)
        cb_is_open = getattr(cb, "is_open", False) if cb is not None else False
        if (
            classifier is not None
            and getattr(classifier, "_pipeline", None) is not None
            and not cb_is_open
        ):
            classifier_ok = True

        # 3. Check Scheduler heartbeat staleness
        heartbeat = get_scheduler_heartbeat()
        if heartbeat is not None:
            now = datetime.now(timezone.utc)
            elapsed_minutes = (now - heartbeat).total_seconds() / 60.0
            if elapsed_minutes <= SCHEDULER_HEARTBEAT_STALENESS_MINUTES:
                scheduler_ok = True

        if db_ok and classifier_ok and scheduler_ok:
            return {
                "status": "healthy",
                "components": {
                    "database": "ok",
                    "classifier": "ok",
                    "scheduler": "ok",
                },
                "last_scheduler_cycle_at": heartbeat.isoformat() if heartbeat else None,
            }
        else:
            return create_error_response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_code="SERVICE_UNAVAILABLE",
                message="One or more components are unhealthy or scheduler heartbeat is stale.",
                request_id=req_id,
            )

    # Authenticated Router setup
    auth_dep = get_api_key_dependency(app_settings.mlsentry_api_key)
    api_v1_router = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])

    # Mount domain route modules
    from mlsentry.api.routes.logs import router as logs_router
    from mlsentry.api.routes.models import router as models_router
    from mlsentry.api.routes.monitoring import router as monitoring_router
    from mlsentry.api.routes.predictions import router as predictions_router

    api_v1_router.include_router(models_router)
    api_v1_router.include_router(predictions_router)
    api_v1_router.include_router(logs_router)
    api_v1_router.include_router(monitoring_router)

    # Attach router to application
    app.include_router(api_v1_router)

    return app


# Default app instance for ASGI servers
app = create_app()
