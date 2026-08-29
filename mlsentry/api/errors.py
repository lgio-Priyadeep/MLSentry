"""Global API error handling, exception hierarchies, and standardized error envelopes.

Every non-2xx response from MLSentry conforms to the standard error envelope:
    {
        "error": "SCREAMING_SNAKE_CASE_CODE",
        "message": "Human-readable developer explanation.",
        "request_id": "550e8400-e29b-41d4-a716-446655440000"
    }

Clients branch exclusively on the `error` string, never on HTTP status or `message`.
Zero Raw Data: Raw feature values and log line strings are scrubbed from all error responses.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class MLSentryAPIException(Exception):
    """Base application exception mapping directly to canonical error envelopes.

    Attributes:
        status_code: HTTP response status code (e.g. 401, 404, 409, 422, 500, 503).
        error_code: Canonical machine-parseable SCREAMING_SNAKE_CASE error string.
        message: Developer-friendly explanation without raw features or PII.
    """

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


def get_request_id(request: Request) -> str:
    """Extract request_id from request state or generate a fallback UUIDv4."""
    if hasattr(request, "state") and hasattr(request.state, "request_id"):
        return str(request.state.request_id)
    return str(uuid.uuid4())


def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    request_id: str,
) -> JSONResponse:
    """Construct standard JSON error response envelope."""
    payload = {
        "error": error_code,
        "message": message,
        "request_id": request_id,
    }
    return JSONResponse(status_code=status_code, content=payload)


async def mlsentry_api_exception_handler(
    request: Request, exc: MLSentryAPIException
) -> JSONResponse:
    """Handler for explicit MLSentryAPIException instances."""
    req_id = get_request_id(request)
    logger.warning(
        "API_EXCEPTION: status=%d, error=%s, request_id=%s, message=%s",
        exc.status_code,
        exc.error_code,
        req_id,
        exc.message,
    )
    return create_error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        request_id=req_id,
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handler for Starlette/FastAPI HTTPException."""
    req_id = get_request_id(request)
    status_code = exc.status_code

    # Map generic HTTP status codes to canonical error codes
    code_map = {
        status.HTTP_400_BAD_REQUEST: "MISSING_FIELD",
        status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
        status.HTTP_403_FORBIDDEN: "FORBIDDEN",
        status.HTTP_404_NOT_FOUND: "NOT_FOUND",
        status.HTTP_409_CONFLICT: "CONFLICT",
        status.HTTP_422_UNPROCESSABLE_ENTITY: "VALIDATION_ERROR",
        status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
        status.HTTP_500_INTERNAL_SERVER_ERROR: "INTERNAL_ERROR",
        status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
    }
    error_code = code_map.get(status_code, "INTERNAL_ERROR")
    message = str(exc.detail) if exc.detail else "An HTTP error occurred."

    logger.warning(
        "HTTP_EXCEPTION: status=%d, error=%s, request_id=%s",
        status_code,
        error_code,
        req_id,
    )
    return create_error_response(
        status_code=status_code,
        error_code=error_code,
        message=message,
        request_id=req_id,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handler for FastAPI/Pydantic RequestValidationError."""
    req_id = get_request_id(request)
    error_code = "VALIDATION_ERROR"
    message = "Request validation failed."

    # Inspect first error to supply specific canonical error codes if matched
    errors = exc.errors()
    if errors:
        first_err = errors[0]
        loc = first_err.get("loc", ())
        msg = first_err.get("msg", "")
        ctx = first_err.get("ctx", {})

        # 1. If an explicit MLSentryAPIException was raised inside a validator
        if "error" in ctx and isinstance(ctx["error"], MLSentryAPIException):
            api_exc = ctx["error"]
            logger.warning(
                "VALIDATION_ERROR (MLSentryAPIException): error=%s, request_id=%s, message=%s",
                api_exc.error_code,
                req_id,
                api_exc.message,
            )
            return create_error_response(
                status_code=api_exc.status_code,
                error_code=api_exc.error_code,
                message=api_exc.message,
                request_id=req_id,
            )

        field_name = str(loc[-1]) if loc else ""
        if field_name == "name":
            error_code = "INVALID_MODEL_NAME"
            message = "Model name must match regex ^[a-z0-9_-]+$"
        elif field_name == "version":
            error_code = "INVALID_MODEL_VERSION"
            message = "Model version must match regex ^\\d+\\.\\d+(\\.\\d+)?$"
        elif field_name == "prediction_type":
            error_code = "INVALID_PREDICTION_TYPE"
            message = "Prediction type must be 'binary', 'multiclass', or 'regression'."
        elif "frequency_map" in msg.lower():
            error_code = "FREQUENCY_MAP_INVALID"
            message = msg
        elif "histogram_bin_edges" in msg.lower() or "histogram_counts" in msg.lower() or "baseline_stats" in msg.lower():
            error_code = "INVALID_BASELINE_STATS"
            message = msg
        elif field_name == "features_json":
            if "size" in msg.lower() or "bytes" in msg.lower():
                error_code = "FEATURES_JSON_TOO_LARGE"
                message = "features_json exceeds maximum permitted size (64 KB / 200 keys)."
            elif "nested" in msg.lower():
                error_code = "FEATURES_JSON_NESTED"
                message = "features_json must contain scalar values only (no nested objects)."
            else:
                error_code = "SCHEMA_VALIDATION_FAILED"
                message = f"features_json validation failed: {msg}"
        elif field_name == "log_line":
            if "empty" in msg.lower():
                error_code = "EMPTY_LOG_LINE"
                message = "log_line cannot be empty."
            else:
                error_code = "INVALID_LOG_LINE"
                message = f"log_line validation failed: {msg}"
        else:
            message = f"Validation failed for field '{field_name}': {msg}"

    logger.warning(
        "VALIDATION_ERROR: error=%s, request_id=%s, message=%s",
        error_code,
        req_id,
        message,
    )
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code=error_code,
        message=message,
        request_id=req_id,
    )


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handler for unexpected unhandled exceptions (500 Internal Error)."""
    req_id = get_request_id(request)
    logger.error(
        "UNHANDLED_EXCEPTION: request_id=%s, error_type=%s",
        req_id,
        exc.__class__.__name__,
        exc_info=True,
    )
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_ERROR",
        message="An unexpected internal server error occurred.",
        request_id=req_id,
    )
