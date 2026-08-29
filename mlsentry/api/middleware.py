"""API middleware components and authentication dependencies.

Includes:
  - RequestIDMiddleware: Generates or propagates UUIDv4 X-Request-ID headers.
  - verify_api_key: Constant-time HMAC comparison dependency for X-API-Key.
"""
from __future__ import annotations

import hmac
import logging
import uuid
from typing import Any, Callable

from fastapi import Header, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from mlsentry.api.errors import MLSentryAPIException
from mlsentry.config.settings import Settings

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that attaches a unique UUIDv4 request_id to request state and response headers."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        incoming_id = request.headers.get("X-Request-ID")
        if incoming_id:
            request_id = incoming_id
        else:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def get_api_key_dependency(expected_key: str) -> Callable[[str | None], str]:
    """Factory returning a FastAPI dependency that verifies X-API-Key using constant-time comparison.

    Args:
        expected_key: Configured secret API key loaded from MLSENTRY_API_KEY.

    Returns:
        FastAPI dependency function.
    """

    async def verify_api_key(
        x_api_key: str | None = Header(None, alias="X-API-Key")
    ) -> str:
        if not x_api_key:
            raise MLSentryAPIException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                error_code="UNAUTHORIZED",
                message="Missing required 'X-API-Key' authentication header.",
            )

        # Constant-time comparison to prevent timing attacks
        is_valid = hmac.compare_digest(
            x_api_key.encode("utf-8"),
            expected_key.encode("utf-8"),
        )
        if not is_valid:
            raise MLSentryAPIException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                error_code="UNAUTHORIZED",
                message="Invalid 'X-API-Key' credential provided.",
            )

        return x_api_key

    return verify_api_key
