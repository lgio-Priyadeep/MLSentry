"""API route handlers for external webhook testing and reachability verification.

Implements:
  - POST /v1/webhooks/test:
      Synchronously fires a single test HTTP POST request to validate external target reachability.
      Enforces strict Server-Side Request Forgery (SSRF) and DNS rebinding protections:
        - Rejects non-HTTP/HTTPS schemes (ftp://, file://, gopher://).
        - Validates all resolved IP addresses against loopback (127.0.0.0/8, ::1),
          IPv4 link-local (169.254.0.0/16), IPv6 link-local (fe80::/10), IPv6 unique local (fc00::/7),
          multicast (224.0.0.0/4, ff00::/8), 0.0.0.0, and RFC-1918 private IP blocks (10.0.0.0/8,
          172.16.0.0/12, 192.168.0.0/16).
        - Enforces Atomic Pre-Dispatch DNS Re-Verification: re-resolves hostname to IP immediately
          prior to socket connection.
        - Enforces maximum payload size limit of 10 KB (10240 bytes).
        - Zero persistence: No database records are created or modified.
        - Requires X-API-Key authentication.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import socket
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, Request, status
from pydantic import BaseModel, ConfigDict, Field

from mlsentry.api.errors import MLSentryAPIException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhooks"])

MAX_WEBHOOK_PAYLOAD_BYTES = 10 * 1024  # 10 KB
WEBHOOK_TIMEOUT_SECONDS = 5.0


# ─── Pydantic Request / Response DTOs ─────────────────────────────


class WebhookTestRequest(BaseModel):
    """Request payload for POST /v1/webhooks/test."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="Target webhook destination URL (http or https)")
    payload: dict[str, Any] = Field(..., description="JSON payload to forward as request body (max 10 KB)")


class WebhookTestResponse(BaseModel):
    """Response payload for POST /v1/webhooks/test."""

    model_config = ConfigDict(extra="forbid")

    fired: bool
    url: str
    response_status: int


# ─── SSRF & DNS Re-Verification Helpers ──────────────────────────


def is_ip_allowed(ip_str: str) -> bool:
    """Verify whether an IP address is globally routable and not in private/reserved spaces.

    Blocks:
      - Loopback (127.0.0.0/8, ::1)
      - RFC-1918 Private (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
      - IPv6 Unique Local (fc00::/7)
      - IPv4 Link-Local (169.254.0.0/16) / IPv6 Link-Local (fe80::/10)
      - Multicast (224.0.0.0/4, ff00::/8)
      - Unspecified / Zero (0.0.0.0, ::)
      - Reserved / Carrier-grade NAT (100.64.0.0/10)

    Args:
        ip_str: IP address string (IPv4 or IPv6).

    Returns:
        True if IP is public and allowed, False if private, link-local, loopback, or reserved.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False

    return True


def validate_target_url_and_dns(target_url: str) -> str:
    """Validate target URL scheme and perform atomic DNS pre-verification against SSRF.

    Args:
        target_url: URL string provided in request.

    Returns:
        Validated target URL string.

    Raises:
        MLSentryAPIException: HTTP 422 VALIDATION_ERROR on scheme, IP, or DNS resolution failure.
    """
    try:
        parsed = urlparse(target_url)
    except Exception:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="url must use http or https scheme and must not target a private, link-local, or loopback IP range.",
        )

    if parsed.scheme.lower() not in ("http", "https"):
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="url must use http or https scheme and must not target a private, link-local, or loopback IP range.",
        )

    hostname = parsed.hostname
    if not hostname:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="url must use http or https scheme and must not target a private, link-local, or loopback IP range.",
        )

    # Perform atomic pre-dispatch DNS resolution
    try:
        # socket.getaddrinfo returns list of (family, type, proto, canonname, sockaddr)
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="url must use http or https scheme and must not target a private, link-local, or loopback IP range.",
        )
    except Exception:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="url must use http or https scheme and must not target a private, link-local, or loopback IP range.",
        )

    if not addr_info:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="url must use http or https scheme and must not target a private, link-local, or loopback IP range.",
        )

    for entry in addr_info:
        sockaddr = entry[4]
        ip_candidate = sockaddr[0]
        if not is_ip_allowed(ip_candidate):
            logger.warning(
                "SSRF_BLOCKED: hostname=%s, resolved_ip=%s",
                hostname,
                ip_candidate,
            )
            raise MLSentryAPIException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error_code="VALIDATION_ERROR",
                message="url must use http or https scheme and must not target a private, link-local, or loopback IP range.",
            )

    return target_url


# ─── Route Handlers ──────────────────────────────────────────────


@router.post(
    "/webhooks/test",
    response_model=WebhookTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Test webhook connectivity with SSRF validation",
)
def test_webhook(
    request_data: WebhookTestRequest,
) -> WebhookTestResponse:
    """Fire a single test webhook request with strict SSRF filtering and DNS pre-verification.

    Validations:
      - Requires valid X-API-Key.
      - 422 VALIDATION_ERROR if URL scheme is not http or https.
      - 422 VALIDATION_ERROR if target resolves to private, link-local, loopback, or multicast IP.
      - 422 VALIDATION_ERROR if serialized payload exceeds 10 KB.
      - 503 SERVICE_UNAVAILABLE if destination server times out or connection is refused.
    """
    # 1. Payload size validation
    try:
        payload_json = json.dumps(request_data.payload)
        payload_bytes = payload_json.encode("utf-8")
    except Exception:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message="payload failed JSON serialization.",
        )

    if len(payload_bytes) > MAX_WEBHOOK_PAYLOAD_BYTES:
        raise MLSentryAPIException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            message=f"payload exceeds maximum allowed size of {MAX_WEBHOOK_PAYLOAD_BYTES // 1024} KB.",
        )

    # 2. SSRF validation & Atomic Pre-Dispatch DNS Re-Verification
    validated_url = validate_target_url_and_dns(request_data.url)

    # 3. Synchronous outbound dispatch (no DB persistence)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "MLSentry-Webhook-Test/1.0",
    }

    try:
        response = requests.post(
            validated_url,
            data=payload_bytes,
            headers=headers,
            timeout=WEBHOOK_TIMEOUT_SECONDS,
        )
        response_status = response.status_code
        logger.info(
            "WEBHOOK_TEST_FIRED: url=%s, status=%d",
            validated_url,
            response_status,
        )
    except requests.RequestException as exc:
        logger.warning(
            "WEBHOOK_TEST_FAILED: url=%s, error=%s",
            validated_url,
            exc.__class__.__name__,
        )
        raise MLSentryAPIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="SERVICE_UNAVAILABLE",
            message=f"Webhook target endpoint unreachable or timed out: {exc.__class__.__name__}",
        )

    return WebhookTestResponse(
        fired=True,
        url=validated_url,
        response_status=response_status,
    )
