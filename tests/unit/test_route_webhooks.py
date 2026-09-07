"""Unit tests for webhook test shim route: POST /v1/webhooks/test (SSRF & DNS validation).

Covers:
  - 401 UNAUTHORIZED on missing or invalid API key
  - 422 VALIDATION_ERROR on non-HTTP/HTTPS scheme (ftp://, file://, gopher://)
  - 422 VALIDATION_ERROR on loopback IP (127.0.0.1, localhost, [::1])
  - 422 VALIDATION_ERROR on RFC-1918 private IPs (10.0.0.1, 172.16.0.1, 192.168.1.1)
  - 422 VALIDATION_ERROR on IPv4 link-local (169.254.169.254) and 0.0.0.0
  - 422 VALIDATION_ERROR on multicast addresses (224.0.0.1)
  - 422 VALIDATION_ERROR on unresolvable hostname (DNS resolution error)
  - 422 VALIDATION_ERROR on oversized payload (> 10 KB)
  - 200 OK on valid public destination URL (with mocked requests.post and socket.getaddrinfo)
  - 503 SERVICE_UNAVAILABLE on destination connection failure or timeout
  - Assertion that no database rows are created (zero persistence)
"""
import socket
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi import APIRouter, Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.exceptions import HTTPException as StarletteHTTPException

from mlsentry.api.errors import (
    MLSentryAPIException,
    generic_exception_handler,
    http_exception_handler,
    mlsentry_api_exception_handler,
    validation_exception_handler,
)
from mlsentry.api.middleware import RequestIDMiddleware, get_api_key_dependency
from mlsentry.api.routes.webhooks import router as webhooks_router
from mlsentry.config.settings import Settings
from mlsentry.db.models import AlertRecord, Base
from mlsentry.db.session import get_session

TEST_API_KEY = "test_secret_webhooks_key_12345"


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
def client(db_session):
    app = FastAPI()
    app.state.settings = Settings(
        mlsentry_api_key=TEST_API_KEY,
        database_url="sqlite:///:memory:",
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(MLSentryAPIException, mlsentry_api_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    auth_dep = get_api_key_dependency(TEST_API_KEY)

    def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    api_v1 = APIRouter(prefix="/v1", dependencies=[Depends(auth_dep)])
    api_v1.include_router(webhooks_router)
    app.include_router(api_v1)

    return TestClient(app)


class TestWebhookAuthentication:
    """Test authentication requirements on POST /v1/webhooks/test."""

    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        payload = {"url": "https://hooks.example.com/alerts", "payload": {"event": "test"}}
        resp = client.post("/v1/webhooks/test", json=payload)
        assert resp.status_code == status.HTTP_401_UNAUTHORIZED
        assert resp.json()["error"] == "UNAUTHORIZED"

    def test_invalid_api_key_returns_401(self, client: TestClient) -> None:
        payload = {"url": "https://hooks.example.com/alerts", "payload": {"event": "test"}}
        resp = client.post(
            "/v1/webhooks/test",
            json=payload,
            headers={"X-API-Key": "wrong_key_xyz"},
        )
        assert resp.status_code == status.HTTP_401_UNAUTHORIZED
        assert resp.json()["error"] == "UNAUTHORIZED"


class TestWebhookSSRFValidation:
    """Test SSRF IP filtering and scheme verification."""

    @pytest.mark.parametrize(
        "invalid_scheme_url",
        [
            "ftp://hooks.example.com/alerts",
            "file:///etc/passwd",
            "gopher://hooks.example.com/",
            "ws://hooks.example.com/alerts",
        ],
    )
    def test_invalid_schemes_rejected_422(
        self, client: TestClient, invalid_scheme_url: str
    ) -> None:
        payload = {"url": invalid_scheme_url, "payload": {"event": "test"}}
        resp = client.post(
            "/v1/webhooks/test",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert resp.json()["error"] == "VALIDATION_ERROR"

    @pytest.mark.parametrize(
        "blocked_ip_url,mocked_ip",
        [
            ("http://127.0.0.1:8080/hook", "127.0.0.1"),
            ("http://localhost:5000/hook", "127.0.0.1"),
            ("http://0.0.0.0:8000/hook", "0.0.0.0"),
            ("http://10.0.0.1/hook", "10.0.0.1"),
            ("http://172.16.0.5/hook", "172.16.0.5"),
            ("http://192.168.1.100/hook", "192.168.1.100"),
            ("http://169.254.169.254/latest/meta-data/", "169.254.169.254"),
            ("http://224.0.0.1/hook", "224.0.0.1"),
            ("http://[::1]/hook", "::1"),
            ("http://[fe80::1]/hook", "fe80::1"),
            ("http://[fc00::1]/hook", "fc00::1"),
        ],
    )
    def test_blocked_ips_rejected_422(
        self, client: TestClient, blocked_ip_url: str, mocked_ip: str
    ) -> None:
        payload = {"url": blocked_ip_url, "payload": {"event": "test"}}
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (mocked_ip, 80))]
            resp = client.post(
                "/v1/webhooks/test",
                json=payload,
                headers={"X-API-Key": TEST_API_KEY},
            )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert resp.json()["error"] == "VALIDATION_ERROR"
        assert "private, link-local, or loopback" in resp.json()["message"]

    def test_dns_resolution_failure_returns_422(self, client: TestClient) -> None:
        payload = {"url": "https://nonexistent-domain-xyz-12345.com/hook", "payload": {"test": True}}
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name or service not known")):
            resp = client.post(
                "/v1/webhooks/test",
                json=payload,
                headers={"X-API-Key": TEST_API_KEY},
            )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert resp.json()["error"] == "VALIDATION_ERROR"

    def test_oversized_payload_rejected_422(self, client: TestClient) -> None:
        # Create payload > 10 KB
        large_payload = {"data": "x" * (11 * 1024)}
        body = {"url": "https://hooks.example.com/alerts", "payload": large_payload}
        resp = client.post(
            "/v1/webhooks/test",
            json=body,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert resp.json()["error"] == "VALIDATION_ERROR"
        assert "10 KB" in resp.json()["message"]


class TestWebhookDispatchExecution:
    """Test outbound dispatch, response wrapping, and zero-persistence guarantee."""

    @patch("mlsentry.api.routes.webhooks.requests.post")
    @patch("socket.getaddrinfo")
    def test_valid_public_webhook_200_ok(
        self,
        mock_dns: MagicMock,
        mock_post: MagicMock,
        client: TestClient,
        db_session,
    ) -> None:
        # Public IP simulation
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        payload = {"url": "https://hooks.example.com/alerts", "payload": {"event": "alert_test", "severity": "CRITICAL"}}
        resp = client.post(
            "/v1/webhooks/test",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["fired"] is True
        assert data["url"] == "https://hooks.example.com/alerts"
        assert data["response_status"] == 200

        mock_post.assert_called_once()
        # Verify zero DB persistence
        assert db_session.query(AlertRecord).count() == 0

    @patch("mlsentry.api.routes.webhooks.requests.post")
    @patch("socket.getaddrinfo")
    def test_target_non_200_propagates_response_status(
        self,
        mock_dns: MagicMock,
        mock_post: MagicMock,
        client: TestClient,
    ) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_post.return_value = mock_resp

        payload = {"url": "https://hooks.example.com/missing-hook", "payload": {"event": "test"}}
        resp = client.post(
            "/v1/webhooks/test",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["fired"] is True
        assert data["response_status"] == 404

    @patch("mlsentry.api.routes.webhooks.requests.post")
    @patch("socket.getaddrinfo")
    def test_destination_timeout_returns_503(
        self,
        mock_dns: MagicMock,
        mock_post: MagicMock,
        client: TestClient,
    ) -> None:
        mock_dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        mock_post.side_effect = requests.Timeout("Connection timed out after 5.0s")

        payload = {"url": "https://hooks.example.com/slow-endpoint", "payload": {"event": "test"}}
        resp = client.post(
            "/v1/webhooks/test",
            json=payload,
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert resp.json()["error"] == "SERVICE_UNAVAILABLE"
