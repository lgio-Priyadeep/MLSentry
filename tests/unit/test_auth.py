"""Unit tests for API key authentication middleware and constant-time verification."""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from mlsentry.api.errors import MLSentryAPIException, mlsentry_api_exception_handler
from mlsentry.api.middleware import RequestIDMiddleware, get_api_key_dependency


def create_auth_test_app(expected_key: str = "secret-test-key-12345") -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(MLSentryAPIException, mlsentry_api_exception_handler)

    auth_dep = get_api_key_dependency(expected_key)

    @app.get("/public")
    async def public_endpoint():
        return {"status": "public_ok"}

    @app.get("/protected", dependencies=[Depends(auth_dep)])
    async def protected_endpoint():
        return {"status": "protected_ok"}

    return app


class TestAuthDependency:
    """Test suite for X-API-Key verification."""

    def test_missing_api_key_returns_401_unauthorized(self) -> None:
        app = create_auth_test_app()
        client = TestClient(app)

        response = client.get("/protected")
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "UNAUTHORIZED"
        assert "Missing" in data["message"]
        assert "request_id" in data

    def test_invalid_api_key_returns_401_unauthorized(self) -> None:
        app = create_auth_test_app()
        client = TestClient(app)

        response = client.get("/protected", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "UNAUTHORIZED"
        assert "Invalid" in data["message"]
        assert "request_id" in data

    def test_valid_api_key_returns_200_ok(self) -> None:
        app = create_auth_test_app()
        client = TestClient(app)

        response = client.get(
            "/protected", headers={"X-API-Key": "secret-test-key-12345"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "protected_ok"}

    def test_public_route_accessible_without_auth_header(self) -> None:
        app = create_auth_test_app()
        client = TestClient(app)

        response = client.get("/public")
        assert response.status_code == 200
        assert response.json() == {"status": "public_ok"}
