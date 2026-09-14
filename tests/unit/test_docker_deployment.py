"""Unit tests verifying Dockerfile, docker-compose.yml, and entrypoint configuration correctness."""
from __future__ import annotations

import os
import stat
import pytest
import yaml


def test_dockerfile_exists_and_conforms():
    """Assert Dockerfile exists and enforces non-root execution and healthchecks."""
    assert os.path.isfile("Dockerfile"), "Dockerfile must exist at repository root."
    with open("Dockerfile", "r", encoding="utf-8") as f:
        content = f.read()

    assert "FROM python:3.11-slim-bookworm" in content
    assert "useradd -u 10001 -g mlsentry" in content
    assert "USER mlsentry" in content
    assert "EXPOSE 8000" in content
    assert "HEALTHCHECK" in content
    assert "ENTRYPOINT" in content


def test_docker_compose_syntax_and_services():
    """Assert docker-compose.yml defines valid services with healthchecks and restart policies."""
    assert os.path.isfile("docker-compose.yml"), "docker-compose.yml must exist at repository root."
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data
    services = compose_data["services"]
    assert "db" in services
    assert "app" in services

    db_service = services["db"]
    assert "postgres:16" in db_service["image"]
    assert db_service["restart"] == "on-failure:3"
    assert "healthcheck" in db_service

    app_service = services["app"]
    assert app_service["restart"] == "on-failure:3"
    assert "8000:8000" in app_service["ports"]
    assert "healthcheck" in app_service
    assert "db" in app_service["depends_on"]


def test_entrypoint_script_is_executable_and_valid():
    """Assert entrypoint.sh exists and contains migration and uvicorn startup commands."""
    assert os.path.isfile("entrypoint.sh"), "entrypoint.sh must exist at repository root."
    with open("entrypoint.sh", "r", encoding="utf-8") as f:
        content = f.read()

    assert "#!/usr/bin/env bash" in content
    assert "alembic upgrade head" in content
    assert "exec uvicorn mlsentry.api.main:app" in content


def test_requirements_lock_pinned():
    """Assert requirements.lock contains pinned dependencies for reproducible builds."""
    assert os.path.isfile("requirements.lock"), "requirements.lock must exist at repository root."
    with open("requirements.lock", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    packages = {line.split("==")[0].split(";")[0] for line in lines if "==" in line}
    assert "fastapi" in packages
    assert "pydantic" in packages
    assert "sqlalchemy" in packages
    assert "alembic" in packages
    assert "apscheduler" in packages
    assert "scipy" in packages
    assert "numpy" in packages
    assert "transformers" in packages
    assert "uvicorn" in packages


def test_dockerignore_excludes_unwanted_files():
    """Assert .dockerignore contains exclusions for virtual environments and test folders."""
    assert os.path.isfile(".dockerignore"), ".dockerignore must exist at repository root."
    with open(".dockerignore", "r", encoding="utf-8") as f:
        ignored = [line.strip() for line in f if line.strip()]

    assert ".git" in ignored
    assert ".venv" in ignored
    assert "__pycache__" in ignored
    assert "tests" in ignored
