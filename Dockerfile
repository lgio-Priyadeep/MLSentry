# ==============================================================================
# Multi-stage Dockerfile for MLSentry Production Service
# Target: Python 3.11 Debian Bookworm Slim with CPU-only runtime
# ==============================================================================

# Stage 1: Build virtual environment and install compiled dependencies
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --find-links https://download.pytorch.org/whl/cpu/torch_stable.html -r /tmp/requirements.lock

# ==============================================================================
# Stage 2: Final lightweight execution runner
# ==============================================================================
FROM python:3.11-slim-bookworm AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME="/tmp/huggingface" \
    TRANSFORMERS_CACHE="/tmp/huggingface" \
    PORT=8000

# Install runtime PostgreSQL client library and healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user and group (MLSentry security constraint)
RUN groupadd -g 10001 mlsentry && \
    useradd -u 10001 -g mlsentry -s /bin/bash -m mlsentry

# Create HuggingFace cache directory with write permissions for non-root user
RUN mkdir -p /tmp/huggingface && chown -R mlsentry:mlsentry /tmp/huggingface

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set up application workspace
WORKDIR /app
COPY --chown=mlsentry:mlsentry . /app

# Ensure entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER mlsentry

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
