# syntax=docker/dockerfile:1.7
# ───────────────────────────────────────────────────────────────────────────
# Player Monetization Intelligence Platform
# Single multi-stage image serving BOTH the FastAPI backend (port 8000) and
# the Streamlit dashboard (port 8501). The docker-compose file selects which
# entry-point runs by overriding CMD per service.
# ───────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Builder: install dependencies into an isolated venv
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder

# uv is Astral's fast Python package manager (same tool used locally).
# We pin to a specific version so builds stay reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

# Build-time system deps (compilers, headers). Discarded in the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests FIRST so this layer caches across code changes.
COPY pyproject.toml uv.lock ./

# --no-dev: skip dev-only deps (pytest, ruff, ...). Keeps the venv lean.
# --frozen: fail if uv.lock disagrees with pyproject.toml.
RUN uv sync --frozen --no-dev


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Runtime: minimal base + venv + application code
# ═══════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime

# Runtime OS deps:
#   curl — used by Docker HEALTHCHECK
#   ca-certificates — HTTPS to MLflow, Gemini, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Non-root user for defense-in-depth.
ARG APP_USER=app
ARG APP_UID=10001
RUN useradd --create-home --shell /bin/bash --uid ${APP_UID} ${APP_USER}

WORKDIR /app

# Copy the built venv from the builder stage.
COPY --from=builder --chown=${APP_USER}:${APP_USER} /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code. `.dockerignore` filters out notebooks, data/, mlruns/, etc.
COPY --chown=${APP_USER}:${APP_USER} src/       ./src/
COPY --chown=${APP_USER}:${APP_USER} dashboard/ ./dashboard/
COPY --chown=${APP_USER}:${APP_USER} scripts/   ./scripts/
COPY --chown=${APP_USER}:${APP_USER} prompts/   ./prompts/
COPY --chown=${APP_USER}:${APP_USER} knowledge/ ./knowledge/

# Model artifacts and calibration plot — baked into the image so the container
# runs standalone. In production these would come from a volume/S3 instead.
COPY --chown=${APP_USER}:${APP_USER} saved_models/ ./saved_models/

USER ${APP_USER}

# Default port for the FastAPI service. Streamlit uses 8501 (see compose).
EXPOSE 8000 8501

# Health check hits FastAPI's /healthz — Streamlit containers override this.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default CMD launches the FastAPI service.
# docker-compose overrides this for the Streamlit container.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
