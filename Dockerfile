# syntax=docker/dockerfile:1.7
# Multi-stage build: dependencies are compiled in a throwaway stage so the
# runtime image carries no build tooling or pip cache.

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG BUILD_SHA=dev
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SCI_BUILD_SHA=${BUILD_SHA} \
    SCI_RAW_DIR=/app/data/raw \
    SCI_DB_PATH=/var/lib/sci/warehouse/supply_chain.duckdb \
    SCI_DQ_REPORT_PATH=/var/lib/sci/reports/dq_report.json \
    SCI_MODEL_DIR=/app/artifacts/model

# Non-root user. /var/lib/sci is pre-created and owned by it so a fresh named
# volume mounted there inherits the right ownership.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /var/lib/sci/warehouse /var/lib/sci/reports /var/lib/sci/logs/assistant \
 && chown -R app:app /var/lib/sci

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY shared ./shared
COPY dq ./dq
COPY pipeline ./pipeline
COPY ml ./ml
COPY app ./app
COPY assistant ./assistant
COPY scripts ./scripts
COPY dq_check.py ./
COPY data/raw ./data/raw
COPY artifacts/model ./artifacts/model

USER app
EXPOSE 8000

# Readiness-based healthcheck: "healthy" means DB, model and DQ report are all loaded.
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["python", "-m", "app"]

# --------------------------------------------------------------------------- #
# Test image: runtime + dev deps + tests. Used by `docker compose --profile test`.
FROM runtime AS test
USER root
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY pyproject.toml ./
COPY tests ./tests
USER app
HEALTHCHECK NONE
CMD ["pytest", "-p", "no:cacheprovider", "-m", "integration", "tests/integration"]
