#!/bin/sh
# Container bootstrap for the one-shot "pipeline" service:
#   1. standalone DQ checks against the raw CSVs  -> JSON report
#   2. layered ingestion into DuckDB (atomic, idempotent rebuild)
set -eu

RAW_DIR="${SCI_RAW_DIR:-/app/data/raw}"

echo "[bootstrap] running data-quality checks on ${RAW_DIR}"
python -m dq --input "${RAW_DIR}" --output "${SCI_DQ_REPORT_PATH}"

echo "[bootstrap] building warehouse at ${SCI_DB_PATH}"
python -m pipeline --input "${RAW_DIR}" --db "${SCI_DB_PATH}"

echo "[bootstrap] done"
