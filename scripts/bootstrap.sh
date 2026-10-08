#!/bin/sh
# Container bootstrap for the one-shot "pipeline" service:
#   1. standalone DQ checks against the raw CSVs -> JSON report
#      + regression gate vs dq/baseline.json (disable with SCI_DQ_GATE=off)
#   2. layered ingestion into DuckDB: build -> data-contract validation -> atomic swap
# Any non-zero exit stops the stack before bad data reaches the API.
set -eu

RAW_DIR="${SCI_RAW_DIR:-/app/data/raw}"
BASELINE="${SCI_DQ_BASELINE:-/app/dq/baseline.json}"

GATE_ARGS=""
if [ "${SCI_DQ_GATE:-on}" != "off" ] && [ -f "${BASELINE}" ]; then
  GATE_ARGS="--baseline ${BASELINE}"
fi

echo "[bootstrap] data-quality checks on ${RAW_DIR} ${GATE_ARGS:+(gate: ${BASELINE})}"
# shellcheck disable=SC2086
python -m dq --input "${RAW_DIR}" --output "${SCI_DQ_REPORT_PATH}" ${GATE_ARGS}

echo "[bootstrap] building warehouse at ${SCI_DB_PATH}"
python -m pipeline --input "${RAW_DIR}" --db "${SCI_DB_PATH}"

echo "[bootstrap] done"
