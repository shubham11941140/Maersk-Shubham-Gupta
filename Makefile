PY ?= python
RAW ?= data/raw
DB ?= data/warehouse/supply_chain.duckdb
DQ_REPORT ?= artifacts/dq_report.json
DQ_BASELINE ?= dq/baseline.json
EXPORT ?= data/export

.DEFAULT_GOAL := help
.PHONY: help install dq dq-baseline dq-docs model-card monitor assistant assistant-demo pipeline pipeline-check export data train run test test-unit test-integration lint format \
        docker-up docker-down docker-logs docker-test ci smoke clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

dq: ## Standalone DQ checks on the raw CSVs + regression gate vs dq/baseline.json
	$(PY) -m dq --input $(RAW) --output $(DQ_REPORT) --baseline $(DQ_BASELINE)

dq-baseline: ## Accept the current DQ results as the new baseline (review the diff!)
	$(PY) -m dq --input $(RAW) --output $(DQ_REPORT) --update-baseline $(DQ_BASELINE)

dq-docs: dq ## Regenerate docs/DATA_QUALITY.md from the latest report
	$(PY) -m dq.render $(DQ_REPORT) > docs/DATA_QUALITY.md

pipeline: ## Build the DuckDB warehouse (raw -> curated/quarantine -> serving, validated, atomic)
	$(PY) -m pipeline --input $(RAW) --db $(DB)

pipeline-check: ## Prove idempotency: build twice, fingerprints must match
	$(PY) -m pipeline --input $(RAW) --db $(DB) > /dev/null
	@a=$$($(PY) -m pipeline --db $(DB) --fingerprint-only); \
	$(PY) -m pipeline --input $(RAW) --db $(DB) > /dev/null; \
	b=$$($(PY) -m pipeline --db $(DB) --fingerprint-only); \
	echo "run 1: $$a"; echo "run 2: $$b"; test "$$a" = "$$b" && echo "idempotent ✔"

export: pipeline ## Export curated / quarantine / serving relations as Parquet
	$(PY) -m pipeline --input $(RAW) --db $(DB) --export-parquet $(EXPORT)

data: dq pipeline ## DQ checks + pipeline

train: data ## Retrain + evaluate the delay model (CV, candidates, permutation test) and refresh the model card
	$(PY) -m ml.train --db $(DB) --out artifacts/model
	$(PY) -m ml.model_card artifacts/model/model_metadata.json > docs/MODEL_CARD.md

model-card: ## Regenerate docs/MODEL_CARD.md from the committed artefact
	$(PY) -m ml.model_card artifacts/model/model_metadata.json > docs/MODEL_CARD.md

monitor: pipeline ## Drift + performance report for the last 90 days of bookings
	$(PY) -m ml.monitor --db $(DB) --model-dir artifacts/model --last-days 90

assistant: ## Interactive GenAI assistant (needs OPENAI_API_KEY and the API on :8000)
	$(PY) -m assistant

assistant-demo: ## Ask the brief's three example questions
	$(PY) -m assistant demo

run: data ## Run the API locally on :8000
	$(PY) -m app

test: ## All tests (unit + integration) with coverage
	$(PY) -m pytest --cov --cov-report=term-missing

test-unit: ## Fast unit tests only (no infrastructure)
	$(PY) -m pytest -m unit

test-integration: ## E2E tests (spins up the API itself unless SCI_BASE_URL is set)
	$(PY) -m pytest -m integration

lint: ## Ruff lint + format check
	ruff check .
	ruff format --check .

format: ## Auto-fix lint + format
	ruff check . --fix
	ruff format .

docker-up: ## Build and start the full stack, wait until healthy
	docker compose up --build -d
	$(PY) scripts/wait_for_http.py http://localhost:8000/health

docker-down: ## Stop the stack and delete its volume
	docker compose --profile test down -v

docker-logs: ## Tail API logs
	docker compose logs -f api

docker-test: ## Run the integration suite inside docker compose against the real containers
	docker compose --profile test build
	docker compose --profile test run --rm -T integration-tests; status=$$?; \
	docker compose --profile test down -v; exit $$status

ci: ## Run the same gates as the CI pipeline (minus Docker) locally
	ruff check .
	ruff format --check .
	$(PY) -m dq --input $(RAW) --output $(DQ_REPORT) --baseline $(DQ_BASELINE)
	$(PY) -m pytest -m unit --cov --cov-fail-under=80
	$(PY) -m pytest -m integration

smoke: ## Post-deploy smoke test (BASE_URL=..., EXPECTED_SHA=... optional)
	$(PY) scripts/smoke_test.py --base-url $${BASE_URL:-http://localhost:8000} $${EXPECTED_SHA:+--expected-sha $$EXPECTED_SHA}

clean: ## Remove local build outputs
	rm -rf data/warehouse $(EXPORT) $(DQ_REPORT) .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
