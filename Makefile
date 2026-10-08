PY ?= python3
RAW ?= data/raw
DB ?= data/warehouse/supply_chain.duckdb
DQ_REPORT ?= artifacts/dq_report.json

.DEFAULT_GOAL := help
.PHONY: help install dq pipeline data train run test test-unit test-integration lint format \
        docker-up docker-down docker-logs docker-test clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

dq: ## Run standalone data-quality checks on the raw CSVs
	$(PY) -m dq --input $(RAW) --output $(DQ_REPORT)

pipeline: ## Build the DuckDB warehouse (raw -> curated -> serving)
	$(PY) -m pipeline --input $(RAW) --db $(DB)

data: dq pipeline ## DQ checks + pipeline

train: data ## Retrain the delay model (writes artifacts/model/)
	$(PY) -m ml.train --db $(DB) --out artifacts/model

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
	docker compose --profile test run --rm integration-tests; status=$$?; \
	docker compose --profile test down -v; exit $$status

clean: ## Remove local build outputs
	rm -rf data/warehouse $(DQ_REPORT) .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
