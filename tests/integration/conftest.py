"""Integration fixtures: a REAL running service exercised over HTTP.

Two modes:
* ``SCI_BASE_URL`` is set (e.g. by ``make test-docker`` / docker compose) -> test that service.
* otherwise -> build the warehouse + DQ report from the raw CSVs into a temp dir and start
  ``python -m app`` (uvicorn) as a subprocess on a free port. Same code path as the
  container, no Docker required, so it also runs in plain CI runners.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest

from dq.runner import build_report, write_report
from pipeline.build import build_warehouse
from tests.conftest import MODEL_DIR, RAW_DIR, ROOT

STARTUP_TIMEOUT_S = 60


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_ready(url: str, proc: subprocess.Popen | None = None) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    last_error: Exception | str | None = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"API process exited early with code {proc.returncode}")
        try:
            resp = httpx.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Service at {url} not ready after {STARTUP_TIMEOUT_S}s: {last_error}")


@pytest.fixture(scope="session")
def base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    external = os.environ.get("SCI_BASE_URL")
    if external:
        url = external.rstrip("/")
        _wait_until_ready(url)
        yield url
        return

    workdir = tmp_path_factory.mktemp("stack")
    db_path, report_path = workdir / "supply_chain.duckdb", workdir / "dq_report.json"
    write_report(build_report(RAW_DIR), report_path)
    build_warehouse(RAW_DIR, db_path)

    port = _free_port()
    env = {
        **os.environ,
        "SCI_HOST": "127.0.0.1",
        "SCI_PORT": str(port),
        "SCI_DB_PATH": str(db_path),
        "SCI_DQ_REPORT_PATH": str(report_path),
        "SCI_MODEL_DIR": str(MODEL_DIR),
        "SCI_ENVIRONMENT": "integration-test",
    }
    log_file = (workdir / "api.log").open("w")
    proc = subprocess.Popen([sys.executable, "-m", "app"], cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_ready(url, proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()


@pytest.fixture(scope="session")
def client(base_url: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=base_url, timeout=10) as c:
        yield c
