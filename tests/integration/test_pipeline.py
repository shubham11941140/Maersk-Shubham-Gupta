"""Pipeline on the real raw files: idempotency, contract, failure safety, export."""

from __future__ import annotations

import duckdb
import pytest

from pipeline import validate as validate_module
from pipeline.__main__ import main as pipeline_main
from pipeline.build import build_warehouse, export_parquet, read_fingerprint
from pipeline.validate import PipelineValidationError
from tests.conftest import RAW_DIR


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    db = tmp_path_factory.mktemp("wh") / "wh.duckdb"
    return build_warehouse(RAW_DIR, db)


def test_pipeline_is_idempotent(tmp_path):
    db = tmp_path / "wh.duckdb"
    first = build_warehouse(RAW_DIR, db)
    second = build_warehouse(RAW_DIR, db)
    assert first.run_id != second.run_id
    assert first.fingerprint == second.fingerprint == read_fingerprint(db)
    assert first.row_counts == second.row_counts
    assert not list(tmp_path.glob(".*.tmp")), "temporary build files must be cleaned up"


def test_all_contract_rules_pass_on_real_data(built):
    assert built.validation
    assert all(v.passed for v in built.validation), [v for v in built.validation if not v.passed]


def test_reconciliation_numbers(built):
    rc = built.row_counts
    assert rc["raw.shipments"] == 5028
    assert rc["curated.shipments"] + rc["quarantine.shipments"] == 5028 - 8  # 8 exact duplicates dropped
    assert rc["curated.port_events"] + rc["quarantine.port_events"] == 25000


def test_metadata_is_recorded(built):
    with duckdb.connect(str(built.db_path), read_only=True) as con:
        fp = con.execute("SELECT content_fingerprint FROM meta.pipeline_run").fetchone()[0]
        rules = con.execute("SELECT count(*) FROM meta.validation_results WHERE violations = 0").fetchone()[0]
    assert fp == built.fingerprint
    assert rules == len(built.validation)


def test_failed_validation_keeps_previous_warehouse(tmp_path, monkeypatch):
    db = tmp_path / "wh.duckdb"
    good = build_warehouse(RAW_DIR, db)
    monkeypatch.setattr(validate_module, "RULES", [*validate_module.RULES, ("always_fails", "test", "SELECT 1")])
    with pytest.raises(PipelineValidationError, match="always_fails"):
        build_warehouse(RAW_DIR, db)
    assert read_fingerprint(db) == good.fingerprint, "live warehouse must be untouched"
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_exit_code_on_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_module, "RULES", [("always_fails", "test", "SELECT 1")])
    assert pipeline_main(["--input", str(RAW_DIR), "--db", str(tmp_path / "x.duckdb")]) == 1


def test_parquet_export(built, tmp_path):
    files = export_parquet(built.db_path, tmp_path / "export")
    assert {f.name for f in files} >= {"curated.shipments.parquet", "serving.route_stats.parquet"}
    n = duckdb.sql(f"SELECT count(*) FROM '{tmp_path / 'export' / 'curated.shipments.parquet'}'").fetchone()[0]
    assert n == built.row_counts["curated.shipments"]


def test_serving_quarterly_view(built):
    with duckdb.connect(str(built.db_path), read_only=True) as con:
        total = con.execute("SELECT sum(shipment_count) FROM serving.route_quarterly_stats").fetchone()[0]
        periods = con.execute("SELECT count(DISTINCT period) FROM serving.route_quarterly_stats").fetchone()[0]
    assert total == built.row_counts["curated.shipments"]
    assert periods >= 5
