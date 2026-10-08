"""The pipeline must be idempotent: running it twice yields identical tables."""

from __future__ import annotations

import duckdb

from pipeline.build import build_warehouse
from tests.conftest import RAW_DIR

FINGERPRINT_SQL = """
SELECT count(*) AS n,
       md5(string_agg(shipment_id || coalesce(CAST(actual_delay_hours AS VARCHAR), '') || status, ','
                      ORDER BY shipment_id)) AS digest
FROM curated.shipments
"""


def _fingerprint(db_path):
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(FINGERPRINT_SQL).fetchone()


def test_pipeline_is_idempotent(tmp_path):
    db = tmp_path / "wh.duckdb"
    first = build_warehouse(RAW_DIR, db)
    fp1 = _fingerprint(db)
    second = build_warehouse(RAW_DIR, db)
    fp2 = _fingerprint(db)
    assert first.row_counts == second.row_counts
    assert fp1 == fp2
    assert not list(tmp_path.glob(".*.tmp")), "temporary build files must be cleaned up"


def test_curated_layer_invariants(tmp_path):
    db = tmp_path / "wh.duckdb"
    result = build_warehouse(RAW_DIR, db)
    with duckdb.connect(str(db), read_only=True) as con:
        dupes = con.execute("SELECT count(*) - count(DISTINCT shipment_id) FROM curated.shipments").fetchone()[0]
        orphans = con.execute(
            "SELECT count(*) FROM curated.shipments s LEFT JOIN curated.ports p ON p.port_code = s.origin_port "
            "WHERE p.port_code IS NULL"
        ).fetchone()[0]
        bad_flags = con.execute(
            "SELECT count(*) FROM curated.shipments WHERE on_time_flag IS DISTINCT FROM (actual_delay_hours <= 24)"
        ).fetchone()[0]
    assert dupes == 0
    assert orphans == 0
    assert bad_flags == 0
    # every raw row (minus exact duplicates) ends up in exactly one of curated / quarantine
    assert (
        result.row_counts["curated.shipments"] + result.row_counts["quarantine.shipments"]
        <= result.row_counts["raw.shipments"]
    )
