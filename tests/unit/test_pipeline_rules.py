"""Pipeline cleaning rules on a tiny hand-built dataset (in-process DuckDB, ~100 ms)."""

from __future__ import annotations

import duckdb
import pytest

from pipeline.build import build_warehouse
from pipeline.validate import RULES
from tests.unit.test_dq_checks import PORTS, data, event, shipment


@pytest.fixture
def warehouse(tmp_path):
    ports = PORTS.copy()
    raw = data(
        shipments=[
            shipment(shipment_id="SHP-00001"),  # clean, 36h late
            shipment(shipment_id="SHP-00001"),  # exact duplicate -> dropped
            shipment(shipment_id="SHP-00002", cargo_type="  electronics "),  # normalised
            shipment(shipment_id="SHP-00002", cargo_type=""),  # conflicting dup, less complete
            shipment(shipment_id="SHP-00003", origin_port="XXTST"),  # unknown port -> quarantine
            shipment(shipment_id="SHP-00004", weight_tons="-5", container_count="996"),
            shipment(shipment_id="SHP-00005", status="Complete", actual_arrival="", actual_departure=""),
            shipment(shipment_id="SHP-00006", actual_departure="2024-12-25 00:00:00"),  # before booking
            shipment(shipment_id="SHP-00007", status="CANCELLED", actual_arrival="", actual_departure=""),
            shipment(shipment_id="SHP-00008", status="delivered", actual_arrival="2025-02-08 20:00:00"),
        ],
        events=[
            event(),
            event(),
            event(event_id="EVT-000002", vessel_id="VSL-9999"),
            event(event_id="EVT-000003", event_type=""),
        ],
        ports=ports,
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw.ports.to_csv(raw_dir / "ports.csv", index=False)
    raw.shipments.to_csv(raw_dir / "shipments.csv", index=False)
    raw.port_events.to_csv(raw_dir / "port_events.csv", index=False)
    db = tmp_path / "wh.duckdb"
    result = build_warehouse(raw_dir, db, max_quarantine_pct=50)  # tiny dataset: 2 of 10 rows
    con = duckdb.connect(str(db), read_only=True)
    yield con, result
    con.close()


def row(con, shipment_id):
    cur = con.execute("SELECT * FROM curated.shipments WHERE shipment_id = ?", [shipment_id])
    cols = [d[0] for d in cur.description]
    values = cur.fetchone()
    return dict(zip(cols, values, strict=True)) if values else None


def test_counts_reconcile(warehouse):
    _, result = warehouse
    rc = result.row_counts
    assert rc["raw.shipments"] == 10
    assert rc["curated.shipments"] == 7
    assert rc["quarantine.shipments"] == 2  # conflicting duplicate + unknown port
    assert rc["curated.port_events"] == 1  # replay dropped, sentinel + missing type quarantined
    assert all(v.passed for v in result.validation)


def test_derived_fields(warehouse):
    con, _ = warehouse
    r = row(con, "SHP-00001")
    assert r["actual_delay_hours"] == 36.0
    assert r["on_time_flag"] is False
    assert r["status"] == "DELAYED"
    assert r["route_key"] == "CNSHA → NLRTM"
    assert r["transit_days_planned"] == 31.0
    assert r["transit_days_actual"] == pytest.approx(32 + 10 / 24)


def test_conflicting_duplicate_keeps_most_complete_version(warehouse):
    con, _ = warehouse
    assert row(con, "SHP-00002")["cargo_type"] == "Electronics"
    reason = con.execute("SELECT quarantine_reason FROM quarantine.shipments WHERE shipment_id='SHP-00002'").fetchone()
    assert reason == ("conflicting_duplicate_id",)


def test_fixes_and_flags(warehouse):
    con, _ = warehouse
    r4 = row(con, "SHP-00004")
    assert (r4["weight_tons"], r4["container_count"]) == (None, None)
    assert {"weight_non_positive", "container_count_outlier"} <= set(r4["dq_flags"])

    r5 = row(con, "SHP-00005")
    assert (r5["status"], r5["actual_delay_hours"], r5["on_time_flag"]) == ("UNKNOWN", None, None)

    r6 = row(con, "SHP-00006")
    assert r6["actual_departure"] is None
    assert r6["actual_delay_hours"] == 36.0
    assert "actual_departure_before_booking" in r6["dq_flags"]

    assert row(con, "SHP-00007")["status"] == "CANCELLED"
    r8 = row(con, "SHP-00008")
    assert (r8["status"], r8["on_time_flag"]) == ("DELIVERED", True)  # 'delivered', 12h late
    assert row(con, "SHP-00003") is None


def test_ports_fix_and_lineage(warehouse):
    con, _ = warehouse
    assert con.execute("SELECT country FROM curated.ports WHERE port_code='BEANR'").fetchone() == ("Belgium",)
    assert con.execute("SELECT _source_row FROM curated.shipments WHERE shipment_id='SHP-00001'").fetchone() == (1,)


def test_quarantine_guard_rejects_build(tmp_path, warehouse):
    from pipeline.validate import PipelineValidationError

    raw_dir = tmp_path / "raw"
    with pytest.raises(PipelineValidationError, match="quarantine_ratio"):
        build_warehouse(raw_dir, tmp_path / "other.duckdb")  # default 10% guard; this data quarantines 20%
    assert not (tmp_path / "other.duckdb").exists()


def test_every_validation_rule_is_named_and_described():
    names = [name for name, _, _ in RULES]
    assert len(names) == len(set(names))
    assert all(desc for _, desc, _ in RULES)
