"""Each DQ check is exercised against a tiny hand-built dataset."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dq import checks
from dq.checks import REGISTRY
from dq.models import Action, Check, RawData, Severity
from dq.runner import build_report, run_check, write_report

PORTS = pd.DataFrame(
    {
        "port_code": ["CNSHA", "NLRTM", "BEANR"],
        "port_name": ["Shanghai", "Rotterdam", "Antwerp"],
        "country": ["China", "Netherlands", ""],
        "region": ["APAC", "EUR", "EUR"],
        "timezone": ["UTC+8", "UTC+1", "UTC+1"],
        "avg_congestion_score": ["0.85", "0.80", "0.70"],
    }
)


def shipment(**overrides: str) -> dict[str, str]:
    row = {
        "shipment_id": "SHP-00001",
        "customer_id": "CUST-1001",
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "vessel_id": "VSL-0001",
        "booking_date": "2025-01-01 08:00:00",
        "planned_departure": "2025-01-08 08:00:00",
        "actual_departure": "2025-01-08 10:00:00",
        "planned_arrival": "2025-02-08 08:00:00",
        "actual_arrival": "2025-02-09 20:00:00",  # 36h late -> consistent with DELAYED
        "container_count": "12",
        "cargo_type": "Electronics",
        "weight_tons": "950.5",
        "status": "DELAYED",
    }
    row.update(overrides)
    return row


def event(**overrides: str) -> dict[str, str]:
    row = {
        "event_id": "EVT-000001",
        "event_timestamp": "2025-01-01 00:00:00",
        "port_code": "CNSHA",
        "vessel_id": "VSL-0001",
        "event_type": "ARRIVAL",
        "delay_minutes": "0",
        "notes": "Normal operations",
    }
    row.update(overrides)
    return row


def data(shipments=(), events=(), ports=PORTS) -> RawData:
    ship_df = pd.DataFrame(list(shipments) or [shipment()])
    ev_df = pd.DataFrame(list(events) or [event()])
    return RawData(ports=ports.copy(), shipments=ship_df, port_events=ev_df)


def affected(fn, raw: RawData) -> list[int]:
    return list(fn(raw).fillna(False).astype(bool).to_numpy().nonzero()[0])


def test_registry_has_unique_names_and_covers_all_datasets():
    names = [c.name for c in REGISTRY]
    assert len(names) == len(set(names))
    assert {c.dataset for c in REGISTRY} == {"ports", "shipments", "port_events"}
    assert len(REGISTRY) >= 12


def test_clean_row_trips_no_shipment_checks():
    raw = data()
    for check in REGISTRY:
        if check.dataset == "shipments":
            assert affected(check.fn, raw) == [], check.name


def test_exact_duplicates_flag_only_the_copy():
    raw = data([shipment(), shipment()])
    assert affected(checks.exact_duplicate_shipments, raw) == [1]
    assert affected(checks.conflicting_duplicate_ids, raw) == []


def test_conflicting_duplicate_ids():
    raw = data([shipment(), shipment(weight_tons="951.0"), shipment(shipment_id="SHP-00002")])
    assert affected(checks.conflicting_duplicate_ids, raw) == [0, 1]


def test_unknown_port_code():
    raw = data([shipment(), shipment(origin_port="XXTST"), shipment(destination_port="ZZZZZ")])
    assert affected(checks.unknown_port_code, raw) == [1, 2]


def test_actual_departure_before_booking():
    raw = data([shipment(), shipment(actual_departure="2024-12-30 00:00:00")])
    assert affected(checks.actual_departure_before_booking, raw) == [1]


def test_planned_sequence_invalid():
    raw = data([shipment(), shipment(planned_arrival="2025-01-07 00:00:00")])
    assert affected(checks.planned_sequence_invalid, raw) == [1]


def test_unparseable_timestamp_ignores_blanks():
    raw = data([shipment(actual_arrival=""), shipment(booking_date="01/02/2025")])
    assert affected(checks.unparseable_shipment_timestamp, raw) == [1]


@pytest.mark.parametrize(
    ("status", "non_canonical", "missing"),
    [
        ("DELAYED", False, False),
        ("delivered", True, False),
        ("Complete", True, False),
        ("N/A", False, True),
        ("", False, True),
    ],
)
def test_status_checks(status, non_canonical, missing):
    raw = data([shipment(status=status)])
    assert bool(affected(checks.status_non_canonical, raw)) is non_canonical
    assert bool(affected(checks.status_missing, raw)) is missing


def test_status_actuals_mismatch():
    raw = data(
        [
            shipment(),  # delayed + arrived -> consistent
            shipment(status="COMPLETED", actual_arrival="", actual_departure=""),  # claims done, never arrived
            shipment(status="CANCELLED", actual_arrival="", actual_departure=""),  # consistent
            shipment(status="CANCELLED"),  # cancelled but arrived
        ]
    )
    assert affected(checks.status_actuals_mismatch, raw) == [1, 3]


def test_cargo_type_checks():
    raw = data(
        [shipment(), shipment(cargo_type="FURNITURE"), shipment(cargo_type=" Chemicals"), shipment(cargo_type="")]
    )
    assert affected(checks.cargo_type_non_canonical, raw) == [1, 2]
    assert affected(checks.cargo_type_missing, raw) == [3]


def test_weight_checks():
    raw = data([shipment(), shipment(weight_tons=""), shipment(weight_tons="-10.5"), shipment(weight_tons="0")])
    assert affected(checks.weight_missing, raw) == [1]
    assert affected(checks.weight_non_positive, raw) == [2, 3]


def test_container_count_checks():
    raw = data(
        [shipment(), shipment(container_count="0"), shipment(container_count="996"), shipment(container_count="100")]
    )
    assert affected(checks.container_count_zero, raw) == [1]
    assert affected(checks.container_count_outlier, raw) == [2]


def test_ports_missing_field():
    assert affected(checks.ports_missing_field, data()) == [2]


def test_event_checks():
    raw = data(
        events=[
            event(),
            event(),  # exact replay
            event(event_id="EVT-000002", port_code="NLRTM"),
            event(event_id="EVT-000002", port_code="CNSHA", vessel_id="VSL-0002"),  # id collision
            event(event_id="EVT-000003", event_type=""),
            event(event_id="EVT-000004", vessel_id="VSL-9999"),
            event(event_id="EVT-000005", event_type="DELAYED", delay_minutes="0"),
            event(event_id="EVT-000006", event_type="TELEPORTED"),
        ]
    )
    assert affected(checks.exact_duplicate_events, raw) == [1]
    assert affected(checks.duplicate_event_id, raw) == [2, 3]
    assert affected(checks.missing_event_type, raw) == [4]
    assert affected(checks.sentinel_vessel_id, raw) == [5]
    assert affected(checks.delayed_event_without_delay, raw) == [6]
    assert affected(checks.invalid_event_type, raw) == [7]


def test_run_check_isolates_failures():
    def boom(_):
        raise KeyError("missing_column")

    broken = Check("x.broken", "shipments", "shipment_id", "d", "det", Severity.INFO, Action.NONE, "r", boom)
    result = run_check(broken, data())
    assert result.status == "error"
    assert "KeyError" in result.error


def test_result_percentages_and_samples():
    check = next(c for c in REGISTRY if c.name == "shipments.unknown_port_code")
    raw = data([shipment(), shipment(shipment_id="SHP-00009", origin_port="XXTST")])
    result = run_check(check, raw)
    assert (result.rows_affected, result.total_rows, result.pct_affected) == (1, 2, 50.0)
    assert result.sample_keys == ["SHP-00009"]
    assert result.to_dict()["status"] == "fail"


def test_build_and_write_report_roundtrip(tmp_path):
    raw = data([shipment(), shipment(container_count="0")])
    raw.ports.to_csv(tmp_path / "ports.csv", index=False)
    raw.shipments.to_csv(tmp_path / "shipments.csv", index=False)
    raw.port_events.to_csv(tmp_path / "port_events.csv", index=False)

    report = build_report(tmp_path)
    out = tmp_path / "out" / "report.json"
    write_report(report, out)

    loaded = json.loads(out.read_text())
    assert loaded["summary"]["checks_run"] == len(REGISTRY)
    by_name = {c["check_name"]: c for c in loaded["checks"]}
    assert by_name["shipments.container_count_zero"]["rows_affected"] == 1
    assert set(by_name["shipments.container_count_zero"]) >= {
        "check_name",
        "rows_affected",
        "pct_affected",
        "severity",
        "action",
        "rationale",
    }


def test_missing_input_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path)


def test_every_check_documents_how_it_was_detected():
    for check in REGISTRY:
        assert check.detection.strip(), check.name
        assert check.rationale.strip(), check.name


def test_status_delay_contradiction():
    raw = data(
        [
            shipment(),  # DELAYED, 36h late: consistent
            shipment(status="DELAYED", actual_arrival="2025-02-08 10:00:00"),  # DELAYED but 2h late
            shipment(status="DELIVERED"),  # DELIVERED but 36h late
            shipment(status="DELIVERED", actual_arrival="2025-02-08 10:00:00"),  # consistent
        ]
    )
    assert affected(checks.status_delay_contradiction, raw) == [1, 2]


def test_planned_transit_outlier_needs_enough_route_samples():
    normal = [shipment(shipment_id=f"SHP-0000{i}") for i in range(5)]  # 31-day transit
    outlier = shipment(shipment_id="SHP-00009", planned_arrival="2025-04-30 08:00:00")  # ~112 days
    assert affected(checks.planned_transit_outlier_for_route, data([*normal, outlier])) == [5]
    assert affected(checks.planned_transit_outlier_for_route, data([normal[0], outlier])) == []


def test_port_timezone_format():
    ports = PORTS.copy()
    ports.loc[1, "timezone"] = "CET"
    assert affected(checks.invalid_timezone_format, data(ports=ports)) == [1]


def test_events_out_of_order_and_notes_contradiction():
    raw = data(
        events=[
            event(event_timestamp="2025-01-02 00:00:00"),
            event(event_id="EVT-000002", event_timestamp="2025-01-01 00:00:00", delay_minutes="15"),
        ]
    )
    assert affected(checks.events_out_of_order, raw) == [1]
    assert affected(checks.notes_contradict_delay, raw) == [1]


def test_report_contains_profile_and_detection(tmp_path):
    raw = data([shipment(), shipment(weight_tons="-1")])
    raw.ports.to_csv(tmp_path / "ports.csv", index=False)
    raw.shipments.to_csv(tmp_path / "shipments.csv", index=False)
    raw.port_events.to_csv(tmp_path / "port_events.csv", index=False)
    report = build_report(tmp_path)
    weight = report["profile"]["shipments"]["column_profiles"]["weight_tons"]
    assert weight["inferred_type"] == "numeric"
    assert weight["min"] == -1.0
    assert all(c["detection"] for c in report["checks"])
    assert report["summary"]["rows_affected_by_dataset"]["shipments"] >= 1
    assert build_report(tmp_path, include_profile=False).get("profile") is None
