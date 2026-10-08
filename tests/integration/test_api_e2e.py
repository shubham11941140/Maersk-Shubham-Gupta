"""End-to-end tests over real HTTP against real data."""

from __future__ import annotations

from datetime import datetime

import pytest

KNOWN_SHIPMENT = "SHP-00421"


def error_code(resp) -> str:
    return resp.json()["error"]["code"]


class TestHealth:
    def test_ready_with_all_dependencies(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert {k: v["status"] for k, v in body["checks"].items()} == {
            "database": "ok",
            "model": "ok",
            "data_quality_report": "ok",
        }
        assert body["version"]

    def test_liveness(self, client):
        assert client.get("/health/live").status_code == 200


class TestShipmentLookup:
    def test_successful_lookup(self, client):
        resp = client.get(f"/shipments/{KNOWN_SHIPMENT}")
        assert resp.status_code == 200
        s = resp.json()
        assert s["shipment_id"] == KNOWN_SHIPMENT
        assert s["route_key"] == f"{s['origin_port']} → {s['destination_port']}"
        assert s["origin"]["port_code"] == s["origin_port"]
        # derived fields are consistent with the raw timestamps
        if s["actual_arrival"]:
            delta = datetime.fromisoformat(s["actual_arrival"]) - datetime.fromisoformat(s["planned_arrival"])
            assert s["actual_delay_hours"] == pytest.approx(delta.total_seconds() / 3600)
            assert s["on_time_flag"] is (s["actual_delay_hours"] <= 24)

    def test_unknown_id_returns_404(self, client):
        resp = client.get("/shipments/SHP-99999")
        assert resp.status_code == 404
        assert error_code(resp) == "SHIPMENT_NOT_FOUND"
        assert resp.json()["error"]["request_id"] == resp.headers["x-request-id"]

    def test_quarantined_shipment_is_not_served(self, client):
        report = client.get("/data-quality/report", params={"dataset": "shipments"}).json()
        check = next(c for c in report["checks"] if c["check_name"] == "shipments.unknown_port_code")
        assert check["sample_keys"], "expected the raw data to contain unknown port codes"
        resp = client.get(f"/shipments/{check['sample_keys'][0]}")
        assert resp.status_code == 404


class TestShipmentList:
    def test_filters_and_pagination(self, client):
        params = {"origin": "CNSHA", "status": "DELAYED", "page_size": 5}
        first = client.get("/shipments", params=params).json()
        second = client.get("/shipments", params={**params, "page": 2}).json()
        assert first["pagination"]["total_items"] == second["pagination"]["total_items"] > 5
        ids_1, ids_2 = {s["shipment_id"] for s in first["items"]}, {s["shipment_id"] for s in second["items"]}
        assert len(ids_1) == 5
        assert not ids_1 & ids_2
        for s in first["items"] + second["items"]:
            assert s["origin_port"] == "CNSHA"
            assert s["status"] == "DELAYED"

    def test_date_range_filter(self, client):
        body = client.get(
            "/shipments", params={"date_from": "2025-03-01", "date_to": "2025-03-31", "page_size": 100}
        ).json()
        assert body["pagination"]["total_items"] > 0
        for s in body["items"]:
            assert "2025-03-01" <= s["planned_departure"][:10] <= "2025-03-31"

    def test_invalid_filter_returns_422(self, client):
        resp = client.get("/shipments", params={"status": "TELEPORTED"})
        assert resp.status_code == 422
        assert error_code(resp) == "VALIDATION_ERROR"


class TestRouteStats:
    def test_stats_consistent_with_listing(self, client):
        stats = client.get("/routes/CNSHA/NLRTM/stats").json()
        listing = client.get("/shipments", params={"origin": "CNSHA", "destination": "NLRTM"}).json()
        assert stats["shipment_count"] == listing["pagination"]["total_items"] > 0
        assert 0.0 <= stats["on_time_rate"] <= 1.0
        assert (
            stats["completed_count"] + stats["cancelled_count"] + stats["unknown_outcome_count"]
            == stats["shipment_count"]
        )

    def test_unknown_port_returns_404(self, client):
        resp = client.get("/routes/XXTST/NLRTM/stats")
        assert resp.status_code == 404
        assert error_code(resp) == "PORT_NOT_FOUND"


class TestDataQualityReport:
    def test_report_shape_and_known_issues(self, client):
        resp = client.get("/data-quality/report")
        assert resp.status_code == 200
        report = resp.json()
        assert report["summary"]["checks_run"] >= 12
        required = {"check_name", "rows_affected", "pct_affected", "severity", "action", "status"}
        for check in report["checks"]:
            assert required <= check.keys()
            assert check["severity"] in {"critical", "warning", "info"}
        by_name = {c["check_name"]: c for c in report["checks"]}
        assert by_name["shipments.exact_duplicate_rows"]["rows_affected"] > 0
        assert by_name["shipments.unknown_port_code"]["status"] == "fail"

    def test_filtering(self, client):
        report = client.get("/data-quality/report", params={"severity": "critical", "only_failed": True}).json()
        assert report["checks"]
        assert all(c["severity"] == "critical" and c["status"] == "fail" for c in report["checks"])


class TestPredictDelay:
    def test_predict_existing_shipment(self, client):
        model_version = client.get("/health").json()["checks"]["model"]["detail"]["model_version"]
        resp = client.post("/predict-delay", json={"shipment_id": KNOWN_SHIPMENT})
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["delay_probability"] <= 1.0
        assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
        assert body["model_version"] == model_version
        assert body["predicted_delayed"] is (body["delay_probability"] >= body["decision_threshold"])

    def test_predict_new_booking(self, client):
        payload = {
            "booking": {
                "origin_port": "CNSHA",
                "destination_port": "NLRTM",
                "cargo_type": "Electronics",
                "container_count": 20,
                "weight_tons": 950.5,
                "booking_date": "2025-09-01T08:00:00",
                "planned_departure": "2025-09-10T08:00:00",
                "planned_arrival": "2025-10-12T08:00:00",
            }
        }
        resp = client.post("/predict-delay", json=payload)
        assert resp.status_code == 200
        assert resp.json()["shipment_id"] is None

    def test_prediction_is_deterministic(self, client):
        a = client.post("/predict-delay", json={"shipment_id": KNOWN_SHIPMENT}).json()
        b = client.post("/predict-delay", json={"shipment_id": KNOWN_SHIPMENT}).json()
        assert a["delay_probability"] == b["delay_probability"]

    def test_unknown_shipment_404(self, client):
        resp = client.post("/predict-delay", json={"shipment_id": "SHP-99999"})
        assert resp.status_code == 404

    def test_invalid_payload_422(self, client):
        resp = client.post("/predict-delay", json={"booking": {"origin_port": "CNSHA"}})
        assert resp.status_code == 422
        assert error_code(resp) == "VALIDATION_ERROR"


def test_request_id_propagation(client):
    resp = client.get("/health/live", headers={"X-Request-ID": "e2e-trace-1"})
    assert resp.headers["x-request-id"] == "e2e-trace-1"
