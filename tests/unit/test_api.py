"""HTTP-contract tests: real FastAPI routing/validation/error handling, fake dependencies.

No database or model files are needed, so these run in milliseconds.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_dq_service, get_prediction_service, get_route_service, get_shipment_service
from app.main import create_app
from app.services.data_quality_service import DataQualityReportService
from app.services.prediction_service import PredictionService
from app.services.route_service import RouteService
from app.services.shipment_service import ShipmentService
from tests.fakes import FakePredictor, FakeShipmentRepository, make_shipment


@pytest.fixture
def repo():
    return FakeShipmentRepository(
        shipments={f"SHP-0000{i}": make_shipment(f"SHP-0000{i}") for i in range(1, 6)},
        stats={"shipment_count": 5, "completed_count": 4, "on_time_count": 3, "avg_delay_hours": 10.0},
    )


@pytest.fixture
def settings(tmp_path):
    report = tmp_path / "dq.json"
    report.write_text(
        json.dumps(
            {
                "summary": {"checks_run": 1},
                "checks": [{"check_name": "x", "dataset": "shipments", "severity": "critical", "status": "fail"}],
            }
        )
    )
    return Settings(
        db_path=tmp_path / "missing.duckdb",
        model_dir=tmp_path / "no-model",
        dq_report_path=report,
        max_page_size=50,
        log_level="WARNING",
    )


@pytest.fixture
def client(settings, repo):
    app = create_app(settings)
    app.dependency_overrides[get_shipment_service] = lambda: ShipmentService(repo)
    app.dependency_overrides[get_route_service] = lambda: RouteService(repo)
    app.dependency_overrides[get_prediction_service] = lambda: PredictionService(FakePredictor(), repo)
    app.dependency_overrides[get_dq_service] = lambda: DataQualityReportService(settings.dq_report_path)
    with TestClient(app) as c:
        yield c


def assert_error(resp, status: int, code: str):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == resp.headers["x-request-id"]


class TestHealth:
    def test_liveness_always_ok(self, client):
        assert client.get("/health/live").json()["status"] == "alive"

    def test_readiness_degraded_when_dependencies_missing(self, client):
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["checks"]["database"]["status"] == "unavailable"
        assert body["checks"]["model"]["status"] == "unavailable"
        assert body["checks"]["data_quality_report"]["status"] == "ok"
        assert body["version"]


class TestShipments:
    def test_get_shipment(self, client):
        resp = client.get("/shipments/shp-00001")  # case-insensitive id
        assert resp.status_code == 200
        body = resp.json()
        assert body["shipment_id"] == "SHP-00001"
        assert body["origin"]["port_name"] == "Shanghai"

    def test_unknown_shipment_404(self, client):
        assert_error(client.get("/shipments/SHP-99999"), 404, "SHIPMENT_NOT_FOUND")

    def test_list_with_pagination(self, client):
        body = client.get("/shipments", params={"page": 2, "page_size": 2}).json()
        assert [i["shipment_id"] for i in body["items"]] == ["SHP-00003", "SHP-00004"]
        assert body["pagination"] == {"page": 2, "page_size": 2, "total_items": 5, "total_pages": 3}

    def test_filters_are_normalised_and_forwarded(self, client, repo):
        client.get("/shipments", params={"origin": "cnsha", "status": "DELAYED", "date_from": "2025-01-01"})
        filters, _ = next(arg for name, arg in repo.calls if name == "list")
        assert filters.origin == "CNSHA"
        assert filters.status == "DELAYED"

    @pytest.mark.parametrize(
        "params",
        [
            {"status": "LOST"},
            {"origin": "TOOLONGCODE"},
            {"page": 0},
            {"date_from": "not-a-date"},
            {"sort_by": "customer_id; DROP TABLE x"},
            {"page_size": 51},
        ],
    )
    def test_invalid_query_params_422(self, client, params):
        resp = client.get("/shipments", params=params)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] in {"VALIDATION_ERROR", "INVALID_REQUEST"}

    def test_inverted_date_range_422(self, client):
        resp = client.get("/shipments", params={"date_from": "2025-03-01", "date_to": "2025-01-01"})
        assert_error(resp, 422, "INVALID_REQUEST")


class TestRoutes:
    def test_route_stats(self, client):
        body = client.get("/routes/CNSHA/NLRTM/stats").json()
        assert body["on_time_rate"] == 0.75
        assert body["route_key"] == "CNSHA → NLRTM"

    def test_unknown_port_404(self, client):
        assert_error(client.get("/routes/CNSHA/XXTST/stats"), 404, "PORT_NOT_FOUND")

    def test_same_port_422(self, client):
        assert_error(client.get("/routes/CNSHA/CNSHA/stats"), 422, "INVALID_REQUEST")


class TestPredict:
    def test_predict_by_shipment_id(self, client):
        body = client.post("/predict-delay", json={"shipment_id": "SHP-00001"}).json()
        assert body["delay_probability"] == 0.72
        assert body["model_version"] == "test-model"

    def test_predict_by_booking(self, client):
        payload = {
            "booking": {
                "origin_port": "CNSHA",
                "destination_port": "NLRTM",
                "cargo_type": "Electronics",
                "booking_date": "2025-01-01T00:00:00",
                "planned_departure": "2025-01-05T00:00:00",
                "planned_arrival": "2025-02-01T00:00:00",
            }
        }
        assert client.post("/predict-delay", json=payload).status_code == 200

    def test_unknown_shipment_404(self, client):
        assert_error(client.post("/predict-delay", json={"shipment_id": "SHP-99999"}), 404, "SHIPMENT_NOT_FOUND")

    @pytest.mark.parametrize("payload", [{}, {"shipment_id": "bad-id"}, {"shipment_id": "SHP-00001", "booking": {}}])
    def test_bad_payload_422(self, client, payload):
        assert_error(client.post("/predict-delay", json=payload), 422, "VALIDATION_ERROR")

    def test_model_not_loaded_503(self, settings, repo):
        app = create_app(settings)
        app.dependency_overrides[get_prediction_service] = lambda: PredictionService(None, repo)
        with TestClient(app) as c:
            assert_error(c.post("/predict-delay", json={"shipment_id": "SHP-00001"}), 503, "SERVICE_UNAVAILABLE")


class TestDataQuality:
    def test_report(self, client):
        body = client.get("/data-quality/report").json()
        assert body["checks"][0]["check_name"] == "x"

    def test_bad_filter_422(self, client):
        assert client.get("/data-quality/report", params={"severity": "panic"}).status_code == 422


class TestCrossCutting:
    def test_request_id_is_echoed(self, client):
        resp = client.get("/health/live", headers={"X-Request-ID": "abc-123"})
        assert resp.headers["x-request-id"] == "abc-123"

    def test_unknown_route_structured_404(self, client):
        assert_error(client.get("/does-not-exist"), 404, "ROUTE_NOT_FOUND")

    def test_wrong_method_405(self, client):
        assert_error(client.get("/predict-delay"), 405, "METHOD_NOT_ALLOWED")

    def test_unhandled_exception_is_structured_500(self, settings):
        class Boom:
            def get(self, _):
                raise RuntimeError("kaboom")

        app = create_app(settings)
        app.dependency_overrides[get_shipment_service] = lambda: Boom()
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/shipments/SHP-00001")
        assert_error(resp, 500, "INTERNAL_ERROR")
        assert "kaboom" not in resp.text  # internals are never leaked to clients
