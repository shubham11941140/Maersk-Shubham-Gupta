from datetime import date

import pytest

from app.errors import DependencyUnavailableError, InvalidRequestError, PortNotFoundError, ShipmentNotFoundError
from app.repositories.shipment_repository import PageRequest, ShipmentFilters
from app.schemas.predictions import BookingFeatures, PredictDelayRequest
from app.services.prediction_service import PredictionService
from app.services.route_service import RouteService
from app.services.shipment_service import ShipmentService
from tests.fakes import FakePredictor, FakeShipmentRepository, make_shipment


@pytest.fixture
def repo():
    return FakeShipmentRepository(
        shipments={"SHP-00001": make_shipment("SHP-00001"), "SHP-00002": make_shipment("SHP-00002", dq_flags=None)}
    )


class TestShipmentService:
    def test_get_maps_port_details(self, repo):
        detail = ShipmentService(repo).get("SHP-00001")
        assert detail.origin.port_name == "Shanghai"
        assert detail.destination.region == "EUR"
        assert detail.dq_flags == []

    def test_get_unknown_raises(self, repo):
        with pytest.raises(ShipmentNotFoundError):
            ShipmentService(repo).get("SHP-99999")

    def test_list_pagination_math(self, repo):
        repo.total = 41
        result = ShipmentService(repo).list(ShipmentFilters(), PageRequest(page=1, page_size=20))
        assert result.pagination.total_pages == 3
        assert len(result.items) == 2

    def test_list_empty_has_zero_pages(self):
        result = ShipmentService(FakeShipmentRepository()).list(ShipmentFilters(), PageRequest())
        assert result.pagination.total_pages == 0

    def test_inverted_date_range_rejected_before_hitting_repo(self, repo):
        with pytest.raises(InvalidRequestError):
            ShipmentService(repo).list(
                ShipmentFilters(date_from=date(2025, 2, 1), date_to=date(2025, 1, 1)), PageRequest()
            )
        assert not any(name == "list" for name, _ in repo.calls)


class TestRouteService:
    def test_on_time_rate_and_rounding(self):
        repo = FakeShipmentRepository(
            stats={
                "shipment_count": 10,
                "completed_count": 8,
                "on_time_count": 6,
                "cancelled_count": 2,
                "avg_delay_hours": 12.3456,
            }
        )
        stats = RouteService(repo).stats("CNSHA", "NLRTM")
        assert stats.on_time_rate == 0.75
        assert stats.avg_delay_hours == 12.35
        assert stats.route_key == "CNSHA → NLRTM"

    def test_no_completed_shipments_gives_null_rate(self):
        stats = RouteService(FakeShipmentRepository(stats={"shipment_count": 0})).stats("CNSHA", "NLRTM")
        assert stats.on_time_rate is None
        assert stats.shipment_count == 0

    def test_unknown_port(self):
        with pytest.raises(PortNotFoundError) as exc:
            RouteService(FakeShipmentRepository()).stats("CNSHA", "XXTST")
        assert exc.value.details == {"port_codes": ["XXTST"]}

    def test_same_origin_destination(self):
        with pytest.raises(InvalidRequestError):
            RouteService(FakeShipmentRepository()).stats("CNSHA", "CNSHA")


class TestPredictionService:
    def test_predict_by_id_uses_booking_time_inputs_only(self, repo):
        predictor = FakePredictor()
        result = PredictionService(predictor, repo).predict(PredictDelayRequest(shipment_id="SHP-00001"))
        assert result.shipment_id == "SHP-00001"
        assert result.risk_band == "HIGH"
        sent = predictor.received[0]
        assert "actual_arrival" not in sent
        assert "status" not in sent

    def test_predict_unknown_shipment(self, repo):
        with pytest.raises(ShipmentNotFoundError):
            PredictionService(FakePredictor(), repo).predict(PredictDelayRequest(shipment_id="SHP-99999"))

    def test_predict_booking_with_unknown_port(self, repo):
        booking = BookingFeatures(**{**repo.get_booking_inputs("SHP-00001"), "destination_port": "ZZZZZ"})
        with pytest.raises(PortNotFoundError):
            PredictionService(FakePredictor(), repo).predict(PredictDelayRequest(booking=booking))

    def test_model_not_loaded(self, repo):
        with pytest.raises(DependencyUnavailableError):
            PredictionService(None, repo).predict(PredictDelayRequest(shipment_id="SHP-00001"))


class TestRequestValidation:
    def test_exactly_one_of_id_or_booking(self):
        with pytest.raises(ValueError, match="exactly one"):
            PredictDelayRequest()

    def test_booking_sequence_validated(self, repo):
        inputs = repo.get_booking_inputs("SHP-00001")
        with pytest.raises(ValueError, match="planned_arrival must be after"):
            BookingFeatures(**{**inputs, "planned_arrival": inputs["planned_departure"]})

    def test_port_codes_normalised(self, repo):
        inputs = repo.get_booking_inputs("SHP-00001")
        assert BookingFeatures(**{**inputs, "origin_port": " sgsin "}).origin_port == "SGSIN"
