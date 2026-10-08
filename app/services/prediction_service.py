"""Delay-risk predictions — glue between the API, the repository and the ML predictor."""

from __future__ import annotations

import logging

from app.errors import DependencyUnavailableError, PortNotFoundError, ShipmentNotFoundError
from app.repositories.shipment_repository import ShipmentRepository
from app.schemas.predictions import BookingFeatures, PredictDelayRequest, PredictDelayResponse
from ml.predictor import Predictor

# Dedicated logger: every served prediction is one structured line. This is the raw
# material for production monitoring (prediction drift, join with outcomes later).
prediction_log = logging.getLogger("app.predictions")


class PredictionService:
    def __init__(self, predictor: Predictor | None, repository: ShipmentRepository) -> None:
        self._predictor = predictor
        self._repo = repository

    def predict(self, request: PredictDelayRequest) -> PredictDelayResponse:
        if self._predictor is None:
            raise DependencyUnavailableError("Delay model is not loaded")

        if request.shipment_id is not None:
            row = self._repo.get_booking_inputs(request.shipment_id)
            if row is None:
                raise ShipmentNotFoundError(request.shipment_id)
            booking = BookingFeatures.model_validate(row)
        else:
            booking = request.booking
            assert booking is not None  # guaranteed by the request validator
            ports = [booking.origin_port, booking.destination_port]
            missing = sorted(set(ports) - self._repo.existing_ports(ports))
            if missing:
                raise PortNotFoundError(missing)

        result = self._predictor.predict(booking.model_dump())
        prediction_log.info(
            "prediction.served",
            extra={
                "model_version": result.model_version,
                "shipment_id": request.shipment_id,
                "delay_probability": result.delay_probability,
                "risk_band": result.risk_band,
                "predicted_delayed": result.predicted_delayed,
                "inputs": booking.model_dump(mode="json"),
            },
        )
        return PredictDelayResponse(
            shipment_id=request.shipment_id,
            delay_probability=result.delay_probability,
            predicted_delayed=result.predicted_delayed,
            risk_band=result.risk_band,
            decision_threshold=result.threshold,
            model_version=result.model_version,
            inputs=booking,
        )
