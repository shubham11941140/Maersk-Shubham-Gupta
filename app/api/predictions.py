from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_prediction_service
from app.schemas.common import error_responses
from app.schemas.predictions import PredictDelayRequest, PredictDelayResponse
from app.services.prediction_service import PredictionService

router = APIRouter(tags=["predictions"])


@router.post(
    "/predict-delay",
    response_model=PredictDelayResponse,
    summary="Predict the risk that a shipment arrives > 24h late (booking-time features only)",
    responses=error_responses(404, 422, 503),
)
def predict_delay(
    request: PredictDelayRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictDelayResponse:
    return service.predict(request)
