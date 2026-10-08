from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints

PortCode = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^[A-Za-z]{5}$"),
]
ShipmentId = Annotated[str, StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^SHP-\d{5}$")]


class ShipmentStatus(str, Enum):
    DELIVERED = "DELIVERED"
    DELAYED = "DELAYED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class ErrorDetail(BaseModel):
    code: str = Field(examples=["SHIPMENT_NOT_FOUND"])
    message: str
    details: dict[str, Any] | list[Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Pagination(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int


# Re-usable OpenAPI documentation of error responses.
def error_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {code: {"model": ErrorResponse} for code in codes}
