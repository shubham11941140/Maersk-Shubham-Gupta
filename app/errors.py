"""Domain exceptions and the handlers that turn them into structured JSON errors.

Error body (always the same shape)::

    {"error": {"code": "SHIPMENT_NOT_FOUND", "message": "...", "details": {...}, "request_id": "..."}}
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_config import request_id_ctx

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = HTTPStatus.NOT_FOUND
    code = "NOT_FOUND"


class ShipmentNotFoundError(NotFoundError):
    code = "SHIPMENT_NOT_FOUND"

    def __init__(self, shipment_id: str) -> None:
        super().__init__(f"Shipment '{shipment_id}' was not found", {"shipment_id": shipment_id})


class PortNotFoundError(NotFoundError):
    code = "PORT_NOT_FOUND"

    def __init__(self, port_codes: list[str]) -> None:
        super().__init__(f"Unknown port code(s): {', '.join(port_codes)}", {"port_codes": port_codes})


class InvalidRequestError(AppError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "INVALID_REQUEST"


class DependencyUnavailableError(AppError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "SERVICE_UNAVAILABLE"


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details if details is not None else {},
            "request_id": request_id_ctx.get(),
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        level = logging.ERROR if exc.status_code >= 500 else logging.INFO
        logger.log(level, "request.app_error", extra={"error_code": exc.code, "error_message": exc.message})
        return JSONResponse(error_body(exc.code, exc.message, exc.details), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"location": list(err.get("loc", ())), "message": err.get("msg"), "type": err.get("type")}
            for err in exc.errors()
        ]
        return JSONResponse(
            error_body("VALIDATION_ERROR", "Request validation failed", {"errors": details}),
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "ROUTE_NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("request.unhandled_exception")
        return JSONResponse(
            error_body("INTERNAL_ERROR", "An unexpected error occurred"),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
