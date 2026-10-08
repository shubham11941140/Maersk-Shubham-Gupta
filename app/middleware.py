"""Request logging, correlation id and last-resort error boundary (pure ASGI middleware)."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import error_body
from app.logging_config import request_id_ctx

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "x-request-id"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1").strip()
        request_id = incoming[:128] if incoming else uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((REQUEST_ID_HEADER.encode(), request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Error boundary: runs while the request id is still in context, so the
            # traceback log and the 500 body both carry it. Internals never leak.
            logger.exception("request.unhandled_exception")
            if response_started:
                raise
            response = JSONResponse(error_body("INTERNAL_ERROR", "An unexpected error occurred"), status_code=500)
            await response(scope, receive, send_wrapper)
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            route = scope.get("route")
            client = scope.get("client")
            logger.log(
                logging.WARNING if status_code >= 500 else logging.INFO,
                "http.request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "route": getattr(route, "path", None),
                    "query": scope.get("query_string", b"").decode("latin-1") or None,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "client_ip": client[0] if client else None,
                },
            )
            request_id_ctx.reset(token)
