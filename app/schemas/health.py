from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ComponentHealth(BaseModel):
    status: Literal["ok", "unavailable"]
    detail: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    build_sha: str
    environment: str
    uptime_seconds: float
    checks: dict[str, ComponentHealth]
