"""Thin HTTP client for the Supply Chain Intelligence API — the assistant's only data access.

The assistant never touches the database directly: it goes through the same validated,
read-only endpoints as every other consumer, so guardrails like input validation,
pagination caps and structured errors are inherited for free.
"""

from __future__ import annotations

from typing import Any

import httpx


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message, self.details = status, code, message, details

    def to_dict(self) -> dict[str, Any]:
        return {"http_status": self.status, "code": self.code, "message": self.message, "details": self.details}


class SupplyChainApi:
    def __init__(self, base_url: str, timeout_s: float = 10.0, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(503, "API_UNREACHABLE", f"Could not reach the data API: {exc}") from exc
        if resp.status_code >= 400:
            try:
                err = resp.json()["error"]
                raise ApiError(
                    resp.status_code, err.get("code", "HTTP_ERROR"), err.get("message", ""), err.get("details")
                )
            except (ValueError, KeyError, TypeError):
                raise ApiError(resp.status_code, "HTTP_ERROR", resp.text[:300]) from None
        return resp.json()

    # --- endpoints used by the tools ------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def ports(self) -> list[dict[str, Any]]:
        return self._request("GET", "/ports")["items"]

    def get_shipment(self, shipment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/shipments/{shipment_id}")

    def list_shipments(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", "/shipments", params={k: v for k, v in params.items() if v is not None})

    def route_stats(self, origin: str, destination: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "GET", f"/routes/{origin}/{destination}/stats", params={k: v for k, v in params.items() if v is not None}
        )

    def route_rankings(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", "/routes/rankings", params={k: v for k, v in params.items() if v is not None})

    def predict_delay(self, shipment_id: str) -> dict[str, Any]:
        return self._request("POST", "/predict-delay", json={"shipment_id": shipment_id})
