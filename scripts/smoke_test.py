"""Post-deployment smoke test. Stdlib only, so it runs anywhere (laptop, CI, bastion).

    python scripts/smoke_test.py --base-url https://sci.example.com --expected-sha <git sha>

Exit code 0 = every check passed. Each check prints PASS/FAIL with latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Response:
    status: int
    body: Any
    latency_ms: float


def call(base_url: str, path: str, payload: dict | None = None, timeout: float = 5.0) -> Response:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - URL comes from the operator
        base_url.rstrip("/") + path,
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json", "X-Request-ID": f"smoke-{int(time.time())}"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    latency = (time.perf_counter() - start) * 1000
    try:
        body = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        body = raw.decode(errors="replace")
    return Response(status, body, latency)


def run(base_url: str, expected_sha: str | None, max_latency_ms: float) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        results.append((name, ok, detail))
        return ok

    health = call(base_url, "/health")
    body = health.body if isinstance(health.body, dict) else {}
    check(
        "health: HTTP 200 + status ok",
        health.status == 200 and body.get("status") == "ok",
        f"{health.status} {body.get('status')}",
    )
    for component, info in (body.get("checks") or {}).items():
        check(f"health: {component}", info.get("status") == "ok", json.dumps(info.get("detail", {}))[:120])
    if expected_sha:
        check(
            "health: build_sha matches release",
            body.get("build_sha") == expected_sha,
            f"deployed={body.get('build_sha')} expected={expected_sha}",
        )

    listing = call(base_url, "/shipments?page_size=1")
    items = (listing.body or {}).get("items", []) if isinstance(listing.body, dict) else []
    if not check("shipments: list returns data", listing.status == 200 and len(items) == 1, f"{listing.status}"):
        return results
    sample = items[0]

    detail = call(base_url, f"/shipments/{sample['shipment_id']}")
    check("shipments: lookup by id", detail.status == 200, f"{sample['shipment_id']} -> {detail.status}")

    stats = call(base_url, f"/routes/{sample['origin_port']}/{sample['destination_port']}/stats")
    check("routes: stats", stats.status == 200 and (stats.body or {}).get("shipment_count", 0) >= 1, f"{stats.status}")

    pred = call(base_url, "/predict-delay", {"shipment_id": sample["shipment_id"]})
    prob = (pred.body or {}).get("delay_probability") if isinstance(pred.body, dict) else None
    check(
        "predict-delay: returns a probability",
        pred.status == 200 and prob is not None and 0 <= prob <= 1,
        f"{pred.status} p={prob}",
    )

    dq = call(base_url, "/data-quality/report")
    n_checks = len((dq.body or {}).get("checks", [])) if isinstance(dq.body, dict) else 0
    check("data-quality: report served", dq.status == 200 and n_checks > 0, f"{dq.status} checks={n_checks}")

    missing = call(base_url, "/shipments/SHP-99999")
    code = (missing.body or {}).get("error", {}).get("code") if isinstance(missing.body, dict) else None
    check("errors: structured 404", missing.status == 404 and code == "SHIPMENT_NOT_FOUND", f"{missing.status} {code}")

    slowest = max(r.latency_ms for r in (health, listing, detail, stats, pred, dq, missing))
    check(f"latency: every call < {max_latency_ms:.0f} ms", slowest < max_latency_ms, f"slowest={slowest:.0f} ms")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-deployment smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--expected-sha", default=None, help="Fail unless /health reports this build_sha")
    parser.add_argument("--max-latency-ms", type=float, default=1000.0)
    args = parser.parse_args(argv)

    try:
        results = run(args.base_url, args.expected_sha, args.max_latency_ms)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        print(f"FAIL  service unreachable at {args.base_url}: {exc}")
        return 1

    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<45} {detail}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
