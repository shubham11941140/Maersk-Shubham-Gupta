"""Content fingerprint of the derived layers — proves idempotency (same input ⇒ same output)."""

from __future__ import annotations

import hashlib

import duckdb

# Tables/views whose content must be deterministic. raw.* (ingest timestamp) and meta.* (run id) are excluded.
FINGERPRINT_RELATIONS = (
    "curated.ports",
    "curated.shipments",
    "curated.port_events",
    "quarantine.shipments",
    "quarantine.port_events",
    "serving.route_stats",
    "serving.route_quarterly_stats",
)


def table_fingerprints(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for rel in FINGERPRINT_RELATIONS:
        rows, digest = con.execute(
            f"SELECT count(*), md5(coalesce(string_agg(CAST(t AS VARCHAR), '\n' ORDER BY CAST(t AS VARCHAR)), '')) "  # noqa: S608
            f"FROM {rel} AS t"
        ).fetchone()
        out[rel] = {"rows": int(rows), "md5": digest}
    return out


def combine(fingerprints: dict[str, dict[str, object]]) -> str:
    parts = [f"{rel}:{v['rows']}:{v['md5']}" for rel, v in sorted(fingerprints.items())]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def warehouse_fingerprint(con: duckdb.DuckDBPyConnection) -> str:
    return combine(table_fingerprints(con))
