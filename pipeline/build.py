"""Orchestrates the layered build.

    raw (exact copy, VARCHAR + lineage) → ref (reference tables from shared/reference.py)
      → curated (+ quarantine) → serving (views) → data-contract validation → atomic swap

Idempotency strategy: the whole warehouse is rebuilt into a temporary file and
atomically swapped into place with ``os.replace``. Running twice produces the
same tables (identical content fingerprint), never duplicates, and a failed or
invalid run leaves the previous warehouse untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from pipeline import sql
from pipeline.fingerprint import combine, table_fingerprints, warehouse_fingerprint
from pipeline.validate import MAX_QUARANTINE_PCT, ValidationResult, assert_valid
from shared.reference import (
    CANONICAL_CARGO_TYPES,
    CANONICAL_EVENT_TYPES,
    MAX_PLAUSIBLE_CONTAINERS,
    ON_TIME_THRESHOLD_HOURS,
    PORT_COUNTRY_FIXES,
    SENTINEL_VESSEL_IDS,
    STATUS_ALIASES,
    UNKNOWN_CARGO_TYPE,
    UNKNOWN_STATUS,
)

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.1.0"
RAW_FILES = {"ports": "ports.csv", "shipments": "shipments.csv", "port_events": "port_events.csv"}


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    db_path: Path
    row_counts: dict[str, int]
    fingerprint: str
    validation: list[ValidationResult]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_reference_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Materialise the Python reference values so SQL and Python share one source of truth."""
    con.execute("CREATE OR REPLACE TABLE ref.cargo_types (match_key VARCHAR PRIMARY KEY, canonical_cargo_type VARCHAR)")
    con.executemany("INSERT INTO ref.cargo_types VALUES (?, ?)", [(c.casefold(), c) for c in CANONICAL_CARGO_TYPES])

    con.execute("CREATE OR REPLACE TABLE ref.status_aliases (raw_status VARCHAR PRIMARY KEY, canonical_status VARCHAR)")
    con.executemany("INSERT INTO ref.status_aliases VALUES (?, ?)", list(STATUS_ALIASES.items()))

    con.execute("CREATE OR REPLACE TABLE ref.event_types (event_type VARCHAR PRIMARY KEY)")
    con.executemany("INSERT INTO ref.event_types VALUES (?)", [(e,) for e in sorted(CANONICAL_EVENT_TYPES)])

    con.execute("CREATE OR REPLACE TABLE ref.sentinel_vessels (vessel_id VARCHAR PRIMARY KEY)")
    con.executemany("INSERT INTO ref.sentinel_vessels VALUES (?)", [(v,) for v in sorted(SENTINEL_VESSEL_IDS)])

    con.execute("CREATE OR REPLACE TABLE ref.port_country_fixes (port_code VARCHAR PRIMARY KEY, country VARCHAR)")
    con.executemany("INSERT INTO ref.port_country_fixes VALUES (?, ?)", list(PORT_COUNTRY_FIXES.items()))


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608 - internal table names


def build_warehouse(input_dir: Path, db_path: Path, max_quarantine_pct: float = MAX_QUARANTINE_PCT) -> PipelineResult:
    input_dir, db_path = Path(input_dir), Path(db_path)
    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC).replace(tzinfo=None)
    sources = {}
    for table, filename in RAW_FILES.items():
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Raw input missing: {path}")
        sources[table] = {"file": filename, "sha256": _sha256(path)}

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(f".{db_path.name}.{run_id}.tmp")
    logger.info("pipeline.start run_id=%s input=%s target=%s", run_id, input_dir, db_path)

    con = duckdb.connect(str(tmp_path))
    try:
        for schema in sql.SCHEMA_NAMES:
            con.execute(sql.SCHEMAS.format(schema))

        # raw
        for table, meta in sources.items():
            con.execute(
                sql.RAW_LOAD.format(table=table),
                [meta["file"], meta["sha256"], started_at, str(input_dir / meta["file"])],
            )
        _load_reference_tables(con)

        # curated + quarantine
        con.execute(sql.CURATED_PORTS)
        con.execute(sql.STAGE_SHIPMENTS)
        con.execute(sql.QUARANTINE_SHIPMENTS)
        con.execute(
            sql.CURATED_SHIPMENTS,
            {
                "max_containers": MAX_PLAUSIBLE_CONTAINERS,
                "unknown_cargo": UNKNOWN_CARGO_TYPE,
                "on_time_hours": ON_TIME_THRESHOLD_HOURS,
                "unknown_status": UNKNOWN_STATUS,
            },
        )
        con.execute(sql.STAGE_EVENTS)
        con.execute(sql.QUARANTINE_EVENTS)
        con.execute(sql.CURATED_EVENTS)

        # serving
        con.execute(sql.SERVING_ROUTE_STATS)
        con.execute(sql.SERVING_ROUTE_QUARTERLY)
        con.execute(sql.SERVING_PORT_DAILY)
        con.execute(sql.SERVING_SHIPMENTS_ENRICHED)

        # data contract: raises PipelineValidationError -> tmp file discarded, live warehouse untouched
        validation = assert_valid(con, max_quarantine_pct)
        con.execute(
            "CREATE OR REPLACE TABLE meta.validation_results (rule VARCHAR, description VARCHAR, violations BIGINT)"
        )
        con.executemany(
            "INSERT INTO meta.validation_results VALUES (?, ?, ?)",
            [(v.rule, v.description, v.violations) for v in validation],
        )
        fingerprints = table_fingerprints(con)
        fingerprint = combine(fingerprints)
        con.execute("CREATE OR REPLACE TABLE meta.table_fingerprints (relation VARCHAR, row_count BIGINT, md5 VARCHAR)")
        con.executemany(
            "INSERT INTO meta.table_fingerprints VALUES (?, ?, ?)",
            [(rel, v["rows"], v["md5"]) for rel, v in fingerprints.items()],
        )

        row_counts = {
            t: _count(con, t)
            for t in (
                "raw.ports",
                "raw.shipments",
                "raw.port_events",
                "curated.ports",
                "curated.shipments",
                "curated.port_events",
                "quarantine.shipments",
                "quarantine.port_events",
            )
        }
        con.execute(
            sql.PIPELINE_RUN,
            [
                run_id,
                started_at,
                datetime.now(UTC).replace(tzinfo=None),
                PIPELINE_VERSION,
                json.dumps(row_counts),
                json.dumps(sources),
                fingerprint,
            ],
        )
        con.execute("CHECKPOINT")
    except Exception:
        con.close()
        tmp_path.unlink(missing_ok=True)
        logger.exception("pipeline.failed run_id=%s", run_id)
        raise
    con.close()

    os.replace(tmp_path, db_path)  # atomic swap
    logger.info("pipeline.done run_id=%s fingerprint=%s row_counts=%s", run_id, fingerprint, json.dumps(row_counts))
    return PipelineResult(
        run_id=run_id, db_path=db_path, row_counts=row_counts, fingerprint=fingerprint, validation=validation
    )


EXPORT_RELATIONS = (
    "curated.ports",
    "curated.shipments",
    "curated.port_events",
    "quarantine.shipments",
    "quarantine.port_events",
    "serving.route_stats",
    "serving.route_quarterly_stats",
    "serving.port_daily_activity",
    "serving.shipments_enriched",
)


def export_parquet(db_path: Path, out_dir: Path) -> list[Path]:
    """Write curated / quarantine / serving relations as Parquet (for BI tools or a data lake)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        for rel in EXPORT_RELATIONS:
            target = out_dir / f"{rel}.parquet"
            con.execute(f"COPY (SELECT * FROM {rel}) TO '{target}' (FORMAT PARQUET)")  # noqa: S608
            written.append(target)
    return written


def read_fingerprint(db_path: Path) -> str:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return warehouse_fingerprint(con)
