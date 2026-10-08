"""CLI for the layered pipeline.

    python -m pipeline --input ./data/raw --db ./data/warehouse/supply_chain.duckdb
    python -m pipeline ... --export-parquet ./data/export        # also write Parquet
    python -m pipeline --db ./data/warehouse/supply_chain.duckdb --fingerprint-only

Exit codes: 0 = built and valid · 1 = data-contract validation failed (live warehouse untouched).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.build import build_warehouse, export_parquet, read_fingerprint
from pipeline.validate import PipelineValidationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the DuckDB warehouse from the raw CSVs.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/warehouse/supply_chain.duckdb"))
    parser.add_argument("--export-parquet", type=Path, help="Also export curated/serving relations as Parquet")
    parser.add_argument("--fingerprint-only", action="store_true", help="Print the content fingerprint and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if args.fingerprint_only:
        print(read_fingerprint(args.db))
        return 0

    try:
        result = build_warehouse(args.input, args.db)
    except PipelineValidationError as exc:
        print(f"PIPELINE REJECTED — {exc}", file=sys.stderr)
        return 1

    summary = {
        "run_id": result.run_id,
        "db": str(result.db_path),
        "fingerprint": result.fingerprint,
        "row_counts": result.row_counts,
        "validation": {"rules": len(result.validation), "passed": sum(v.passed for v in result.validation)},
    }
    if args.export_parquet:
        summary["parquet"] = [str(p) for p in export_parquet(result.db_path, args.export_parquet)]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
