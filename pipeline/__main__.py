"""CLI: ``python -m pipeline --input ./data/raw --db ./data/warehouse/supply_chain.duckdb``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pipeline.build import build_warehouse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the DuckDB warehouse from the raw CSVs.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=Path("data/warehouse/supply_chain.duckdb"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    result = build_warehouse(args.input, args.db)
    print(json.dumps({"run_id": result.run_id, "db": str(result.db_path), "row_counts": result.row_counts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
