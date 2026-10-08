"""CLI: ``python -m dq --input ./data/raw --output ./artifacts/dq_report.json``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dq.runner import build_report, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data-quality checks against the raw CSV files.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Directory containing the raw CSVs")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dq_report.json"), help="Report path")
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit 1 if any critical check fails (useful as a CI / pre-ingest gate).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = build_report(args.input)
    write_report(report, args.output)

    summary = report["summary"]
    print(f"DQ report written to {args.output}")
    print(
        f"  checks run: {summary['checks_run']}  passed: {summary['checks_passed']}  failed: {summary['checks_failed']}"
    )
    for check in report["checks"]:
        if check["status"] != "pass":
            sev, name, rows, action = check["severity"], check["check_name"], check["rows_affected"], check["action"]
            print(f"  [{sev:>8}] {name:<45} {rows:>6} rows  -> {action}")

    if summary["checks_errored"]:
        return 2
    if args.fail_on_critical and summary["failed_by_severity"].get("critical"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
