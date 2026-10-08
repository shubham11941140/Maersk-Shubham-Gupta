"""CLI for the standalone data-quality module.

    python -m dq --input ./data/raw                                  # write report
    python -m dq --input ./data/raw --baseline dq/baseline.json      # + regression gate (CI)
    python -m dq --input ./data/raw --update-baseline dq/baseline.json

Exit codes: 0 = OK · 1 = gate failed (regression, or critical issue with --fail-on-critical) · 2 = a check crashed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dq.gate import compare, load_baseline, write_baseline
from dq.runner import build_report, write_report


def _print_summary(report: dict, output: Path) -> None:
    summary = report["summary"]
    print(f"DQ report written to {output}")
    print(
        f"  checks run: {summary['checks_run']}  passed: {summary['checks_passed']}  "
        f"failed: {summary['checks_failed']}  errored: {summary['checks_errored']}"
    )
    for check in report["checks"]:
        if check["status"] != "pass":
            sev, name, rows, action = check["severity"], check["check_name"], check["rows_affected"], check["action"]
            print(f"  [{sev:>8}] {name:<48} {rows:>6} rows  -> {action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run data-quality checks against the raw CSV files.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="Directory containing the raw CSVs")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dq_report.json"), help="Report path")
    parser.add_argument("--no-profile", action="store_true", help="Skip column profiling in the report")
    parser.add_argument("--baseline", type=Path, help="Fail (exit 1) on regressions vs this baseline file")
    parser.add_argument("--tolerance", type=float, default=10.0, help="Allowed %% growth per failing check")
    parser.add_argument("--update-baseline", type=Path, help="Write a new baseline from this run")
    parser.add_argument(
        "--fail-on-critical", action="store_true", help="Exit 1 if any critical check fails (strict mode)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = build_report(args.input, include_profile=not args.no_profile)
    write_report(report, args.output)
    _print_summary(report, args.output)

    if report["summary"]["checks_errored"]:
        return 2

    if args.update_baseline:
        write_baseline(report, args.update_baseline)
        print(f"Baseline updated: {args.update_baseline}")

    exit_code = 0
    if args.baseline:
        regressions = compare(report, load_baseline(args.baseline), args.tolerance)
        if regressions:
            print(f"\nDQ GATE FAILED — {len(regressions)} regression(s) vs {args.baseline}:")
            for r in regressions:
                print(f"  - {r}")
            exit_code = 1
        else:
            print(f"\nDQ gate passed: no regressions vs {args.baseline} (tolerance {args.tolerance:g}%)")
    if args.fail_on_critical and report["summary"]["failed_by_severity"].get("critical"):
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
