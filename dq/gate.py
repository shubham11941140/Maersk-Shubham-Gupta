"""Regression gate: compare a fresh report with a committed baseline.

The raw data has *known* issues that the pipeline handles, so "fail on any
critical issue" would be permanently red. What CI must catch is change:

* a check that used to pass now fails (a new kind of problem), or
* a failing check's row count grows beyond the tolerance (the problem got worse), or
* a check crashed.

Improvements (fewer rows) never fail the gate; refresh the baseline to lock them in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Regression:
    check_name: str
    reason: str

    def __str__(self) -> str:
        return f"{self.check_name}: {self.reason}"


def baseline_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_version": report.get("report_version"),
        "input_files": {k: v.get("sha256") for k, v in report.get("input_files", {}).items()},
        "checks": {c["check_name"]: c["rows_affected"] for c in report["checks"]},
    }


def load_baseline(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_baseline(report: dict[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(baseline_from_report(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare(report: dict[str, Any], baseline: dict[str, Any], tolerance_pct: float = 10.0) -> list[Regression]:
    expected: dict[str, int] = baseline.get("checks", {})
    regressions: list[Regression] = []
    for check in report["checks"]:
        name, rows = check["check_name"], int(check["rows_affected"])
        if check.get("status") == "error":
            regressions.append(Regression(name, f"check errored: {check.get('error')}"))
            continue
        if name not in expected:
            if rows:
                regressions.append(Regression(name, f"new check fails on {rows} rows (no baseline)"))
            continue
        base = int(expected[name])
        if base == 0 and rows > 0:
            regressions.append(Regression(name, f"previously passing, now {rows} rows"))
        elif base > 0 and rows > base * (1 + tolerance_pct / 100):
            regressions.append(Regression(name, f"{base} -> {rows} rows (> {tolerance_pct:g}% tolerance)"))
    return regressions
