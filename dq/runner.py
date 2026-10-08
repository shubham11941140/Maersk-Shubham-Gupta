"""Load raw CSVs, run every registered check, build the JSON report."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from dq import checks as _checks  # noqa: F401  (import registers the checks)
from dq.checks import REGISTRY
from dq.models import Check, CheckResult, RawData
from dq.profile import profile_dataset

logger = logging.getLogger(__name__)

REPORT_VERSION = "1.0"
DATASET_FILES = {"ports": "ports.csv", "shipments": "shipments.csv", "port_events": "port_events.csv"}
_MAX_SAMPLE_KEYS = 10


def load_raw(input_dir: Path) -> RawData:
    """Read every CSV with all columns as strings, so the checks see exactly what is on disk."""
    frames: dict[str, pd.DataFrame] = {}
    for dataset, filename in DATASET_FILES.items():
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected raw file not found: {path}")
        frames[dataset] = pd.read_csv(path, dtype=str, keep_default_na=False)
    return RawData(**frames)


def run_check(check: Check, data: RawData) -> CheckResult:
    df = data.get(check.dataset)
    try:
        mask = check.fn(data).fillna(False).astype(bool)
        affected = df.loc[mask, check.key_column]
        samples = sorted({str(k) for k in affected})[:_MAX_SAMPLE_KEYS]
        return CheckResult(
            check_name=check.name,
            dataset=check.dataset,
            description=check.description,
            detection=check.detection,
            severity=check.severity,
            action=check.action,
            rationale=check.rationale,
            rows_affected=int(mask.sum()),
            total_rows=len(df),
            sample_keys=samples,
        )
    except Exception as exc:  # one broken check must not hide the results of the others
        logger.exception("DQ check %s failed to execute", check.name)
        return CheckResult(
            check_name=check.name,
            dataset=check.dataset,
            description=check.description,
            detection=check.detection,
            severity=check.severity,
            action=check.action,
            rationale=check.rationale,
            rows_affected=0,
            total_rows=len(df),
            error=f"{type(exc).__name__}: {exc}",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(input_dir: Path, checks: list[Check] | None = None, include_profile: bool = True) -> dict:
    data = load_raw(input_dir)
    results = [run_check(c, data) for c in (checks or REGISTRY)]
    failed = [r for r in results if r.status == "fail"]
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_files": {
            name: {"file": fn, "rows": len(data.get(name)), "sha256": _sha256(input_dir / fn)}
            for name, fn in DATASET_FILES.items()
        },
        "summary": {
            "checks_run": len(results),
            "checks_passed": sum(r.status == "pass" for r in results),
            "checks_failed": len(failed),
            "checks_errored": sum(r.status == "error" for r in results),
            "failed_by_severity": dict(Counter(r.severity.value for r in failed)),
            "rows_affected_by_dataset": {
                name: int(sum(r.rows_affected for r in failed if r.dataset == name)) for name in DATASET_FILES
            },
        },
        "checks": [r.to_dict() for r in results],
    }
    if include_profile:
        report["profile"] = {name: profile_dataset(data.get(name)) for name in DATASET_FILES}
    return report


def write_report(report: dict, output: Path) -> None:
    """Write atomically so readers (the API) never see a half-written file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(output)
