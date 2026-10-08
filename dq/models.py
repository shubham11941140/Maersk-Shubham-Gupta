"""Data structures for DQ checks and their results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Severity(str, Enum):
    CRITICAL = "critical"  # would corrupt metrics or the model if left in
    WARNING = "warning"  # degrades quality; fixed or flagged in curation
    INFO = "info"  # worth knowing, no action needed


class Action(str, Enum):
    DROP = "drop"
    FIX = "fix"
    FLAG = "flag"
    QUARANTINE = "quarantine"
    NONE = "none"


@dataclass(frozen=True)
class RawData:
    """The three raw datasets, loaded as strings so nothing is silently coerced."""

    ports: pd.DataFrame
    shipments: pd.DataFrame
    port_events: pd.DataFrame

    def get(self, dataset: str) -> pd.DataFrame:
        return getattr(self, dataset)


# A check function receives the raw data and returns a boolean mask over the
# rows of its dataset: True = row is affected by the issue.
CheckFn = Callable[[RawData], pd.Series]


@dataclass(frozen=True)
class Check:
    name: str
    dataset: str
    key_column: str
    description: str
    detection: str
    severity: Severity
    action: Action
    rationale: str
    fn: CheckFn


@dataclass(frozen=True)
class CheckResult:
    check_name: str
    dataset: str
    description: str
    detection: str
    severity: Severity
    action: Action
    rationale: str
    rows_affected: int
    total_rows: int
    sample_keys: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def pct_affected(self) -> float:
        return round(100.0 * self.rows_affected / self.total_rows, 3) if self.total_rows else 0.0

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        return "fail" if self.rows_affected else "pass"

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "dataset": self.dataset,
            "description": self.description,
            "detection": self.detection,
            "status": self.status,
            "severity": self.severity.value,
            "rows_affected": self.rows_affected,
            "total_rows": self.total_rows,
            "pct_affected": self.pct_affected,
            "action": self.action.value,
            "rationale": self.rationale,
            "sample_keys": self.sample_keys,
            "error": self.error,
        }
