"""Column-level profiling of the raw files (shape, blanks, cardinality, ranges).

Profiling is what surfaced most of the checks in ``dq.checks``; publishing it in
the report lets reviewers see the evidence behind each decision.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from shared.reference import DATETIME_FORMAT

_TOP_N = 5
_LOW_CARDINALITY = 30
_PARSE_RATE = 0.95


def _round(value: Any) -> Any:
    if hasattr(value, "item"):  # numpy scalar -> plain Python, so json can serialise it
        value = value.item()
    return round(float(value), 4) if isinstance(value, float | int) else value


def profile_column(series: pd.Series) -> dict[str, Any]:
    raw = series.fillna("").astype(str)
    blank = raw.str.strip() == ""
    present = raw[~blank]
    out: dict[str, Any] = {
        "blank_count": int(blank.sum()),
        "blank_pct": round(100.0 * blank.mean(), 3) if len(raw) else 0.0,
        "distinct_count": int(present.nunique()),
    }
    if present.empty:
        out["inferred_type"] = "empty"
        return out

    numeric = pd.to_numeric(present, errors="coerce")
    stamps = pd.to_datetime(present, format=DATETIME_FORMAT, errors="coerce")
    if numeric.notna().mean() >= _PARSE_RATE:
        out |= {
            "inferred_type": "numeric",
            "unparseable_count": int(numeric.isna().sum()),
            "min": _round(numeric.min()),
            "p50": _round(numeric.median()),
            "max": _round(numeric.max()),
        }
    elif stamps.notna().mean() >= _PARSE_RATE:
        out |= {
            "inferred_type": "timestamp",
            "unparseable_count": int(stamps.isna().sum()),
            "min": str(stamps.min()),
            "max": str(stamps.max()),
        }
    else:
        out["inferred_type"] = "string"
    if out["distinct_count"] <= _LOW_CARDINALITY:
        out["top_values"] = {str(k): int(v) for k, v in present.value_counts().head(_TOP_N).items()}
    return out


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "column_profiles": {col: profile_column(df[col]) for col in df.columns},
    }
