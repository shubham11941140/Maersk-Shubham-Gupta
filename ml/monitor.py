"""Offline model monitoring job.

    python -m ml.monitor --db data/warehouse/supply_chain.duckdb --from 2025-07-01
    python -m ml.monitor --db ... --last-days 90      # window = the last 90 days of bookings in the data
    python -m ml.monitor ... --fail-on-retrain        # exit 3 when a retrain trigger fires (for schedulers)

Scores every booking in the window with the *deployed* artefact, compares feature and
prediction distributions with the training reference (PSI), measures live performance on
the shipments whose outcome is already known, and prints an explicit retrain decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

from ml.features import BOOKING_TIME_INPUTS, build_target
from ml.monitoring import drift_report
from ml.predictor import SklearnDelayPredictor

EXIT_RETRAIN = 3


def run(db_path: Path, model_dir: Path, date_from: date, date_to: date | None = None) -> dict:
    predictor = SklearnDelayPredictor.from_dir(model_dir)
    meta = predictor.metadata
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(
            f"SELECT {', '.join(BOOKING_TIME_INPUTS)}, actual_delay_hours FROM curated.shipments "  # noqa: S608
            "WHERE booking_date >= ? AND (CAST(? AS DATE) IS NULL OR booking_date < CAST(? AS DATE) + 1) "
            "AND status <> 'CANCELLED' ORDER BY booking_date",
            [date_from, date_to, date_to],
        ).df()
    if df.empty:
        raise SystemExit(f"No bookings between {date_from} and {date_to or 'now'}")

    features = predictor.features(df)
    proba = predictor.predict_proba_batch(df)
    labelled = df["actual_delay_hours"].notna().to_numpy()
    y = build_target(df.loc[labelled, "actual_delay_hours"]).to_numpy()

    chosen = meta.get("selection", {}).get("chosen", "logreg_booking")
    validated_auc = meta.get("experiments", {}).get(chosen, {}).get("summary", {}).get("roc_auc", {}).get("mean")
    report = drift_report(meta["monitoring_reference"], features, proba)
    if labelled.sum():
        perf_report = drift_report(meta["monitoring_reference"], features[labelled], proba[labelled], y, validated_auc)
        if "performance" in perf_report:
            report["performance"] = perf_report["performance"]
            report["decision"] = perf_report["decision"]
    trained_until = meta.get("data", {}).get("booking_date_max")
    overlaps = bool(trained_until) and str(date_from) <= str(trained_until)[:10]
    return {
        "model_version": predictor.model_version,
        "overlaps_training_data": overlaps,
        "note": (
            f"Window overlaps the training data (bookings up to {trained_until}); performance is in-sample and "
            "only demonstrates the job. In production, run it on bookings made after the model was trained."
            if overlaps
            else None
        ),
        "window": {"from": str(date_from), "to": str(date_to) if date_to else None},
        "bookings": len(df),
        "labelled": int(labelled.sum()),
        **report,
    }


def last_days_start(db_path: Path, days: int) -> date:
    with duckdb.connect(str(db_path), read_only=True) as con:
        latest = con.execute("SELECT max(booking_date)::DATE FROM curated.shipments").fetchone()[0]
    return latest - timedelta(days=days)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drift + performance monitoring for the delay model.")
    parser.add_argument("--db", type=Path, default=Path("data/warehouse/supply_chain.duckdb"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model"))
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--from", dest="date_from", type=date.fromisoformat)
    window.add_argument("--last-days", type=int, help="Window = the last N days of bookings in the warehouse")
    parser.add_argument("--to", dest="date_to", type=date.fromisoformat)
    parser.add_argument("--fail-on-retrain", action="store_true")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this file")
    args = parser.parse_args(argv)
    date_from = args.date_from or last_days_start(args.db, args.last_days)
    report = run(args.db, args.model_dir, date_from, args.date_to)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return EXIT_RETRAIN if args.fail_on_retrain and report["decision"]["retrain"] else 0


if __name__ == "__main__":
    sys.exit(main())
