"""Train the booking-time delay-risk model and write the artefact.

    python -m ml.train --db data/warehouse/supply_chain.duckdb --out artifacts/model

Methodology
-----------
* Target: arrived > 24h late (``actual_delay_hours > 24``). Rows without an observed
  outcome (cancelled / UNKNOWN) are excluded — we cannot label them.
* Temporal split on booking_date: oldest 70% train, next 15% validation, newest 15%
  test. A random split would leak future seasonality into training.
* Model: one-hot + scaled logistic regression. Simple, calibrated-ish, explainable.
* Decision threshold is fixed at 0.5 on a class-balanced model. Tuning it for F1 on
  validation degenerates to "flag everything" when the signal is weak (it did here),
  which would look good on F1 and be useless to an operator.
* Test metrics are reported next to trivial baselines so the number has context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    NUMERIC_FEATURES,
    PortInfo,
    build_features,
    build_target,
)

logger = logging.getLogger(__name__)

MODEL_FILENAME = "delay_model.joblib"
METADATA_FILENAME = "model_metadata.json"
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5


def load_training_frame(db_path: Path) -> tuple[pd.DataFrame, dict[str, PortInfo]]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(
            """
            SELECT shipment_id, origin_port, destination_port, cargo_type, container_count, weight_tons,
                   booking_date, planned_departure, planned_arrival, actual_delay_hours
            FROM curated.shipments
            WHERE actual_delay_hours IS NOT NULL
            ORDER BY booking_date, shipment_id
            """
        ).df()
        ports = {
            code: PortInfo(region=region, congestion=float(score))
            for code, region, score in con.execute(
                "SELECT port_code, region, avg_congestion_score FROM curated.ports"
            ).fetchall()
        }
    return df, ports


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), list(CATEGORICAL_FEATURES)),
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                list(NUMERIC_FEATURES),
            ),
        ]
    )
    model = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def temporal_split(n: int, train_frac: float = 0.70, val_frac: float = 0.15) -> tuple[slice, slice, slice]:
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "positive_rate_true": round(float(y_true.mean()), 4),
        "positive_rate_pred": round(float(pred.mean()), 4),
    }


def train(db_path: Path, out_dir: Path) -> dict:
    df, ports = load_training_frame(db_path)
    X = build_features(df, ports)
    y = build_target(df["actual_delay_hours"]).to_numpy()
    assert not (set(X.columns) & LEAKY_COLUMNS), "leaky column reached the feature frame"

    tr, va, te = temporal_split(len(X))
    pipe = build_pipeline().fit(X.iloc[tr], y[tr])
    threshold = DECISION_THRESHOLD

    val_metrics = metrics(y[va], pipe.predict_proba(X.iloc[va])[:, 1], threshold)
    test_proba = pipe.predict_proba(X.iloc[te])[:, 1]
    test_metrics = metrics(y[te], test_proba, threshold)

    base_rate = float(y[tr].mean())
    baselines = {
        "always_predict_delayed": metrics(y[te], np.ones(len(y[te])), 0.5),
        "train_base_rate_constant": metrics(y[te], np.full(len(y[te]), base_rate), 0.5),
    }

    # Refit on train+validation (all but the untouched test slice was used for selection).
    final = build_pipeline().fit(X.iloc[: va.stop], y[: va.stop])

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILENAME
    joblib.dump(final, model_path)
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    trained_at = datetime.now(UTC)

    metadata = {
        "model_name": "delay_risk_logreg",
        "model_version": trained_at.strftime("%Y%m%d%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "artifact_sha256": sha,
        "target": "actual_delay_hours > 24",
        "decision_threshold": threshold,
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "split": {
            "method": "temporal by booking_date (70/15/15)",
            "train_rows": tr.stop,
            "val_rows": va.stop - va.start,
            "test_rows": te.stop - te.start,
            "test_booking_date_from": str(df["booking_date"].iloc[te.start]),
        },
        "metrics": {"validation": val_metrics, "test": test_metrics, "test_baselines": baselines},
        "ports": {code: {"region": p.region, "congestion": p.congestion} for code, p in ports.items()},
    }
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the delay-risk model.")
    parser.add_argument("--db", type=Path, default=Path("data/warehouse/supply_chain.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/model"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    meta = train(args.db, args.out)
    print(
        json.dumps(
            {
                "model_version": meta["model_version"],
                "threshold": meta["decision_threshold"],
                "metrics": meta["metrics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
