"""Train, evaluate and package the booking-time delay-risk model.

    python -m ml.train --db data/warehouse/supply_chain.duckdb --out artifacts/model

Methodology
-----------
* **Target**: arrived more than 24h late (``actual_delay_hours > 24``). Shipments without an
  observed outcome (cancelled / UNKNOWN) are excluded — they cannot be labelled.
* **Features**: booking-time only (``ml.features``). A test asserts no leaky column reaches
  the model; point-in-time history features are evaluated as an experiment.
* **Splits** (all temporal, ordered by booking_date — a random split would leak the future):
  - development = oldest 85% → rolling-origin CV (4 expanding-window folds) for model selection;
  - test = newest 15% → touched once, for the final numbers (+ bootstrap 95% CIs).
* **Candidates**: a prior-only baseline, logistic regression, gradient boosting, and logistic
  regression + point-in-time history features. Selection rule: highest mean CV PR-AUC, *unless*
  the gain over the simplest servable model is within one standard deviation — then keep the
  simplest. Non-servable candidates (need a feature store) are reported but not selectable.
* **Sanity check**: a label-permutation test — if the real model doesn't beat models trained on
  shuffled labels, the features carry no detectable signal, and we say so.
* **Threshold** fixed at 0.5 on a class-balanced model (F1-tuning degenerates to "flag all").
* **Production artefact**: the selected model refit on *all* labelled data (most recent data
  matters for a temporal problem); the reported test metrics come from the dev-only fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.evaluation import (
    bootstrap_ci,
    calibration_bins,
    classification_metrics,
    cross_validate,
    permutation_test,
    threshold_table,
)
from ml.features import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_DOCS,
    LEAKY_COLUMNS,
    NUMERIC_FEATURES,
    PortInfo,
    build_features,
    build_target,
)
from ml.history_features import HISTORY_FEATURES, load_history_features
from ml.monitoring import build_reference

logger = logging.getLogger(__name__)

MODEL_FILENAME = "delay_model.joblib"
METADATA_FILENAME = "model_metadata.json"
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5
DEV_FRACTION = 0.85
N_FOLDS = 4


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
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


def load_history(db_path: Path, shipment_ids: pd.Series) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        hist = load_history_features(con)
    return hist.reindex(shipment_ids.to_numpy()).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Candidate models
# --------------------------------------------------------------------------- #
def _preprocess(numeric: list[str], dense: bool = False) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=not dense),
                list(CATEGORICAL_FEATURES),
            ),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ]
    )


def build_pipeline() -> Pipeline:
    """The production model: one-hot + scaled, class-balanced logistic regression."""
    model = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("preprocess", _preprocess(list(NUMERIC_FEATURES))), ("model", model)])


def _hgb() -> Pipeline:
    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=RANDOM_STATE
    )
    return Pipeline([("preprocess", _preprocess(list(NUMERIC_FEATURES), dense=True)), ("model", model)])


def _prior() -> Pipeline:
    return Pipeline([("preprocess", _preprocess(list(NUMERIC_FEATURES))), ("model", DummyClassifier(strategy="prior"))])


def _logreg_history() -> Pipeline:
    model = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", random_state=RANDOM_STATE)
    numeric = list(NUMERIC_FEATURES) + list(HISTORY_FEATURES)
    return Pipeline([("preprocess", _preprocess(numeric)), ("model", model)])


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[], Pipeline]
    uses_history: bool
    servable: bool
    complexity: int  # lower = simpler; used to break near-ties
    description: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("prior_baseline", _prior, False, True, 0, "Predicts the training base rate for everyone."),
    Candidate("logreg_booking", build_pipeline, False, True, 1, "Logistic regression, booking-time features."),
    Candidate("hgb_booking", _hgb, False, True, 2, "Gradient boosting, booking-time features."),
    Candidate(
        "logreg_booking_plus_history",
        _logreg_history,
        True,
        False,
        3,
        "Logistic regression + point-in-time vessel/customer/route/port history. Needs a feature store.",
    ),
)


def select_candidate(results: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Highest mean CV PR-AUC among servable, non-baseline models — unless the gain over the
    simplest servable model is within one std, in which case simplicity wins."""
    servable = [c for c in CANDIDATES if c.servable and c.name != "prior_baseline"]
    simplest = min(servable, key=lambda c: c.complexity)
    best = max(servable, key=lambda c: results[c.name]["summary"]["pr_auc"]["mean"])
    if best.name == simplest.name:
        return best.name, f"{best.name} has the highest mean CV PR-AUC and is the simplest servable model."
    gain = results[best.name]["summary"]["pr_auc"]["mean"] - results[simplest.name]["summary"]["pr_auc"]["mean"]
    std = results[simplest.name]["summary"]["pr_auc"]["std"]
    if gain <= std:
        return simplest.name, (
            f"{best.name} beats {simplest.name} by only {gain:.4f} mean CV PR-AUC (≤ 1 std = {std:.4f}); "
            f"keeping the simpler, more explainable {simplest.name}."
        )
    return best.name, f"{best.name} beats {simplest.name} by {gain:.4f} mean CV PR-AUC (> 1 std = {std:.4f})."


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(db_path: Path, out_dir: Path, run_experiments: bool = True, n_permutations: int = 20) -> dict:
    df, ports = load_training_frame(db_path)
    X = build_features(df, ports)
    y = build_target(df["actual_delay_hours"]).to_numpy()
    assert not (set(X.columns) & LEAKY_COLUMNS), "leaky column reached the feature frame"

    dev_end = int(len(X) * DEV_FRACTION)
    dev, test = slice(0, dev_end), slice(dev_end, len(X))

    # ---- model selection on the development set (rolling-origin CV) ----
    X_hist = None
    experiments: dict[str, dict[str, Any]] = {}
    candidates = CANDIDATES if run_experiments else tuple(c for c in CANDIDATES if c.name == "logreg_booking")
    for cand in candidates:
        frame = X
        if cand.uses_history:
            if X_hist is None:
                X_hist = pd.concat([X, load_history(db_path, df["shipment_id"])], axis=1)
            frame = X_hist
        cv = cross_validate(cand.factory, frame.iloc[dev], y[dev], DECISION_THRESHOLD, N_FOLDS)
        experiments[cand.name] = cv | {"description": cand.description, "servable": cand.servable}
        logger.info("cv %s pr_auc=%s roc_auc=%s", cand.name, cv["summary"]["pr_auc"], cv["summary"]["roc_auc"])

    if run_experiments:
        chosen_name, rationale = select_candidate(experiments)
    else:
        chosen_name, rationale = "logreg_booking", "experiments skipped (--no-experiments)"
    chosen = next(c for c in CANDIDATES if c.name == chosen_name)

    # ---- final, one-shot evaluation on the untouched test slice ----
    dev_model = chosen.factory().fit(X.iloc[dev], y[dev])
    test_proba = dev_model.predict_proba(X.iloc[test])[:, 1]
    y_test = y[test]
    base_rate = float(y[dev].mean())
    test_eval = {
        "metrics": classification_metrics(y_test, test_proba, DECISION_THRESHOLD),
        "roc_auc_ci": bootstrap_ci(y_test, test_proba),
        "pr_auc_ci": bootstrap_ci(y_test, test_proba, metric=average_precision_score),
        "threshold_table": threshold_table(y_test, test_proba),
        "calibration": calibration_bins(y_test, test_proba),
        "baselines": {
            "always_predict_delayed": classification_metrics(y_test, np.ones(len(y_test)), DECISION_THRESHOLD),
            "prior_base_rate": classification_metrics(y_test, np.full(len(y_test), base_rate), DECISION_THRESHOLD),
        },
    }
    sanity = (
        permutation_test(
            chosen.factory,
            X.iloc[dev],
            y[dev],
            experiments[chosen_name]["summary"]["roc_auc"]["mean"],
            n_permutations,
            N_FOLDS,
        )
        if run_experiments and n_permutations
        else None
    )

    # ---- production artefact: refit on all labelled data ----
    final = chosen.factory().fit(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILENAME
    joblib.dump(final, model_path)
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    trained_at = datetime.now(UTC)
    reference = build_reference(X, final.predict_proba(X)[:, 1], float(y.mean()))

    metadata = {
        "model_name": f"delay_risk_{chosen_name}",
        "model_version": trained_at.strftime("%Y%m%d%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "artifact_sha256": sha,
        "target": "actual_delay_hours > 24",
        "decision_threshold": DECISION_THRESHOLD,
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "feature_docs": FEATURE_DOCS,
        "excluded_features": EXCLUDED_FEATURES,
        "data": {
            "labelled_rows": len(X),
            "positive_rate": round(float(y.mean()), 4),
            "booking_date_min": str(df["booking_date"].min()),
            "booking_date_max": str(df["booking_date"].max()),
        },
        "split": {
            "method": f"temporal by booking_date; dev = oldest {DEV_FRACTION:.0%} (rolling-origin CV, "
            f"{N_FOLDS} expanding folds), test = newest {1 - DEV_FRACTION:.0%} (evaluated once)",
            "dev_rows": dev.stop,
            "test_rows": test.stop - test.start,
            "test_booking_date_from": str(df["booking_date"].iloc[test.start]),
            "production_fit": "selected model refit on all labelled rows",
        },
        "selection": {"chosen": chosen_name, "rationale": rationale},
        "experiments": experiments,
        "evaluation": {"test": test_eval, "permutation_test": sanity},
        # kept for backwards compatibility with earlier tooling / README
        "metrics": {"test": test_eval["metrics"], "test_baselines": test_eval["baselines"]},
        "monitoring_reference": reference,
        "ports": {code: {"region": p.region, "congestion": p.congestion} for code, p in ports.items()},
    }
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the delay-risk model.")
    parser.add_argument("--db", type=Path, default=Path("data/warehouse/supply_chain.duckdb"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--no-experiments", action="store_true", help="Skip candidate comparison + permutation test")
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = train(args.db, args.out, run_experiments=not args.no_experiments, n_permutations=args.permutations)
    ev = meta["evaluation"]
    print(
        json.dumps(
            {
                "model": meta["model_name"],
                "version": meta["model_version"],
                "selection": meta["selection"],
                "cv": {k: v["summary"] for k, v in meta["experiments"].items()},
                "test": ev["test"]["metrics"],
                "roc_auc_ci": ev["test"]["roc_auc_ci"],
                "permutation_test": ev["permutation_test"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
