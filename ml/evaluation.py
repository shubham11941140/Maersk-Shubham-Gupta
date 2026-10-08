"""Honest evaluation utilities: temporal CV, bootstrap CIs, permutation test, calibration.

Everything here is model-agnostic: functions take a factory that returns a fresh,
unfitted estimator, so the same protocol is applied to every candidate.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

EstimatorFactory = Callable[[], Any]


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    has_both = len(np.unique(y_true)) == 2
    return {
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if has_both else float("nan"),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4) if has_both else float("nan"),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "positive_rate_true": round(float(y_true.mean()), 4),
        "positive_rate_pred": round(float(pred.mean()), 4),
        "n": len(y_true),
    }


def rolling_origin_splits(n: int, n_folds: int = 4, min_train_frac: float = 0.4) -> list[tuple[slice, slice]]:
    """Expanding-window splits over time-ordered rows.

    The first ``min_train_frac`` of rows is always training; the remainder is cut into
    ``n_folds`` consecutive test blocks. Fold k trains on everything before block k.
    Training data is therefore always strictly older than test data.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    return [(slice(0, int(edges[k])), slice(int(edges[k]), int(edges[k + 1]))) for k in range(n_folds)]


def cross_validate(
    factory: EstimatorFactory,
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float,
    n_folds: int = 4,
) -> dict[str, Any]:
    folds = []
    for train_idx, test_idx in rolling_origin_splits(len(X), n_folds):
        model = factory().fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        folds.append(classification_metrics(y[test_idx], proba, threshold) | {"train_n": train_idx.stop})
    keys = ("precision", "recall", "f1", "roc_auc", "pr_auc", "brier")
    summary = {
        k: {
            "mean": round(float(np.mean([f[k] for f in folds])), 4),
            "std": round(float(np.std([f[k] for f in folds])), 4),
        }
        for k in keys
    }
    return {"folds": folds, "summary": summary}


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric(y_true[idx], proba[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "point": round(float(metric(y_true, proba)), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "confidence": 1 - alpha,
    }


def permutation_test(
    factory: EstimatorFactory,
    X: pd.DataFrame,
    y: np.ndarray,
    observed_auc: float,
    n_permutations: int = 20,
    n_folds: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Re-run the CV with shuffled labels. If the real model is no better than these
    null models, the features carry no detectable signal about the target."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        null.append(cross_validate(factory, X, y_perm, 0.5, n_folds)["summary"]["roc_auc"]["mean"])
    p_value = (1 + sum(a >= observed_auc for a in null)) / (1 + len(null))
    return {
        "observed_cv_auc": round(float(observed_auc), 4),
        "null_auc_mean": round(float(np.mean(null)), 4),
        "null_auc_p95": round(float(np.quantile(null, 0.95)), 4),
        "p_value": round(float(p_value), 4),
        "n_permutations": n_permutations,
    }


def threshold_table(y_true: np.ndarray, proba: np.ndarray, thresholds=None) -> list[dict[str, float]]:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.30, 0.71, 0.05), 2)
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, proba, float(t))
        rows.append(
            {
                "threshold": float(t),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "flagged_rate": m["positive_rate_pred"],
            }
        )
    return rows


def calibration_bins(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in pairwise(edges):
        mask = (proba >= lo) & (proba < hi if hi < 1 else proba <= hi)
        if mask.any():
            out.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(mask.sum()),
                    "mean_predicted": round(float(proba[mask].mean()), 4),
                    "observed_rate": round(float(y_true[mask].mean()), 4),
                }
            )
    return out
