"""Production monitoring for the delay model: drift (PSI) and live performance.

At training time ``build_reference`` stores the distribution of every feature and of
the predicted probability in the model metadata. In production ``drift_report``
compares a recent window against that reference; ``retrain_decision`` turns the
numbers into an explicit yes/no with reasons.

PSI (population stability index) bands, the usual industry convention:
  < 0.10 stable · 0.10–0.25 moderate shift (investigate) · > 0.25 major shift (act)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PSI_WARN = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4
_N_BINS = 10
_OTHER = "__other__"

# Retrain triggers (documented in the README → "Monitoring the model in production").
AUC_DROP_TRIGGER = 0.05  # live ROC-AUC this far below the validated CV AUC
BASE_RATE_SHIFT_TRIGGER = 0.05  # absolute change in the share of late shipments
UNSEEN_CATEGORY_TRIGGER = 0.05  # share of traffic with categories unseen in training
MIN_LABELLED_FOR_PERFORMANCE = 200
# Calendar features legitimately differ between any short window and a full year of training
# data. Their PSI is reported but never triggers a retrain on its own.
SEASONAL_FEATURES = frozenset({"departure_month"})


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_reference(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna().astype(float)
    edges = np.unique(np.quantile(clean, np.linspace(0, 1, _N_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    counts = np.histogram(clean, bins=edges)[0]
    return {
        "type": "numeric",
        "edges": [float(x) for x in edges],
        "proportions": [round(float(c / max(len(clean), 1)), 6) for c in counts],
        "null_rate": round(float(values.isna().mean()), 6),
    }


def _categorical_reference(values: pd.Series, top_n: int = 30) -> dict[str, Any]:
    shares = values.fillna("NULL").astype(str).value_counts(normalize=True)
    top = shares.head(top_n)
    props = {str(k): round(float(v), 6) for k, v in top.items()}
    props[_OTHER] = round(float(max(0.0, 1 - top.sum())), 6)
    return {"type": "categorical", "proportions": props}


def build_reference(features: pd.DataFrame, proba: np.ndarray, base_rate: float) -> dict[str, Any]:
    ref: dict[str, Any] = {"features": {}, "base_rate": round(float(base_rate), 4)}
    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]):
            ref["features"][col] = _numeric_reference(features[col])
        else:
            ref["features"][col] = _categorical_reference(features[col])
    ref["prediction"] = _numeric_reference(pd.Series(proba))
    return ref


def feature_psi(reference: dict[str, Any], values: pd.Series) -> dict[str, Any]:
    if reference["type"] == "numeric":
        clean = values.dropna().astype(float)
        counts = np.histogram(clean, bins=np.asarray(reference["edges"]))[0]
        actual = counts / max(len(clean), 1)
        return {
            "psi": round(_psi(reference["proportions"], actual), 4),
            "null_rate": round(float(values.isna().mean()), 4),
            "reference_null_rate": reference["null_rate"],
        }
    ref_props: dict[str, float] = reference["proportions"]
    shares = values.fillna("NULL").astype(str).value_counts(normalize=True)
    known = [k for k in ref_props if k != _OTHER]
    actual = [float(shares.get(k, 0.0)) for k in known] + [float(shares[~shares.index.isin(known)].sum())]
    expected = [ref_props[k] for k in known] + [ref_props[_OTHER]]
    return {"psi": round(_psi(expected, actual), 4), "unseen_share": round(actual[-1], 4)}


def _band(psi: float) -> str:
    return "alert" if psi > PSI_ALERT else "warn" if psi > PSI_WARN else "ok"


@dataclass
class RetrainDecision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)


def drift_report(
    reference: dict[str, Any],
    features: pd.DataFrame,
    proba: np.ndarray,
    y_true: np.ndarray | None = None,
    validated_auc: float | None = None,
) -> dict[str, Any]:
    from ml.evaluation import classification_metrics  # local import keeps this module light

    per_feature = {}
    for col, ref in reference["features"].items():
        if col in features:
            r = feature_psi(ref, features[col])
            per_feature[col] = r | {"status": "seasonal" if col in SEASONAL_FEATURES else _band(r["psi"])}
    pred = feature_psi(reference["prediction"], pd.Series(proba))
    report: dict[str, Any] = {
        "n": len(features),
        "features": per_feature,
        "prediction": pred | {"status": _band(pred["psi"]), "mean": round(float(np.mean(proba)), 4)},
    }
    if y_true is not None and len(y_true) >= MIN_LABELLED_FOR_PERFORMANCE:
        perf = classification_metrics(np.asarray(y_true), proba, 0.5)
        report["performance"] = perf | {
            "validated_cv_auc": validated_auc,
            "base_rate_reference": reference["base_rate"],
        }
    report["decision"] = retrain_decision(report, reference).__dict__
    return report


def retrain_decision(report: dict[str, Any], reference: dict[str, Any]) -> RetrainDecision:
    reasons = []
    for col, r in report["features"].items():
        if r["status"] == "alert":
            reasons.append(f"feature drift: {col} PSI {r['psi']} > {PSI_ALERT}")
        if r.get("unseen_share", 0) > UNSEEN_CATEGORY_TRIGGER:
            reasons.append(f"unseen categories in {col}: {r['unseen_share']:.1%} of traffic")
    if report["prediction"]["status"] == "alert":
        reasons.append(f"prediction drift: PSI {report['prediction']['psi']} > {PSI_ALERT}")
    perf = report.get("performance")
    if perf:
        if perf.get("validated_cv_auc") and perf["roc_auc"] < perf["validated_cv_auc"] - AUC_DROP_TRIGGER:
            reasons.append(
                f"live ROC-AUC {perf['roc_auc']} is > {AUC_DROP_TRIGGER} below validated {perf['validated_cv_auc']}"
            )
        if abs(perf["positive_rate_true"] - reference["base_rate"]) > BASE_RATE_SHIFT_TRIGGER:
            reasons.append(f"late-shipment base rate moved {reference['base_rate']} → {perf['positive_rate_true']}")
    return RetrainDecision(retrain=bool(reasons), reasons=reasons)
