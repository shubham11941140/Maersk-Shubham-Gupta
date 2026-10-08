import numpy as np
import pandas as pd

from ml.monitoring import (
    PSI_ALERT,
    build_reference,
    drift_report,
    feature_psi,
    retrain_decision,
)


def frame(n=2000, shift=0.0, seed=0, ports=("CNSHA", "NLRTM")):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "weight_tons": rng.normal(1000 + shift, 100, n),
            "departure_month": rng.integers(1, 13, n).astype(float),
            "origin_port": rng.choice(ports, n),
        }
    )


def proba(seed=0, n=2000):
    return 0.5 + np.random.default_rng(seed).normal(0, 0.05, n)


def reference(seed=0):
    X = frame(seed=seed)
    return build_reference(X, proba(seed), 0.40)


def test_same_distribution_is_stable():
    ref = reference()
    r = drift_report(ref, frame(seed=1), proba(1))
    assert r["features"]["weight_tons"]["status"] == "ok"
    assert r["features"]["origin_port"]["status"] == "ok"
    assert r["decision"]["retrain"] is False


def test_numeric_shift_triggers_alert_and_retrain():
    ref = reference()
    r = drift_report(ref, frame(shift=300, seed=1), proba(1))
    assert r["features"]["weight_tons"]["psi"] > PSI_ALERT
    assert r["decision"]["retrain"]
    assert any("weight_tons" in reason for reason in r["decision"]["reasons"])


def test_unseen_categories_are_measured():
    ref = reference()
    shifted = frame(seed=1, ports=("CNSHA", "XXNEW"))
    res = feature_psi(ref["features"]["origin_port"], shifted["origin_port"])
    assert res["unseen_share"] > 0.4


def test_seasonal_feature_never_triggers_alone():
    ref = reference()
    summer = frame(seed=1).assign(departure_month=7.0)
    r = drift_report(ref, summer, proba(1))
    assert r["features"]["departure_month"]["status"] == "seasonal"
    assert not any("departure_month" in x for x in r["decision"]["reasons"])


def test_performance_triggers():
    ref = reference()
    report = {
        "features": {},
        "prediction": {"status": "ok", "psi": 0.0},
        "performance": {"roc_auc": 0.60, "validated_cv_auc": 0.70, "positive_rate_true": 0.55},
    }
    reasons = retrain_decision(report, ref).reasons
    assert any("ROC-AUC" in r for r in reasons)
    assert any("base rate" in r for r in reasons)
