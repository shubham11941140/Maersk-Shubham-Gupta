import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.evaluation import (
    bootstrap_ci,
    calibration_bins,
    classification_metrics,
    cross_validate,
    permutation_test,
    rolling_origin_splits,
    threshold_table,
)
from ml.train import CANDIDATES, select_candidate


def test_rolling_origin_splits_are_strictly_temporal_and_cover_the_tail():
    splits = rolling_origin_splits(100, n_folds=4, min_train_frac=0.4)
    assert len(splits) == 4
    for train, test in splits:
        assert train.start == 0
        assert train.stop == test.start  # train is always strictly before test
    assert splits[0][0].stop == 40
    assert splits[-1][1].stop == 100
    assert [t.start for _, t in splits] == sorted(t.start for _, t in splits)


def test_classification_metrics():
    m = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert (m["precision"], m["recall"], m["roc_auc"], m["n"]) == (0.5, 0.5, 0.75, 4)


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = (x + rng.normal(scale=0.5, size=400) > 0).astype(int)
    return pd.DataFrame({"x": x}), y


def factory():
    return LogisticRegression()


def test_cross_validate_and_permutation_detect_real_signal(separable):
    X, y = separable
    cv = cross_validate(factory, X, y, 0.5, n_folds=3)
    assert cv["summary"]["roc_auc"]["mean"] > 0.85
    perm = permutation_test(factory, X, y, cv["summary"]["roc_auc"]["mean"], n_permutations=5, n_folds=3)
    assert perm["p_value"] == pytest.approx(1 / 6, abs=1e-3)  # better than every shuffled-label model
    assert perm["null_auc_mean"] < 0.65


def test_permutation_test_does_not_find_signal_in_noise():
    rng = np.random.default_rng(1)
    X, y = pd.DataFrame({"x": rng.normal(size=400)}), rng.integers(0, 2, 400)
    cv = cross_validate(factory, X, y, 0.5, n_folds=3)
    perm = permutation_test(factory, X, y, cv["summary"]["roc_auc"]["mean"], n_permutations=10, n_folds=3)
    assert perm["p_value"] > 0.05


def test_bootstrap_ci_brackets_the_point_estimate(separable):
    X, y = separable
    ci = bootstrap_ci(y, X["x"].to_numpy(), n_boot=200)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_threshold_table_and_calibration():
    y, p = np.array([0, 1, 1, 0]), np.array([0.2, 0.8, 0.6, 0.4])
    rows = threshold_table(y, p, thresholds=[0.5])
    assert rows == [{"threshold": 0.5, "precision": 1.0, "recall": 1.0, "f1": 1.0, "flagged_rate": 0.5}]
    bins = calibration_bins(y, p, n_bins=2)
    assert [b["n"] for b in bins] == [2, 2]


def _summary(pr_mean, pr_std):
    return {"summary": {"pr_auc": {"mean": pr_mean, "std": pr_std}}}


def test_selection_prefers_simplest_unless_gain_exceeds_one_std():
    base = {c.name: _summary(0.40, 0.02) for c in CANDIDATES}
    assert select_candidate(base)[0] == "logreg_booking"
    small_gain = base | {"hgb_booking": _summary(0.41, 0.01)}
    assert select_candidate(small_gain)[0] == "logreg_booking"
    big_gain = base | {"hgb_booking": _summary(0.50, 0.01)}
    assert select_candidate(big_gain)[0] == "hgb_booking"
    non_servable = base | {"logreg_booking_plus_history": _summary(0.90, 0.01)}
    assert select_candidate(non_servable)[0] == "logreg_booking"  # needs a feature store -> not selectable
