"""Training + monitoring jobs on the real warehouse (fast mode: no candidate sweep / permutations)."""

from __future__ import annotations

import pytest

from ml import monitor
from ml.predictor import SklearnDelayPredictor
from ml.train import train
from pipeline.build import build_warehouse
from tests.conftest import MODEL_DIR, RAW_DIR


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    db = tmp_path_factory.mktemp("wh") / "wh.duckdb"
    build_warehouse(RAW_DIR, db)
    return db


def test_training_produces_a_loadable_documented_artefact(warehouse, tmp_path):
    meta = train(warehouse, tmp_path, run_experiments=False)
    predictor = SklearnDelayPredictor.from_dir(tmp_path)
    assert predictor.model_version == meta["model_version"]
    ev = meta["evaluation"]["test"]
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= ev["metrics"][key] <= 1.0
    assert ev["roc_auc_ci"]["ci_low"] <= ev["roc_auc_ci"]["point"] <= ev["roc_auc_ci"]["ci_high"]
    assert set(meta["feature_docs"]) == set(meta["feature_columns"])
    assert "monitoring_reference" in meta


def test_committed_artefact_has_full_evaluation():
    meta = SklearnDelayPredictor.from_dir(MODEL_DIR).metadata
    assert {"prior_baseline", "logreg_booking", "hgb_booking", "logreg_booking_plus_history"} <= set(
        meta["experiments"]
    )
    assert meta["evaluation"]["permutation_test"]["n_permutations"] >= 10
    assert meta["selection"]["chosen"] in meta["experiments"]


def test_monitoring_job(warehouse):
    report = monitor.run(warehouse, MODEL_DIR, __import__("datetime").date(2025, 7, 1))
    assert report["bookings"] > 0
    assert report["overlaps_training_data"] is True
    assert set(report["features"]) >= {"origin_port", "weight_tons"}
    assert "retrain" in report["decision"]
    assert monitor.main(["--db", str(warehouse), "--model-dir", str(MODEL_DIR), "--from", "2025-07-01"]) == 0


def test_monitoring_last_days_window(warehouse, tmp_path):
    out = tmp_path / "monitor.json"
    assert (
        monitor.main(["--db", str(warehouse), "--model-dir", str(MODEL_DIR), "--last-days", "60", "--output", str(out)])
        == 0
    )
    assert out.exists()
