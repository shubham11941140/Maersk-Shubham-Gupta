import json
import shutil

import pytest

from ml.predictor import ModelLoadError, SklearnDelayPredictor
from tests.conftest import MODEL_DIR

BOOKING = {
    "origin_port": "CNSHA",
    "destination_port": "NLRTM",
    "cargo_type": "Electronics",
    "container_count": 10,
    "weight_tons": 500.0,
    "booking_date": "2025-01-01 00:00:00",
    "planned_departure": "2025-01-08 00:00:00",
    "planned_arrival": "2025-02-08 00:00:00",
}


def test_committed_artifact_loads_and_predicts():
    predictor = SklearnDelayPredictor.from_dir(MODEL_DIR)
    result = predictor.predict(BOOKING)
    assert 0.0 <= result.delay_probability <= 1.0
    assert result.risk_band in {"LOW", "MEDIUM", "HIGH"}
    assert result.model_version == predictor.model_version


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        SklearnDelayPredictor.from_dir(tmp_path)


def test_tampered_artifact_is_rejected(tmp_path):
    shutil.copytree(MODEL_DIR, tmp_path / "m")
    meta_path = tmp_path / "m" / "model_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["artifact_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError, match="checksum"):
        SklearnDelayPredictor.from_dir(tmp_path / "m")
