"""Serving-side model wrapper. Loaded once at API start-up, never retrains."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd

from ml.features import PortInfo, build_features
from ml.train import METADATA_FILENAME, MODEL_FILENAME


class ModelLoadError(RuntimeError):
    """Raised when the artefact is missing, tampered with or incompatible."""


@dataclass(frozen=True)
class Prediction:
    delay_probability: float
    predicted_delayed: bool
    risk_band: str
    threshold: float
    model_version: str


class Predictor(Protocol):
    """Interface the API depends on — lets tests swap in a fake model."""

    @property
    def model_version(self) -> str: ...

    def predict(self, booking: dict[str, Any]) -> Prediction: ...


def risk_band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold - 0.15:
        return "MEDIUM"
    return "LOW"


class SklearnDelayPredictor:
    def __init__(self, pipeline: Any, metadata: dict[str, Any]) -> None:
        self._pipeline = pipeline
        self._metadata = metadata
        self._threshold = float(metadata["decision_threshold"])
        self._ports = {
            code: PortInfo(region=v["region"], congestion=float(v["congestion"]))
            for code, v in metadata.get("ports", {}).items()
        }

    @classmethod
    def from_dir(cls, model_dir: Path) -> SklearnDelayPredictor:
        model_path, meta_path = model_dir / MODEL_FILENAME, model_dir / METADATA_FILENAME
        if not model_path.is_file() or not meta_path.is_file():
            raise ModelLoadError(f"Model artefact not found in {model_dir}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        # joblib == pickle: only load artefacts we produced. Verify the hash first.
        actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_sha != metadata.get("artifact_sha256"):
            raise ModelLoadError("Model artefact checksum mismatch — refusing to load")
        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:  # version skew, corrupt file, ...
            raise ModelLoadError(f"Could not deserialise model: {exc}") from exc
        return cls(pipeline, metadata)

    @property
    def model_version(self) -> str:
        return str(self._metadata["model_version"])

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def features(self, bookings: pd.DataFrame) -> pd.DataFrame:
        return build_features(bookings, self._ports)

    def predict_proba_batch(self, bookings: pd.DataFrame) -> Any:
        """Probabilities for many bookings at once (used by offline monitoring)."""
        return self._pipeline.predict_proba(self.features(bookings))[:, 1]

    def predict(self, booking: dict[str, Any]) -> Prediction:
        features = build_features(pd.DataFrame([booking]), self._ports)
        proba = float(self._pipeline.predict_proba(features)[0, 1])
        return Prediction(
            delay_probability=round(proba, 4),
            predicted_delayed=proba >= self._threshold,
            risk_band=risk_band(proba, self._threshold),
            threshold=self._threshold,
            model_version=self.model_version,
        )
