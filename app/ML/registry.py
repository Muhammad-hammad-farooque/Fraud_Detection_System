"""Versioned model storage.

Each saved model lives in its own directory with a manifest beside it:

    app/ML/artifacts/<version>/model.pkl
    app/ML/artifacts/<version>/manifest.json
    app/ML/artifacts/ACTIVE                  <- the version serving traffic

A model is never loaded without its manifest, and never loaded if the feature
order it was trained on differs from features.FEATURE_ORDER. That check is the
train/serve skew tripwire: it turns a silent wrong prediction into a refusal
to start.
"""
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ..features import FEATURE_ORDER

DEFAULT_DIR = Path(__file__).resolve().parent / "artifacts"
ACTIVE_FILE = "ACTIVE"
MODEL_FILE = "model.pkl"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class ModelManifest:
    version: str            # semver or timestamp
    trained_at: datetime
    feature_order: list[str]
    metrics: dict[str, float]
    training_rows: int
    algorithm: str

    def to_json(self) -> str:
        data = asdict(self)
        data["trained_at"] = self.trained_at.isoformat()
        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ModelManifest":
        data = json.loads(text)
        data["trained_at"] = datetime.fromisoformat(data["trained_at"])
        return cls(**data)


class RegistryError(RuntimeError):
    """A model or manifest is missing, unreadable or inconsistent."""


class FeatureOrderMismatch(RegistryError):
    """The model was trained on a different feature order than serving uses."""


def registry_dir() -> Path:
    return Path(os.getenv("MODEL_REGISTRY_DIR", str(DEFAULT_DIR)))


def _check_feature_order(model: Any, manifest: ModelManifest) -> None:
    if manifest.feature_order != FEATURE_ORDER:
        raise FeatureOrderMismatch(
            f"model {manifest.version} was trained on {manifest.feature_order}, "
            f"but serving computes {FEATURE_ORDER}"
        )
    trained_on = getattr(model, "feature_names_in_", None)
    if trained_on is None:
        raise FeatureOrderMismatch(
            f"model {manifest.version} carries no feature names; train it on a DataFrame "
            "with FEATURE_ORDER columns so the order can be verified"
        )
    if list(trained_on) != manifest.feature_order:
        raise FeatureOrderMismatch(
            f"model {manifest.version} was fitted on {list(trained_on)}, "
            f"but its manifest claims {manifest.feature_order}"
        )


def save_model(
    model: Any,
    algorithm: str,
    training_rows: int,
    metrics: dict[str, float],
    trained_at: datetime | None = None,
    version: str | None = None,
    root: Path | None = None,
) -> ModelManifest:
    """Write a model and its manifest as a new version. Does not activate it."""
    trained_at = trained_at or datetime.now(timezone.utc)
    version = version or f"{algorithm.lower()}-{trained_at:%Y%m%dT%H%M%S%fZ}"
    manifest = ModelManifest(
        version=version,
        trained_at=trained_at,
        feature_order=list(FEATURE_ORDER),
        metrics={name: float(value) for name, value in metrics.items()},
        training_rows=int(training_rows),
        algorithm=algorithm,
    )
    _check_feature_order(model, manifest)       # refuse to store a model that could never load

    target = (root or registry_dir()) / version
    if target.exists():
        raise RegistryError(f"model version {version} already exists; versions are immutable")
    target.mkdir(parents=True)
    joblib.dump(model, target / MODEL_FILE)
    (target / MANIFEST_FILE).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def activate(version: str, root: Path | None = None) -> None:
    """Point serving at a saved version. It must load cleanly first."""
    root = root or registry_dir()
    load_model(version, root=root)                  # refuse to activate a broken model
    (root / ACTIVE_FILE).write_text(version + "\n", encoding="utf-8")


def active_version(root: Path | None = None) -> str:
    path = (root or registry_dir()) / ACTIVE_FILE
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RegistryError(f"no active model: {path} does not exist")


def list_versions(root: Path | None = None) -> list[str]:
    root = root or registry_dir()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / MANIFEST_FILE).exists())


def load_model(version: str | None = None, root: Path | None = None) -> tuple[Any, ModelManifest]:
    """Loads a specific version, or the active one. Raises if the manifest's
    feature_order does not match features.FEATURE_ORDER - this is the
    train/serve skew tripwire."""
    root = root or registry_dir()
    version = version or active_version(root)
    target = root / version
    try:
        manifest = ModelManifest.from_json((target / MANIFEST_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RegistryError(f"model version {version} has no manifest at {target}")
    if manifest.version != version:
        raise RegistryError(f"manifest in {target} names version {manifest.version}")
    try:
        model = joblib.load(target / MODEL_FILE)
    except FileNotFoundError:
        raise RegistryError(f"model version {version} has no {MODEL_FILE}")
    _check_feature_order(model, manifest)
    return model, manifest
