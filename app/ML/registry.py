"""Versioned model storage.

Each saved model lives in its own directory with a manifest beside it:

    app/ML/artifacts/<version>/model.pkl
    app/ML/artifacts/<version>/manifest.json
    app/ML/artifacts/ACTIVE                  <- the version serving traffic

A model is never loaded without its manifest, and never loaded if it needs a
feature that features.FEATURE_ORDER does not compute. That check is the
train/serve skew tripwire: it turns a silent wrong prediction into a refusal
to start. A model may use a subset of FEATURE_ORDER; serving hands it exactly
the columns its manifest lists.
"""
import json
import os
from dataclasses import asdict, dataclass, field
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
    # Permutation importance on the held-out window (T-18). Empty for models
    # registered before it was recorded.
    feature_importance: dict[str, float] = field(default_factory=dict)
    # The reliability curve on the held-out window (T-19): per probability bin,
    # the mean predicted and the observed fraud rate, and how many fell in it.
    reliability: list[dict] = field(default_factory=list)

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
    """The model's features and what serving computes do not line up."""


def registry_dir() -> Path:
    return Path(os.getenv("MODEL_REGISTRY_DIR", str(DEFAULT_DIR)))


def _check_feature_order(model: Any, manifest: ModelManifest) -> None:
    """Serving must compute every feature the model needs, and the model must
    have been fitted on exactly the features its manifest lists.

    A model may use a subset of FEATURE_ORDER: serving hands it exactly its own
    columns, by name and in its own order (T-16). What it may not do is need a
    feature serving does not compute - that is train/serve skew.
    """
    missing = [name for name in manifest.feature_order if name not in FEATURE_ORDER]
    if missing:
        raise FeatureOrderMismatch(
            f"model {manifest.version} needs features serving does not compute: {missing}"
        )
    trained_on = getattr(model, "feature_names_in_", None)
    if trained_on is None:
        raise FeatureOrderMismatch(
            f"model {manifest.version} carries no feature names; train it on a DataFrame "
            "with named feature columns so they can be verified"
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
    feature_importance: dict[str, float] | None = None,
    reliability: list[dict] | None = None,
) -> ModelManifest:
    """Write a model and its manifest as a new version. Does not activate it."""
    trained_at = trained_at or datetime.now(timezone.utc)
    version = version or f"{algorithm.lower()}-{trained_at:%Y%m%dT%H%M%S%fZ}"
    manifest = ModelManifest(
        version=version,
        trained_at=trained_at,
        # The features the model was actually fitted on, which may be a subset
        # of FEATURE_ORDER. A model without names fails the check below.
        feature_order=list(getattr(model, "feature_names_in_", [])),
        metrics={name: float(value) for name, value in metrics.items()},
        training_rows=int(training_rows),
        algorithm=algorithm,
        feature_importance={name: float(value) for name, value in (feature_importance or {}).items()},
        reliability=list(reliability or []),
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
    """Loads a specific version, or the active one. Raises if the model needs a
    feature serving does not compute, or was fitted on columns its manifest
    does not list - this is the train/serve skew tripwire."""
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
