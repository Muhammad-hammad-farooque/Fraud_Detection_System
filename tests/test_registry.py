"""
Model registry — versioned models, manifests, and the skew tripwire (T-13).
"""
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.features import BASELINE_FEATURES, FEATURE_ORDER
from app.ML import registry
from app.ML.models import MANIFEST, MODEL_VERSION
from app.ML.registry import (
    FeatureOrderMismatch,
    ModelManifest,
    RegistryError,
    activate,
    active_version,
    list_versions,
    load_model,
    save_model,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
TRAINED_AT = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _fit(columns=None):
    columns = columns or BASELINE_FEATURES
    rows = [[100, 1.0, 0, 0, 0], [9000, 20.0, 1, 1, 9]] * 4
    X = pd.DataFrame(rows, columns=columns)
    y = [0, 1] * 4
    return RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)


def _save(root, version="v1", model=None, **metrics):
    return save_model(model or _fit(), "RandomForestClassifier", training_rows=8,
                      metrics=metrics or {"auc": 0.9}, trained_at=TRAINED_AT,
                      version=version, root=root)


class TestSaveAndLoad:
    def test_every_saved_model_has_a_manifest_beside_it(self, tmp_path):
        _save(tmp_path)
        assert (tmp_path / "v1" / "model.pkl").exists()
        manifest = json.loads((tmp_path / "v1" / "manifest.json").read_text())
        assert manifest["feature_order"] == BASELINE_FEATURES     # what it was fitted on
        assert manifest["training_rows"] == 8
        assert manifest["algorithm"] == "RandomForestClassifier"
        assert manifest["metrics"] == {"auc": 0.9}
        assert manifest["trained_at"].startswith("2026-09-01T12:00")

    def test_round_trip(self, tmp_path):
        saved = _save(tmp_path)
        model, manifest = load_model("v1", root=tmp_path)
        assert manifest == saved
        assert list(model.feature_names_in_) == BASELINE_FEATURES

    def test_versions_are_immutable(self, tmp_path):
        _save(tmp_path)
        with pytest.raises(RegistryError, match="already exists"):
            _save(tmp_path)

    def test_default_version_is_a_timestamp(self, tmp_path):
        manifest = save_model(_fit(), "RandomForestClassifier", 8, {}, trained_at=TRAINED_AT, root=tmp_path)
        assert manifest.version.startswith("randomforestclassifier-20260901T120000")

    def test_list_versions(self, tmp_path):
        _save(tmp_path, "v1")
        _save(tmp_path, "v2")
        assert list_versions(tmp_path) == ["v1", "v2"]
        assert list_versions(tmp_path / "empty") == []


class TestActivation:
    def test_activate_and_load_the_active_version(self, tmp_path):
        _save(tmp_path, "v1")
        _save(tmp_path, "v2")
        activate("v2", root=tmp_path)
        assert active_version(tmp_path) == "v2"
        assert load_model(root=tmp_path)[1].version == "v2"

    def test_no_active_model_is_an_error(self, tmp_path):
        with pytest.raises(RegistryError, match="no active model"):
            load_model(root=tmp_path)

    def test_a_broken_version_cannot_be_activated(self, tmp_path):
        _save(tmp_path, "v1")
        activate("v1", root=tmp_path)
        (tmp_path / "v2").mkdir()
        (tmp_path / "v2" / "manifest.json").write_text(
            (tmp_path / "v1" / "manifest.json").read_text().replace('"v1"', '"v2"'))
        with pytest.raises(RegistryError, match="no model.pkl"):
            activate("v2", root=tmp_path)
        assert active_version(tmp_path) == "v1"          # serving is untouched

    def test_unknown_version(self, tmp_path):
        with pytest.raises(RegistryError, match="no manifest"):
            load_model("ghost", root=tmp_path)


class TestSkewTripwire:
    def test_a_model_needing_a_feature_serving_does_not_compute_is_refused(self, tmp_path):
        """The skew tripwire after T-16: a subset is fine, an unknown feature is not."""
        _save(tmp_path)
        path = tmp_path / "v1" / "manifest.json"
        data = json.loads(path.read_text())
        data["feature_order"] = BASELINE_FEATURES + ["zodiac_sign"]
        path.write_text(json.dumps(data))
        with pytest.raises(FeatureOrderMismatch, match="serving does not compute.*zodiac_sign"):
            load_model("v1", root=tmp_path)

    def test_a_manifest_reordering_the_fitted_columns_is_refused(self, tmp_path):
        """Every feature is known to serving, but not in the order the model was fitted on."""
        _save(tmp_path)
        path = tmp_path / "v1" / "manifest.json"
        data = json.loads(path.read_text())
        data["feature_order"] = list(reversed(BASELINE_FEATURES))
        path.write_text(json.dumps(data))
        with pytest.raises(FeatureOrderMismatch, match="fitted on"):
            load_model("v1", root=tmp_path)

    def test_a_model_using_a_subset_of_serving_features_loads(self, tmp_path):
        """T-16 grew FEATURE_ORDER to 42; a model trained on fewer keeps working."""
        assert set(BASELINE_FEATURES) < set(FEATURE_ORDER)
        _save(tmp_path)
        model, manifest = load_model("v1", root=tmp_path)
        assert manifest.feature_order == BASELINE_FEATURES

    def test_a_model_on_the_full_feature_set_loads(self, tmp_path):
        rows = [[i % 7 for i in range(len(FEATURE_ORDER))], [i % 5 for i in range(len(FEATURE_ORDER))]] * 4
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(
            pd.DataFrame(rows, columns=FEATURE_ORDER), [0, 1] * 4)
        _save(tmp_path, model=model)
        assert load_model("v1", root=tmp_path)[1].feature_order == FEATURE_ORDER

    def test_model_fitted_on_other_columns_is_refused_at_save(self, tmp_path):
        stale = _fit(columns=["amount", "amount_deviation", "is_new_location", "is_flagged_device", "velocity"])
        with pytest.raises(FeatureOrderMismatch, match="serving does not compute.*'velocity'"):
            _save(tmp_path, model=stale)
        assert not (tmp_path / "v1").exists()

    def test_model_fitted_on_other_columns_is_refused_at_load(self, tmp_path):
        """Someone swapped model.pkl by hand; the manifest no longer describes it."""
        _save(tmp_path)
        stale = _fit(columns=["amount", "amount_deviation", "is_new_location", "is_flagged_device", "velocity"])
        joblib.dump(stale, tmp_path / "v1" / "model.pkl")
        with pytest.raises(FeatureOrderMismatch, match="fitted on"):
            load_model("v1", root=tmp_path)

    def test_model_without_feature_names_is_refused(self, tmp_path):
        """Fitted on a bare array, the order cannot be verified at all."""
        bare = RandomForestClassifier(n_estimators=5, random_state=0).fit([[1, 1, 0, 0, 0], [9, 9, 1, 1, 9]], [0, 1])
        with pytest.raises(FeatureOrderMismatch, match="no feature names"):
            _save(tmp_path, model=bare)

    def test_manifest_naming_another_version_is_refused(self, tmp_path):
        _save(tmp_path, "v1")
        (tmp_path / "v1").rename(tmp_path / "v9")
        with pytest.raises(RegistryError, match="names version v1"):
            load_model("v9", root=tmp_path)


class TestShippedModel:
    def test_the_active_shipped_model_loads_and_matches_serving(self):
        model, manifest = load_model()
        assert manifest.feature_order == BASELINE_FEATURES
        assert set(manifest.feature_order) <= set(FEATURE_ORDER)
        assert manifest.version == MODEL_VERSION == MANIFEST.version

    def test_shipped_metric_is_labelled_in_sample(self):
        """20 rows, fitted and scored on the same data: never present that as a holdout AUC."""
        assert set(MANIFEST.metrics) == {"in_sample_auc"}
        assert MANIFEST.training_rows == 20


class TestVersionStamping:
    def test_decisions_and_audit_rows_carry_the_model_version(self, client, auth_headers, db_session):
        from app import models

        txn = client.post("/transactions/", json={"location": "Lahore", "amount": 100.0, "device_id": "d1"},
                          headers=auth_headers).json()
        assert txn["model_version"] == MODEL_VERSION
        db_session.expire_all()
        audit = db_session.query(models.DecisionAudit).filter_by(transaction_id=txn["id"]).one()
        assert audit.model_version == MODEL_VERSION


def _start_api(extra_env):
    """Import the API in a fresh interpreter, as the server would at startup."""
    env = {**os.environ, "SECRET_KEY": "x", "DATABASE_URL": "sqlite:///:memory:", **extra_env}
    return subprocess.run([sys.executable, "-c", "import app.main"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


class TestStartupRefusesASkewedModel:
    def test_api_will_not_start_on_a_mismatched_model(self, tmp_path):
        _save(tmp_path)
        activate("v1", root=tmp_path)
        path = tmp_path / "v1" / "manifest.json"
        data = json.loads(path.read_text())
        data["feature_order"] = BASELINE_FEATURES + ["zodiac_sign"]
        path.write_text(json.dumps(data))

        result = _start_api({"MODEL_REGISTRY_DIR": str(tmp_path)})
        assert result.returncode != 0
        assert "FeatureOrderMismatch" in result.stderr

    def test_api_starts_on_a_healthy_registry(self, tmp_path):
        _save(tmp_path)
        activate("v1", root=tmp_path)
        result = _start_api({"MODEL_REGISTRY_DIR": str(tmp_path)})
        assert result.returncode == 0, result.stderr
