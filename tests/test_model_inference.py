"""
Unit tests for app/ML/models.py — inference contract (T-08).
"""
import warnings

import pytest

from app.features import FEATURE_ORDER, FeatureVector
from app.ML.models import MANIFEST, MODEL_VERSION, model, predict_fraud

QUIET = FeatureVector(100.0, 1.0, 0, 0, 0)
LOUD = FeatureVector(9000.0, 90.0, 1, 1, 12)


class TestPredictFraud:
    def test_takes_a_feature_vector(self):
        prediction, probability = predict_fraud(QUIET)
        assert prediction in (0, 1)
        assert 0.0 <= probability <= 1.0

    def test_emits_no_feature_name_warning(self):
        """A11: a bare numpy array made sklearn warn and match by position."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            predict_fraud(LOUD)

    def test_model_feature_names_match_its_manifest_and_serving(self):
        assert list(model.feature_names_in_) == MANIFEST.feature_order
        assert set(MANIFEST.feature_order) <= set(FEATURE_ORDER)

    def test_the_model_receives_only_its_own_columns(self, monkeypatch):
        """FEATURE_ORDER has 42 features; the baseline model must still get its 5."""
        seen = []
        real = model.predict_proba
        monkeypatch.setattr(model, "predict_proba", lambda frame: seen.append(list(frame.columns)) or real(frame))
        predict_fraud(LOUD)
        assert seen                                              # predict() calls it too
        assert all(columns == MANIFEST.feature_order for columns in seen)

    def test_obvious_fraud_scores_above_obvious_legitimate(self):
        assert predict_fraud(LOUD)[1] > predict_fraud(QUIET)[1]

    def test_model_version_is_stamped(self):
        assert MODEL_VERSION
