"""
Unit tests for app/ML/models.py — inference contract (T-08).
"""
import warnings

import pytest

from app.features import FEATURE_ORDER, FeatureVector
from app.ML.models import MODEL_VERSION, model, predict_fraud

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

    def test_model_feature_names_match_feature_order(self):
        assert list(model.feature_names_in_) == FEATURE_ORDER

    def test_obvious_fraud_scores_above_obvious_legitimate(self):
        assert predict_fraud(LOUD)[1] > predict_fraud(QUIET)[1]

    def test_model_version_is_stamped(self):
        assert MODEL_VERSION
