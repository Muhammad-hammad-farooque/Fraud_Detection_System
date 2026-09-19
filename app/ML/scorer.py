"""Single-transaction scoring without the per-call DataFrame overhead (T-19).

Scoring one transaction through scikit-learn's public API builds a one-row
DataFrame, then validates and converts it twice - once in the calibration
wrapper, once in LightGBM's. On the shipped model that overhead is ~6 ms per
decision against 0.27 ms for the model's own arithmetic, and it pushes p99
past the 10 ms budget.

build_scorer returns a function from FeatureVector to calibrated probability
that goes straight to the booster and applies the fitted calibrator itself.
Columns are still selected by name, in the order the manifest lists, so the
train/serve skew protection of T-16 is unchanged. Anything the fast path does
not recognise falls back to predict_proba. tests/test_calibrated_model.py
holds the fast path to within 1e-12 of predict_proba.
"""
from collections.abc import Callable
from typing import Any

import numpy as np

from ..features import FeatureVector


def _fast_parts(model: Any):
    """(booster, calibrator) for a CalibratedClassifierCV over a frozen LightGBM
    classifier with a single calibrator, else None."""
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if not calibrated or len(calibrated) != 1:
        return None
    member = calibrated[0]
    if len(getattr(member, "calibrators", [])) != 1:
        return None
    estimator = getattr(member.estimator, "estimator", member.estimator)      # unwrap FrozenEstimator
    booster = getattr(estimator, "booster_", None)
    if booster is None or getattr(estimator, "objective_", None) != "binary":
        return None
    return booster, member.calibrators[0]


def build_scorer(model: Any, feature_order: list[str]) -> Callable[[FeatureVector], float]:
    """A function from a feature vector to the model's fraud probability."""
    columns = list(feature_order)
    parts = _fast_parts(model)

    if parts is None:
        def slow(fv: FeatureVector) -> float:
            return float(model.predict_proba(fv.to_frame(columns=columns))[0][1])
        return slow

    booster, calibrator = parts

    def fast(fv: FeatureVector) -> float:
        row = np.array([[float(getattr(fv, name)) for name in columns]])
        # The calibrator was fitted on what scikit-learn fed it: LGBMClassifier
        # has a decision_function, which CalibratedClassifierCV prefers over
        # predict_proba, so that is the raw log-odds - not the probability.
        # Feeding it probabilities returned nonsense; the parity test caught it.
        raw = booster.predict(row, raw_score=True)
        return float(np.clip(calibrator.predict(raw)[0], 0.0, 1.0))

    return fast
