"""Serving-side inference over the active registered model.

The active model is loaded once, when the API starts. load_model refuses to
return a model whose feature order differs from FEATURE_ORDER, so a skewed
model stops the API at startup rather than scoring silently wrong. Activating
a different version takes effect on the next restart.
"""
from ..features import FeatureVector
from .registry import load_model

model, MANIFEST = load_model()
MODEL_VERSION = MANIFEST.version


def predict_fraud(fv: FeatureVector) -> tuple[int, float]:
    """Returns (prediction, fraud_probability) for one feature vector.

    The vector is passed as a DataFrame whose column order is FEATURE_ORDER, so
    the model matches features by name rather than silently by position.

    The probability is a RandomForest predict_proba output and is NOT calibrated:
    a score of 0.7 does not mean 70% of such transactions are fraud. It is
    consumed numerically by app/scoring.py regardless, which T-19 fixes by
    swapping in a calibrated gradient-boosted model (A10).
    """
    frame = fv.to_frame()
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return prediction, probability
