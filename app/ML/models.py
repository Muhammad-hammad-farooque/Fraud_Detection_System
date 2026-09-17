import os

import joblib

from ..features import FeatureVector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

# Placeholder until the model registry lands (T-13): model.pkl carries no
# version, training date or metrics of its own.
MODEL_VERSION = "rf-baseline-unversioned"


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
