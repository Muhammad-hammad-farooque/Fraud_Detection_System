"""Serving-side inference over the active registered model.

The active model is loaded once, when the API starts. load_model refuses to
return a model whose feature order differs from FEATURE_ORDER, so a skewed
model stops the API at startup rather than scoring silently wrong. Activating
a different version takes effect on the next restart.
"""
from ..features import FeatureVector
from .registry import load_model
from .scorer import build_scorer

model, MANIFEST = load_model()
MODEL_VERSION = MANIFEST.version
_score = build_scorer(model, MANIFEST.feature_order)


def predict_fraud(fv: FeatureVector) -> tuple[int, float]:
    """Returns (prediction, fraud_probability) for one feature vector.

    The model receives exactly the columns its manifest lists, by name and in
    its own order, so a model trained on a subset of FEATURE_ORDER keeps working
    as features are added.

    Whether that probability is calibrated depends on the active model. Models
    trained since T-19 are a LightGBM booster wrapped in calibration, whose
    manifest carries a reliability curve; app/scoring.py blends the
    probability numerically and assumes it is one (A10). The forest trained in
    T-18, which still serves because LightGBM has not beaten it through the
    promotion gate, is NOT calibrated. The class is derived from the same
    probability, so each decision calls the model once.
    """
    probability = _score(fv)
    return int(probability >= 0.5), probability
