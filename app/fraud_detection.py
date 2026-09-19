from datetime import datetime, timezone

from .features import FeatureVector, TransactionInput, compute_features
from .repositories.transaction_repo import (
    get_device_aggregates,
    get_population_aggregates,
    get_user_aggregates,
)
from .scoring import ScoreBreakdown, score
from .ML.models import MODEL_VERSION, predict_fraud


def features_at(db, candidate: TransactionInput, user_id: int) -> FeatureVector:
    """The feature vector for `candidate`, as of candidate.at.

    The only way features are computed - by live scoring at request time, and
    by retraining at each historical transaction's own timestamp. Four
    statements, whatever the history size - plus one on the first decision of
    each UTC day, to snapshot the population's amount distribution.
    """
    return compute_features(
        candidate,
        get_user_aggregates(db, user_id, candidate),
        get_device_aggregates(db, candidate.device_id, candidate.at),
        get_population_aggregates(db, candidate),
    )


def score_features(fv: FeatureVector) -> ScoreBreakdown:
    """Run the rules and the model over an already-computed feature vector."""
    _, probability = predict_fraud(fv)
    return score(fv, probability, MODEL_VERSION)


def score_transaction(db, transaction, user_id: int, now: datetime | None = None) -> ScoreBreakdown:
    """Score a transaction and return the full breakdown behind the number.

    `now` is the instant the decision is made; the caller should store the
    transaction with this same timestamp, so retraining reconstructs exactly
    the features this decision saw.
    """
    now = now or datetime.now(timezone.utc)
    return score_features(features_at(db, TransactionInput.from_request(transaction, at=now), user_id))


def calculate_risk(db, transaction, user_id: int) -> float:
    """Final risk score in [0, 1]. Callers needing the rule trace use score_transaction."""
    return score_transaction(db, transaction, user_id).final_score
