from datetime import datetime, timezone

from .features import FeatureVector, UserAggregates, compute_features
from .repositories.transaction_repo import get_device_user_count, get_user_aggregates
from .scoring import ScoreBreakdown, score
from .ML.models import MODEL_VERSION, predict_fraud


def build_feature_vector(
    transaction, aggregates: UserAggregates, device_user_count: int
) -> FeatureVector:
    """Serving-side feature path. Aggregates come from SQL, never from a row scan."""
    return compute_features(
        amount=transaction.amount,
        location=transaction.location,
        aggregates=aggregates,
        device_user_count=device_user_count,
    )


def score_features(fv: FeatureVector) -> ScoreBreakdown:
    """Run the rules and the model over an already-computed feature vector."""
    _, probability = predict_fraud(fv)
    return score(fv, probability, MODEL_VERSION)


def score_transaction(db, transaction, user_id: int, now: datetime | None = None) -> ScoreBreakdown:
    """Score a transaction and return the full breakdown behind the number."""
    now = now or datetime.now(timezone.utc)
    aggregates = get_user_aggregates(db, user_id, now, transaction.location)
    device_user_count = get_device_user_count(db, transaction.device_id)
    return score_features(build_feature_vector(transaction, aggregates, device_user_count))


def calculate_risk(db, transaction, user_id: int) -> float:
    """Final risk score in [0, 1]. Callers needing the rule trace use score_transaction."""
    return score_transaction(db, transaction, user_id).final_score
