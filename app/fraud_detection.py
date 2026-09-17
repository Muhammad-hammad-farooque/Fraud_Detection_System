from datetime import datetime, timezone
from .features import FeatureVector, aggregates_from_history, compute_features
from .scoring import ScoreBreakdown, score
from .services.fraud_services import count_device_users
from .ML.models import MODEL_VERSION, predict_fraud


def build_feature_vector(db, transaction, user_transactions, now: datetime) -> FeatureVector:
    """Serving-side feature path: history + device lookup -> compute_features."""
    aggregates = aggregates_from_history(user_transactions, now)
    return compute_features(
        amount=transaction.amount,
        location=transaction.location,
        aggregates=aggregates,
        device_user_count=count_device_users(db, transaction.device_id),
    )


def score_transaction(db, transaction, user_transactions) -> ScoreBreakdown:
    """Score a transaction and return the full breakdown behind the number."""
    fv = build_feature_vector(db, transaction, user_transactions, datetime.now(timezone.utc))
    _, probability = predict_fraud(
        amount=fv.amount,
        amount_deviation=fv.amount_deviation,
        is_new_location=fv.is_new_location,
        is_flagged_device=fv.is_flagged_device,
        velocity=fv.velocity_2m,
    )
    return score(fv, probability, MODEL_VERSION)


def calculate_risk(db, transaction, user_transactions) -> float:
    """Final risk score in [0, 1]. Callers needing the rule trace use score_transaction."""
    return score_transaction(db, transaction, user_transactions).final_score
