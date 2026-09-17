from datetime import datetime, timezone
from .features import FeatureVector, aggregates_from_history, compute_features
from .services.fraud_services import count_device_users
from .ML.models import predict_fraud


def build_feature_vector(db, transaction, user_transactions, now: datetime) -> FeatureVector:
    """Serving-side feature path: history + device lookup -> compute_features."""
    aggregates = aggregates_from_history(user_transactions, now)
    return compute_features(
        amount=transaction.amount,
        location=transaction.location,
        aggregates=aggregates,
        device_user_count=count_device_users(db, transaction.device_id),
    )


def calculate_risk(db, transaction, user_transactions):
    fv = build_feature_vector(db, transaction, user_transactions, datetime.now(timezone.utc))
    risk_score = 0.0

    # Rule 1: Absolute high-amount threshold
    if fv.amount > 5000:
        risk_score += 0.4

    # Rule 2: Amount far above the user's own average
    if fv.amount_deviation > 3:
        risk_score += 0.4

    # Rule 3: Transaction from a location the user has never used
    if fv.is_new_location:
        risk_score += 0.2

    # Rule 4: Device shared by 3+ distinct users
    if fv.is_flagged_device:
        risk_score += 0.3

    # Rule 5: Rapid transaction velocity — 5+ transactions in the last 2 minutes
    if fv.velocity_2m >= 5:
        risk_score += 0.5

    # Rule 6: ML model boost
    _, probability = predict_fraud(
        amount=fv.amount,
        amount_deviation=fv.amount_deviation,
        is_new_location=fv.is_new_location,
        is_flagged_device=fv.is_flagged_device,
        velocity=fv.velocity_2m,
    )
    risk_score += probability * 0.3

    return min(risk_score, 1.0)
