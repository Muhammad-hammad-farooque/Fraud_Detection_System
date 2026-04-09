from datetime import datetime, timedelta, timezone
from .services.fraud_services import detect_device_fraud
from .ML.models import predict_fraud

def calculate_risk(db, transaction, user_transactions):
    risk_score = 0.0

    # Rule 1: Absolute high-amount threshold
    if transaction.amount > 5000:
        risk_score += 0.4

    # Rule 2: Amount far above the user's own average
    amount_deviation = 1.0
    if user_transactions:
        avg_amount = sum(t.amount for t in user_transactions) / len(user_transactions)
        if avg_amount > 0:
            amount_deviation = transaction.amount / avg_amount
        if amount_deviation > 3:
            risk_score += 0.4

    # Rule 3: Transaction from a location the user has never used
    known_locations = {t.location for t in user_transactions}
    is_new_location = 0 if transaction.location in known_locations else 1
    if is_new_location:
        risk_score += 0.2

    # Rule 4: Device shared by 3+ distinct users
    is_flagged_device = 1 if detect_device_fraud(db, transaction.device_id) else 0
    if is_flagged_device:
        risk_score += 0.3

    # Rule 5: Rapid transaction velocity — 5+ transactions in the last 2 minutes
    now = datetime.now(timezone.utc)
    recent = [
        t for t in user_transactions
        if (t.created_at.replace(tzinfo=timezone.utc) if t.created_at.tzinfo is None else t.created_at)
        >= now - timedelta(seconds=120)
    ]
    velocity = len(recent)
    if velocity >= 5:
        risk_score += 0.5

    # Rule 6: ML model boost
    _, probability = predict_fraud(
        amount=transaction.amount,
        amount_deviation=amount_deviation,
        is_new_location=is_new_location,
        is_flagged_device=is_flagged_device,
        velocity=velocity,
    )
    risk_score += probability * 0.3

    return min(risk_score, 1.0)
