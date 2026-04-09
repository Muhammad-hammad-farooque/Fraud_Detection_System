from datetime import datetime, timezone
from .. import models

def get_risk_level(risk_score: float) -> str:
    if risk_score < 0.3:
        return "LOW"
    elif risk_score < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

def get_decision(risk_level: str) -> str:
    if risk_level == "LOW":
        return "ALLOW"
    elif risk_level == "MEDIUM":
        return "MANUAL_CHECK"
    else:
        return "REJECT"

def detect_device_fraud(db, device_id: str) -> bool:
    """Returns True if the device has been used by 3 or more distinct users."""
    users = (
        db.query(models.Transaction.user_id)
        .filter(models.Transaction.device_id == device_id)
        .distinct()
        .all()
    )
    return len(users) >= 3

def verify_claim(db, claim_data, transaction) -> str:
    """
    Three-step claim verification:
      1. Data Enrichment  — check user's claim history
      2. Hard Rules       — reject stale claims
      3. Pattern Matching — flag suspicious behaviour for human review
    """
    user_id = transaction.user_id

    # Step 1: Serial claimer check
    previous_claims_count = (
        db.query(models.Claim)
        .join(models.Transaction)
        .filter(models.Transaction.user_id == user_id)
        .count()
    )
    if previous_claims_count > 3:
        return "REJECTED"

    # Step 2: Age — reject claims older than 90 days
    now = datetime.now(timezone.utc)
    tx_time = transaction.created_at
    if tx_time.tzinfo is None:
        tx_time = tx_time.replace(tzinfo=timezone.utc)
    if (now - tx_time).days > 90:
        return "REJECTED"

    # Step 3: Pattern matching — approve clean cases, flag grey areas
    if transaction.is_fraud is False and previous_claims_count == 0:
        return "APPROVED"

    return "MANUAL_REVIEW"
