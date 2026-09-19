from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas
from ..audit import record_decision
from ..fraud_detection import score_transaction
from ..dependencies import get_db, require_customer
from ..policy import Decision, PolicyContext, decide, load_policy_config
from ..models import CaseSource
from ..services.case_service import open_case
from ..services.fraud_services import get_risk_level

router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"]
)


@router.post("/", response_model=schemas.TransactionResponse)
def create_transaction(
    transaction: schemas.TransactionCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_customer)
):
    breakdown  = score_transaction(db, transaction, current_user.id)
    risk_score = breakdown.final_score
    risk_level = get_risk_level(risk_score)
    policy     = load_policy_config()
    context    = PolicyContext(amount=transaction.amount)
    decision   = decide(risk_score, context, policy)
    predicted_fraud = decision == Decision.REJECT

    new_transaction = models.Transaction(
        user_id=current_user.id,
        location=transaction.location,
        amount=transaction.amount,
        device_id=transaction.device_id,
        predicted_fraud=predicted_fraud,
        risk_score=risk_score,
        risk_level=risk_level,
        decision=decision.value,
        policy_version=policy.version,
        model_version=breakdown.model_version,
    )

    db.add(new_transaction)
    # Flush for the id. The audit row, and the case for a REVIEW, go in the same
    # database transaction: a decision without its audit record, or a REVIEW
    # without a case, must never be persisted (P6, A13).
    db.flush()
    record_decision(db, new_transaction, breakdown, decision, policy, context)
    if decision == Decision.REVIEW:
        open_case(db, new_transaction, CaseSource.POLICY_REVIEW)
    db.commit()
    db.refresh(new_transaction)
    return new_transaction


@router.get("/", response_model=List[schemas.TransactionResponse])
def list_transactions(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_customer)
):
    """Return all transactions for the logged-in user."""
    return (
        db.query(models.Transaction)
        .filter(models.Transaction.user_id == current_user.id)
        .all()
    )


@router.get("/{transaction_id}", response_model=schemas.TransactionResponse)
def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_customer)
):
    transaction = (
        db.query(models.Transaction)
        .filter(
            models.Transaction.id == transaction_id,
            models.Transaction.user_id == current_user.id
        )
        .first()
    )
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction
