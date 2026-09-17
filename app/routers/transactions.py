from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas
from ..fraud_detection import score_transaction
from ..dependencies import get_db, get_current_user
from ..policy import Decision, PolicyContext, decide, load_policy_config
from ..services.fraud_services import get_risk_level

router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"]
)


@router.post("/", response_model=schemas.TransactionResponse)
def create_transaction(
    transaction: schemas.TransactionCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    breakdown  = score_transaction(db, transaction, current_user.id)
    risk_score = breakdown.final_score
    risk_level = get_risk_level(risk_score)
    policy     = load_policy_config()
    decision   = decide(risk_score, PolicyContext(amount=transaction.amount), policy)
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
    )

    db.add(new_transaction)
    db.commit()
    db.refresh(new_transaction)
    return new_transaction


@router.get("/", response_model=List[schemas.TransactionResponse])
def list_transactions(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
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
    current_user: models.User = Depends(get_current_user)
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
