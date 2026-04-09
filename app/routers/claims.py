from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas
from ..dependencies import get_db, get_current_user
from ..services.fraud_services import verify_claim

router = APIRouter(
    prefix="/claims",
    tags=["Claims"]
)


@router.get("/", response_model=List[schemas.ClaimResponse])
def list_claims(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Return all claims filed by the logged-in user."""
    return (
        db.query(models.Claim)
        .join(models.Transaction)
        .filter(models.Transaction.user_id == current_user.id)
        .all()
    )


@router.post("/", response_model=schemas.ClaimResponse, status_code=201)
def create_claim(
    claim: schemas.ClaimCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """File a dispute claim on a transaction. Only the transaction owner can claim."""
    transaction = (
        db.query(models.Transaction)
        .filter(
            models.Transaction.id == claim.transaction_id,
            models.Transaction.user_id == current_user.id
        )
        .first()
    )
    if not transaction:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found or does not belong to you"
        )

    status = verify_claim(db, claim, transaction)

    new_claim = models.Claim(
        transaction_id=claim.transaction_id,
        reason=claim.reason,
        amount=claim.amount,
        status=status
    )

    db.add(new_claim)
    db.commit()
    db.refresh(new_claim)
    return new_claim


@router.get("/{claim_id}", response_model=schemas.ClaimResponse)
def get_claim(
    claim_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get a claim by ID. Only the transaction owner can view it."""
    claim = db.query(models.Claim).filter(models.Claim.id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    # Verify the claim belongs to the current user's transaction
    transaction = (
        db.query(models.Transaction)
        .filter(
            models.Transaction.id == claim.transaction_id,
            models.Transaction.user_id == current_user.id
        )
        .first()
    )
    if not transaction:
        raise HTTPException(status_code=403, detail="Access denied")

    return claim
