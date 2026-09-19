from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..dependencies import get_db, require_analyst

# Analyst routes read across every customer. T-11 adds the audit view and
# T-12 the case queue here. The /v1 prefix arrives with T-27.
router = APIRouter(
    prefix="/analyst",
    tags=["Analyst"]
)


@router.get("/transactions/{transaction_id}", response_model=schemas.TransactionResponse)
def get_any_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    analyst: models.User = Depends(require_analyst)
):
    """Look up any customer's transaction for review. Analysts and admins only."""
    transaction = db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction
