from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import models, schemas
from ..dependencies import get_db
from app.services.fraud_services import check_fraud

router = APIRouter(
    prefix="/transactions",
    tags=["Transaction"]
)

@router.post("/", response_model=schemas.TransactionResponce)
def create_transaction(transaction: schemas.TransactionResponce, db:Session=Depends(get_db)):
    user_history = db.query(models.Transaction).filter(models.Transaction.user_id==transaction.user_id).all()
    is_fraud = check_fraud(transaction, user_history)
    db_transaction=models.transaction(
        user_id=transaction.user_id,
        location=transaction.location,
        ammount=transaction.ammount,
        device_id=transaction.device_id,
        is_fraud=is_fraud
    )

    db.add(db_transaction)
    db.commit()
    db.refresh()
    return db_transaction
