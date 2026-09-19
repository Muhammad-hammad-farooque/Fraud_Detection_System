import hashlib
import json
from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from .. import models, schemas
from ..audit import record_decision
from ..fraud_detection import score_transaction
from ..dependencies import get_db, require_customer
from ..policy import Decision, PolicyContext, decide, load_policy_config
from ..models import CaseSource
from ..services import step_up
from ..services.case_service import open_case
from ..services.step_up import StepUpError
from ..services.fraud_services import get_risk_level

router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"]
)


REPLAY_HEADER = "Idempotent-Replayed"


def request_fingerprint(transaction: schemas.TransactionCreate) -> str:
    """Stable hash of the fields that define the payment."""
    canonical = json.dumps(transaction.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_by_idempotency_key(db: Session, user_id: int, key: str) -> models.Transaction | None:
    return (
        db.query(models.Transaction)
        .filter(models.Transaction.user_id == user_id, models.Transaction.idempotency_key == key)
        .first()
    )


def _replay(existing: models.Transaction, fingerprint: str, response: Response) -> models.Transaction:
    """Answer a repeated key with the stored transaction, or refuse a reused one."""
    if existing.idempotency_fingerprint != fingerprint:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key was already used for a different request",
        )
    response.headers[REPLAY_HEADER] = "true"
    return existing


@router.post("/", response_model=schemas.TransactionResponse)
def create_transaction(
    transaction: schemas.TransactionCreate,
    response: Response,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_customer),
    idempotency_key: str | None = Header(
        default=None, alias="Idempotency-Key", min_length=1, max_length=255,
        description="Client-generated key; a retry with the same key returns the original "
                    "transaction instead of creating and scoring a new one.",
    ),
):
    fingerprint = request_fingerprint(transaction) if idempotency_key else None
    if idempotency_key:
        existing = find_by_idempotency_key(db, current_user.id, idempotency_key)
        if existing is not None:
            # Not rescored: a retry must not write a second audit row, open a
            # second case, or count towards the velocity rule (A15).
            return _replay(existing, fingerprint, response)

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
        idempotency_key=idempotency_key,
        idempotency_fingerprint=fingerprint,
    )

    db.add(new_transaction)
    try:
        # Flush for the id. The audit row, and the case for a REVIEW, go in the
        # same database transaction: a decision without its audit record, or a
        # REVIEW without a case, must never be persisted (P6, A13). A duplicate
        # idempotency key surfaces here, at the flush, not at the commit.
        db.flush()
        record_decision(db, new_transaction, breakdown, decision, policy, context)
        if decision == Decision.REVIEW:
            open_case(db, new_transaction, CaseSource.POLICY_REVIEW)
        challenge_code = None
        if decision == Decision.STEP_UP:
            # A STEP_UP without a challenge would be a dead end (A13).
            challenge, challenge_code = step_up.issue_challenge(db, new_transaction)
        db.commit()
    except IntegrityError:
        # Two requests with the same key raced past the lookup. The unique
        # constraint let the other one win; this one's transaction, audit row
        # and case are rolled back together, and the winner is returned.
        db.rollback()
        if idempotency_key is None:
            raise
        existing = find_by_idempotency_key(db, current_user.id, idempotency_key)
        if existing is None:
            raise
        return _replay(existing, fingerprint, response)
    db.refresh(new_transaction)
    if challenge_code is None:
        return new_transaction

    # Deliver only now that the payment is committed: a rolled-back payment
    # must never send the customer a code.
    step_up.get_sender().send(current_user, new_transaction.step_up, challenge_code)
    body = schemas.TransactionResponse.model_validate(new_transaction)
    if step_up.load_step_up_config().dev_echo:
        body.step_up.dev_code = challenge_code
    return body


@router.post("/{transaction_id}/step-up", response_model=schemas.TransactionResponse)
def verify_step_up(
    transaction_id: int,
    body: schemas.StepUpVerify,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_customer)
):
    """Answer the step-up challenge on one of your own transactions.

    Returns the transaction with its challenge: PASSED allows the payment,
    EXPIRED rejects it, and FAILED holds it for an analyst.
    """
    transaction = (
        db.query(models.Transaction)
        .filter(models.Transaction.id == transaction_id,
                models.Transaction.user_id == current_user.id)
        .first()
    )
    if not transaction or transaction.step_up is None:
        raise HTTPException(status_code=404, detail="No step-up challenge for this transaction")
    try:
        step_up.verify(db, transaction.step_up, body.code)
    except StepUpError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    db.commit()
    db.refresh(transaction)
    return transaction


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
