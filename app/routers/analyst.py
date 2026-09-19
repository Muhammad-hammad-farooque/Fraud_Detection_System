from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from .. import models, schemas
from ..audit import replay
from ..dependencies import get_db, require_analyst
from ..models import CaseStatus
from ..services import case_service
from ..services.case_service import CaseError

# Analyst routes read across every customer.
# The /v1 prefix arrives with T-27.
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


@router.get("/transactions/{transaction_id}/audit", response_model=List[schemas.AuditResponse])
def get_transaction_audit(
    transaction_id: int,
    db: Session = Depends(get_db),
    analyst: models.User = Depends(require_analyst)
):
    """The decision record for a transaction, each row replayed on the way out.

    replay_matches says the stored row still reproduces its own score and
    action; rules_still_agree says whether today's rules would fire the same
    way on the same input.
    """
    if not db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first():
        raise HTTPException(status_code=404, detail="Transaction not found")

    audits = (
        db.query(models.DecisionAudit)
        .filter(models.DecisionAudit.transaction_id == transaction_id)
        .order_by(models.DecisionAudit.created_at)
        .all()
    )
    responses = []
    for audit in audits:
        result = replay(audit)
        responses.append(schemas.AuditResponse(
            id=audit.id,
            transaction_id=audit.transaction_id,
            feature_vector=audit.feature_vector,
            rule_hits=audit.rule_hits,
            rule_score=audit.rule_score,
            model_prob=audit.model_prob,
            final_score=audit.final_score,
            decision=audit.decision,
            model_version=audit.model_version,
            policy_version=audit.policy_version,
            scoring_params=audit.scoring_params,
            policy_config=audit.policy_config,
            policy_context=audit.policy_context,
            created_at=audit.created_at,
            replay_matches=result.matches,
            rules_still_agree=result.rules_still_agree,
        ))
    return responses


def _case_response(case: models.Case) -> schemas.CaseResponse:
    txn = case.transaction
    return schemas.CaseResponse(
        id=case.id,
        transaction_id=case.transaction_id,
        claim_id=case.claim_id,
        source=case.source,
        status=case.status,
        priority=case.priority,
        risk_score=txn.risk_score,
        amount=txn.amount,
        decision=txn.decision,
        created_at=case.created_at,
        sla_due_at=case.sla_due_at,
        sla_breached=case_service.is_sla_breached(case),
        assigned_to=case.assigned_to,
        assigned_at=case.assigned_at,
        resolved_by=case.resolved_by,
        resolved_at=case.resolved_at,
        notes=case.notes,
    )


def _get_case(db: Session, case_id: int) -> models.Case:
    case = db.query(models.Case).filter(models.Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.get("/cases", response_model=List[schemas.CaseResponse])
def list_cases(
    status: CaseStatus | None = None,
    priority: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    analyst: models.User = Depends(require_analyst)
):
    """The review queue, highest risk first and then oldest first."""
    cases = case_service.list_queue(db, status=status, priority=priority, limit=limit, offset=offset)
    return [_case_response(case) for case in cases]


@router.get("/cases/{case_id}", response_model=schemas.CaseResponse)
def get_case(
    case_id: int,
    db: Session = Depends(get_db),
    analyst: models.User = Depends(require_analyst)
):
    return _case_response(_get_case(db, case_id))


@router.post("/cases/{case_id}/assign", response_model=schemas.CaseResponse)
def assign_case(
    case_id: int,
    db: Session = Depends(get_db),
    analyst: models.User = Depends(require_analyst)
):
    """Claim a case for yourself."""
    case = _get_case(db, case_id)
    try:
        case_service.assign(db, case, analyst)
    except CaseError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    db.commit()
    db.refresh(case)
    return _case_response(case)


@router.post("/cases/{case_id}/resolve", response_model=schemas.OutcomeResponse)
def resolve_case(
    case_id: int,
    body: schemas.CaseResolve,
    db: Session = Depends(get_db),
    analyst: models.User = Depends(require_analyst)
):
    """Record a fraud determination. Writes the TransactionOutcome label."""
    case = _get_case(db, case_id)
    try:
        outcome = case_service.resolve(db, case, analyst, body.is_fraud_confirmed, body.notes)
    except CaseError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    db.commit()
    db.refresh(outcome)
    return outcome
