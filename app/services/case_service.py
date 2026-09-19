"""Analyst case lifecycle: open, assign, resolve.

This is where the feedback loop closes. Resolving a case writes the
TransactionOutcome that retraining and monitoring read, so every function here
that changes state does so inside the caller's database transaction and never
commits on its own.
"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from .. import models
from ..features import as_utc
from ..models import CaseSource, CaseStatus, OutcomeSource, Role

HIGH_PRIORITY = "HIGH"
NORMAL_PRIORITY = "NORMAL"


@dataclass(frozen=True)
class CaseConfig:
    """Queue tuning. Read from the environment so ops can retune without a deploy."""
    high_priority_score: float
    high_priority_sla_hours: float
    normal_sla_hours: float


def load_case_config() -> CaseConfig:
    return CaseConfig(
        high_priority_score=float(os.getenv("CASE_HIGH_PRIORITY_SCORE", "0.6")),
        high_priority_sla_hours=float(os.getenv("CASE_HIGH_PRIORITY_SLA_HOURS", "4")),
        normal_sla_hours=float(os.getenv("CASE_NORMAL_SLA_HOURS", "24")),
    )


class CaseError(Exception):
    """A lifecycle rule was broken. `status_code` is the HTTP status it maps to."""

    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


def _priority_and_sla(risk_score: float, now: datetime, cfg: CaseConfig) -> tuple[str, datetime]:
    if risk_score >= cfg.high_priority_score:
        return HIGH_PRIORITY, now + timedelta(hours=cfg.high_priority_sla_hours)
    return NORMAL_PRIORITY, now + timedelta(hours=cfg.normal_sla_hours)


def open_case(
    db: Session,
    transaction: models.Transaction,
    source: CaseSource,
    claim: models.Claim | None = None,
    now: datetime | None = None,
    cfg: CaseConfig | None = None,
) -> models.Case:
    """Open a case for a transaction, or return the one it already has.

    A transaction has at most one case. A claim arriving on a transaction that is
    already under review attaches to the existing case rather than opening a
    second, so the analyst sees the dispute alongside the original flag.
    """
    existing = db.query(models.Case).filter(models.Case.transaction_id == transaction.id).first()
    if existing is not None:
        if claim is not None and existing.claim_id is None:
            existing.claim_id = claim.id
        return existing

    now = now or datetime.now(timezone.utc)
    priority, sla_due_at = _priority_and_sla(transaction.risk_score, now, cfg or load_case_config())
    case = models.Case(
        transaction_id=transaction.id,
        claim_id=claim.id if claim is not None else None,
        source=source.value,
        status=CaseStatus.OPEN.value,
        priority=priority,
        created_at=now,
        sla_due_at=sla_due_at,
    )
    db.add(case)
    return case


def is_sla_breached(case: models.Case, now: datetime | None = None) -> bool:
    """An unresolved case past its deadline. Resolved cases are never in breach."""
    if case.status == CaseStatus.RESOLVED.value:
        return False
    return (now or datetime.now(timezone.utc)) > as_utc(case.sla_due_at)


def list_queue(
    db: Session,
    status: CaseStatus | None = None,
    priority: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[models.Case]:
    """Cases ordered for work: highest risk first, then oldest first."""
    query = db.query(models.Case).join(
        models.Transaction, models.Transaction.id == models.Case.transaction_id
    )
    if status is not None:
        query = query.filter(models.Case.status == status.value)
    if priority is not None:
        query = query.filter(models.Case.priority == priority)
    return (
        query.order_by(models.Transaction.risk_score.desc(), models.Case.created_at.asc())
        .limit(limit)
        .offset(offset)
        .all()
    )


def assign(db: Session, case: models.Case, analyst: models.User, now: datetime | None = None) -> models.Case:
    """Claim a case. Re-assigning to the current holder is a no-op."""
    if case.status == CaseStatus.RESOLVED.value:
        raise CaseError("Case is already resolved")
    if case.assigned_to is not None and case.assigned_to != analyst.id:
        raise CaseError("Case is assigned to another analyst")
    if case.assigned_to == analyst.id:
        return case

    case.assigned_to = analyst.id
    case.assigned_at = now or datetime.now(timezone.utc)
    case.status = CaseStatus.ASSIGNED.value
    return case


def resolve(
    db: Session,
    case: models.Case,
    analyst: models.User,
    is_fraud_confirmed: bool,
    notes: str | None,
    now: datetime | None = None,
) -> models.TransactionOutcome:
    """Record the analyst's determination and close the case.

    Writes exactly one TransactionOutcome with source=ANALYST, sets the
    transaction's resolved_decision, settles any attached MANUAL_REVIEW claim,
    and marks the case RESOLVED - all in the caller's transaction.
    """
    if case.status == CaseStatus.RESOLVED.value:
        raise CaseError("Case is already resolved")
    if case.assigned_to != analyst.id and analyst.role != Role.ADMIN.value:
        raise CaseError("Assign the case to yourself before resolving it")

    existing = (
        db.query(models.TransactionOutcome)
        .filter(models.TransactionOutcome.transaction_id == case.transaction_id)
        .first()
    )
    if existing is not None:
        raise CaseError(
            f"Transaction already has a {existing.source} outcome; it cannot be relabelled here"
        )

    now = now or datetime.now(timezone.utc)
    outcome = models.TransactionOutcome(
        transaction_id=case.transaction_id,
        is_fraud_confirmed=is_fraud_confirmed,
        source=OutcomeSource.ANALYST.value,
        confirmed_by=analyst.id,
        confirmed_at=now,
        notes=notes,
    )
    db.add(outcome)

    case.transaction.resolved_decision = "REJECT" if is_fraud_confirmed else "ALLOW"

    # A customer disputing a transaction is saying "this was not me". Confirmed
    # fraud upholds the dispute; a legitimate transaction does not.
    if case.claim is not None and case.claim.status == "MANUAL_REVIEW":
        case.claim.status = "APPROVED" if is_fraud_confirmed else "REJECTED"

    case.status = CaseStatus.RESOLVED.value
    case.assigned_to = case.assigned_to or analyst.id
    case.resolved_by = analyst.id
    case.resolved_at = now
    case.notes = notes
    return outcome
