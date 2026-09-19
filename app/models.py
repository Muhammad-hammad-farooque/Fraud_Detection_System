from sqlalchemy import JSON, Column, Index, Integer, Float, ForeignKey, String, DateTime, Boolean, UniqueConstraint, event
from enum import StrEnum
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime, timezone

class Channel(StrEnum):
    WEB    = "web"
    MOBILE = "mobile"
    POS    = "pos"      # card present, at a terminal
    ATM    = "atm"


class Role(StrEnum):
    CUSTOMER = "CUSTOMER"
    ANALYST  = "ANALYST"
    ADMIN    = "ADMIN"


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    # Never set from a request body: registration always yields CUSTOMER, and
    # only an admin (or scripts/set_role.py) can change it.
    role = Column(String, nullable=False, default=Role.CUSTOMER.value,
                  server_default=Role.CUSTOMER.value)
    transactions = relationship("Transaction", back_populates="user")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    location = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    device_id = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    # The model's own output. NEVER a label: ground truth lives in
    # TransactionOutcome, which is what retraining and monitoring read (A4).
    predicted_fraud = Column(Boolean, default=False)
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String, default="LOW")
    decision = Column(String, default="ALLOW")

    # Payment context (T-15). All nullable: older rows and clients that do not
    # send them stay valid. T-16 builds merchant, channel, network and
    # geo-velocity features on these.
    merchant_id       = Column(String(64), nullable=True)
    merchant_category = Column(String(4), nullable=True)     # ISO 18245 MCC, e.g. "5411"
    currency          = Column(String(3), nullable=True)     # ISO 4217, e.g. "PKR"
    channel           = Column(String(8), nullable=True)     # Channel
    ip_address        = Column(String(45), nullable=True)    # IPv4 or IPv6
    card_token        = Column(String(64), nullable=True)    # a token, never a card number
    external_txn_id   = Column(String(128), nullable=True)   # the processor's own reference
    latitude          = Column(Float, nullable=True)
    longitude         = Column(Float, nullable=True)
    policy_version = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    # Client-supplied retry key (A15). A repeat of the same key by the same
    # customer returns this row instead of creating and scoring a new one.
    idempotency_key = Column(String(255), nullable=True)
    # Hash of the request body the key was first used with. The same key sent
    # with a different payment is a client bug, and is refused rather than
    # silently answered with the old payment.
    idempotency_fingerprint = Column(String(64), nullable=True)
    # `decision` is what the engine decided and is never rewritten - the audit
    # trail depends on it. When an analyst resolves a review, the outcome lands
    # here instead: ALLOW if the transaction was legitimate, REJECT if fraud.
    resolved_decision = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        # Every scoring query filters on exactly this pair.
        Index("ix_txn_user_created", "user_id", "created_at"),
        # One transaction per customer per key. NULL keys never collide, so
        # requests without the header are unaffected.
        UniqueConstraint("user_id", "idempotency_key", name="uq_txn_user_idempotency_key"),
    )

    user = relationship("User", back_populates="transactions")
    claims = relationship("Claim", back_populates="transaction")
    outcome = relationship("TransactionOutcome", back_populates="transaction", uselist=False)
    case = relationship("Case", back_populates="transaction", uselist=False)
    step_up = relationship("StepUpChallenge", back_populates="transaction", uselist=False)

class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    reason = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="PENDING")  # PENDING, APPROVED, REJECTED, MANUAL_REVIEW
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    transaction = relationship("Transaction", back_populates="claims")


class OutcomeSource(StrEnum):
    ANALYST    = "ANALYST"       # investigator determination
    CHARGEBACK = "CHARGEBACK"    # authoritative, arrives 30-120 days later
    CUSTOMER   = "CUSTOMER"      # self-reported, weaker
    EXPLORATION = "EXPLORATION"  # random holdout allowed through for unbiased labels


class TransactionOutcome(Base):
    """Confirmed ground truth for one transaction.

    Deliberately separate from Claim: a customer dispute is a request, not a
    fraud determination. Only an investigator, a chargeback, or a deliberate
    exploration holdout produces a label that may be trained on.
    """
    __tablename__ = "transaction_outcomes"

    id                 = Column(Integer, primary_key=True)
    transaction_id     = Column(Integer, ForeignKey("transactions.id"), unique=True, nullable=False)
    is_fraud_confirmed = Column(Boolean, nullable=False)
    source             = Column(String, nullable=False)
    confirmed_by       = Column(Integer, ForeignKey("users.id"), nullable=True)
    confirmed_at       = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    notes              = Column(String, nullable=True)

    transaction = relationship("Transaction", back_populates="outcome")


class CaseStatus(StrEnum):
    OPEN     = "OPEN"
    ASSIGNED = "ASSIGNED"
    RESOLVED = "RESOLVED"


class CaseSource(StrEnum):
    POLICY_REVIEW  = "POLICY_REVIEW"   # the policy layer decided REVIEW
    CLAIM_REVIEW   = "CLAIM_REVIEW"    # claim verification returned MANUAL_REVIEW
    STEP_UP_FAILED = "STEP_UP_FAILED"  # the customer exhausted their step-up attempts


class Case(Base):
    """One unit of analyst work: a transaction awaiting a human determination.

    Resolving a case is what writes a TransactionOutcome, which is the only
    label source retraining and monitoring accept.
    """
    __tablename__ = "cases"

    id             = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), unique=True, nullable=False)
    claim_id       = Column(Integer, ForeignKey("claims.id"), nullable=True)
    source         = Column(String, nullable=False)
    status         = Column(String, nullable=False, default=CaseStatus.OPEN.value)
    priority       = Column(String, nullable=False)
    created_at     = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    sla_due_at     = Column(DateTime(timezone=True), nullable=False)
    assigned_to    = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_at    = Column(DateTime(timezone=True), nullable=True)
    resolved_by    = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_at    = Column(DateTime(timezone=True), nullable=True)
    notes          = Column(String, nullable=True)

    __table_args__ = (Index("ix_case_status_created", "status", "created_at"),)

    transaction = relationship("Transaction", back_populates="case")
    claim = relationship("Claim")


class DecisionAudit(Base):
    """Immutable record of one scoring decision (design principle P6).

    Holds everything needed to reproduce the decision from this row alone:
    the full feature vector, every rule that fired with its weight, the model
    output, and - beyond the original contract - the scoring and policy
    parameters in force at the time. Rule weights and thresholds become
    configurable in T-13, so replaying with today's settings would silently
    produce a different answer.

    Append-only. The ORM refuses updates and deletes below, and
    migrations/004_decision_audits.sql adds a trigger that does the same in
    PostgreSQL.
    """
    __tablename__ = "decision_audits"

    id             = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False, index=True)
    feature_vector = Column(JSON, nullable=False)   # full input, for replay
    rule_hits      = Column(JSON, nullable=False)   # [{rule_id, weight}, ...]
    rule_score     = Column(Float, nullable=False)
    model_prob     = Column(Float, nullable=False)
    final_score    = Column(Float, nullable=False)
    decision       = Column(String, nullable=False)
    model_version  = Column(String, nullable=False)
    policy_version = Column(String, nullable=False)
    scoring_params = Column(JSON, nullable=False)   # {total_rule_weight, w_rules, w_model}
    policy_config  = Column(JSON, nullable=False)   # thresholds in force
    policy_context = Column(JSON, nullable=False)   # amount, tier, account age
    created_at     = Column(DateTime(timezone=True), nullable=False,
                            default=lambda: datetime.now(timezone.utc))


class AuditImmutableError(Exception):
    """Raised on any attempt to modify or remove a DecisionAudit row."""


@event.listens_for(DecisionAudit, "before_update")
def _refuse_audit_update(mapper, connection, target):
    raise AuditImmutableError("decision_audits is append-only; rows cannot be updated")


@event.listens_for(DecisionAudit, "before_delete")
def _refuse_audit_delete(mapper, connection, target):
    raise AuditImmutableError("decision_audits is append-only; rows cannot be deleted")


class ChallengeStatus(StrEnum):
    PENDING = "PENDING"
    PASSED  = "PASSED"
    FAILED  = "FAILED"    # attempts exhausted: held for an analyst
    EXPIRED = "EXPIRED"


class StepUpChallenge(Base):
    """Extra authentication requested by a STEP_UP decision (T-14b, P4).

    The code itself is never stored: only an HMAC of it, keyed with the server
    secret. Passing a challenge is not a fraud label - one-time codes can be
    phished or SIM-swapped - so resolving one writes no TransactionOutcome.
    """
    __tablename__ = "step_up_challenges"

    id             = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), unique=True, nullable=False)
    method         = Column(String, nullable=False)          # "OTP"; pluggable for 3-D Secure
    code_hash      = Column(String(64), nullable=False)
    status         = Column(String, nullable=False, default=ChallengeStatus.PENDING.value)
    attempts       = Column(Integer, nullable=False, default=0)
    max_attempts   = Column(Integer, nullable=False)
    created_at     = Column(DateTime(timezone=True), nullable=False,
                            default=lambda: datetime.now(timezone.utc))
    expires_at     = Column(DateTime(timezone=True), nullable=False)
    resolved_at    = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_step_up_status_expires", "status", "expires_at"),)

    transaction = relationship("Transaction", back_populates="step_up")

    @property
    def attempts_remaining(self) -> int:
        return max(self.max_attempts - self.attempts, 0)
