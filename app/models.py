from sqlalchemy import Column, Index, Integer, Float, ForeignKey, String, DateTime, Boolean
from enum import StrEnum
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime, timezone

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
    policy_version = Column(String, nullable=True)
    # `decision` is what the engine decided and is never rewritten - the audit
    # trail depends on it. When an analyst resolves a review, the outcome lands
    # here instead: ALLOW if the transaction was legitimate, REJECT if fraud.
    resolved_decision = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Every scoring query filters on exactly this pair.
    __table_args__ = (Index("ix_txn_user_created", "user_id", "created_at"),)

    user = relationship("User", back_populates="transactions")
    claims = relationship("Claim", back_populates="transaction")
    outcome = relationship("TransactionOutcome", back_populates="transaction", uselist=False)
    case = relationship("Case", back_populates="transaction", uselist=False)

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
    POLICY_REVIEW = "POLICY_REVIEW"   # the policy layer decided REVIEW
    CLAIM_REVIEW  = "CLAIM_REVIEW"    # claim verification returned MANUAL_REVIEW


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
