from sqlalchemy import Column, Index, Integer, Float, ForeignKey, String, DateTime, Boolean
from enum import StrEnum
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime, timezone

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Every scoring query filters on exactly this pair.
    __table_args__ = (Index("ix_txn_user_created", "user_id", "created_at"),)

    user = relationship("User", back_populates="transactions")
    claims = relationship("Claim", back_populates="transaction")
    outcome = relationship("TransactionOutcome", back_populates="transaction", uselist=False)

class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    reason = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="PENDING")  # PENDING, APPROVED, REJECTED, MANUAL_REVIEW
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    confirmed_at       = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    notes              = Column(String, nullable=True)

    transaction = relationship("Transaction", back_populates="outcome")
