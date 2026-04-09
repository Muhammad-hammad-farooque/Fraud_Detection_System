from sqlalchemy import Column, Integer, Float, ForeignKey, String, DateTime, Boolean
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
    is_fraud = Column(Boolean, default=False)
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String, default="LOW")
    decision = Column(String, default="ALLOW")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="transactions")
    claims = relationship("Claim", back_populates="transaction")

class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    reason = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="PENDING")  # PENDING, APPROVED, REJECTED, MANUAL_REVIEW
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    transaction = relationship("Transaction", back_populates="claims")
