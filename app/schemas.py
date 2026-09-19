from pydantic import BaseModel, EmailStr, ConfigDict, Field
from datetime import datetime

from .models import Role

# ── User ─────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str
    role: str


class RoleUpdate(BaseModel):
    role: Role

# ── Auth ─────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ── Transaction ──────────────────────────────────────────────────
class TransactionCreate(BaseModel):
    location: str
    amount: float
    device_id: str

class TransactionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    location: str
    amount: float
    device_id: str
    predicted_fraud: bool
    risk_score: float
    risk_level: str
    decision: str
    policy_version: str | None = None
    created_at: datetime

# ── Claim ────────────────────────────────────────────────────────
class ClaimCreate(BaseModel):
    transaction_id: int
    reason: str
    # Must also be <= the disputed transaction's amount, which the router checks
    # once it has loaded the transaction (A14).
    amount: float = Field(gt=0)

class ClaimResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    transaction_id: int
    reason: str
    amount: float
    status: str
    created_at: datetime
