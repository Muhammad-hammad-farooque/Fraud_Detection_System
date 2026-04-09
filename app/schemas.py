from pydantic import BaseModel, EmailStr, ConfigDict
from datetime import datetime

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
    is_fraud: bool
    risk_score: float
    risk_level: str
    decision: str
    created_at: datetime

# ── Claim ────────────────────────────────────────────────────────
class ClaimCreate(BaseModel):
    transaction_id: int
    reason: str
    amount: float

class ClaimResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    transaction_id: int
    reason: str
    amount: float
    status: str
    created_at: datetime
