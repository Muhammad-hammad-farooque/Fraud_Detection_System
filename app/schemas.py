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

class StepUpInfo(BaseModel):
    """The step-up challenge on a STEP_UP transaction, if it has one."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    method: str
    status: str
    expires_at: datetime
    attempts_remaining: int
    # Development only (STEP_UP_DEV_ECHO=true), and only on the response that
    # created the challenge. In production the code arrives out of band.
    dev_code: str | None = None


class StepUpVerify(BaseModel):
    code: str = Field(min_length=1, max_length=16)


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
    model_version: str | None = None
    resolved_decision: str | None = None
    step_up: StepUpInfo | None = None
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


# ── Case ─────────────────────────────────────────────────────────
class CaseResponse(BaseModel):
    id: int
    transaction_id: int
    claim_id: int | None
    source: str
    status: str
    priority: str
    risk_score: float
    amount: float
    decision: str
    created_at: datetime
    sla_due_at: datetime
    sla_breached: bool
    assigned_to: int | None
    assigned_at: datetime | None
    resolved_by: int | None
    resolved_at: datetime | None
    notes: str | None


class CaseResolve(BaseModel):
    is_fraud_confirmed: bool
    notes: str | None = None


class OutcomeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    transaction_id: int
    is_fraud_confirmed: bool
    source: str
    confirmed_by: int | None
    confirmed_at: datetime
    notes: str | None


# ── Audit ────────────────────────────────────────────────────────
class AuditResponse(BaseModel):
    id: int
    transaction_id: int
    feature_vector: dict
    rule_hits: list[dict]
    rule_score: float
    model_prob: float
    final_score: float
    decision: str
    model_version: str
    policy_version: str
    scoring_params: dict
    policy_config: dict
    policy_context: dict
    created_at: datetime
    replay_matches: bool
    rules_still_agree: bool
