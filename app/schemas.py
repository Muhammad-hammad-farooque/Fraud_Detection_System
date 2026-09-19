from pydantic import BaseModel, EmailStr, ConfigDict, Field, IPvAnyAddress, field_validator, model_validator
from datetime import datetime

from .models import Channel, Role

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
def _looks_like_card_number(value: str) -> bool:
    """13-19 digits passing the Luhn check: a primary account number."""
    digits = value.replace(" ", "").replace("-", "")
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


class TransactionCreate(BaseModel):
    location: str
    amount: float
    device_id: str

    # Payment context (T-15). Every field is optional so existing clients keep
    # working; the engine does not read them until T-16.
    merchant_id: str | None = Field(default=None, min_length=1, max_length=64)
    merchant_category: str | None = Field(default=None, pattern=r"^\d{4}$",
                                          description="ISO 18245 merchant category code")
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$", description="ISO 4217 code")
    channel: Channel | None = None
    ip_address: IPvAnyAddress | None = None
    card_token: str | None = Field(default=None, min_length=1, max_length=64)
    external_txn_id: str | None = Field(default=None, min_length=1, max_length=128)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    @field_validator("card_token")
    @classmethod
    def _not_a_card_number(cls, value: str | None) -> str | None:
        # PCI DSS: a primary account number must never reach this database.
        if value is not None and _looks_like_card_number(value):
            raise ValueError("card_token must be a token from the card vault, not a card number")
        return value

    @model_validator(mode="after")
    def _coordinates_come_in_pairs(self) -> "TransactionCreate":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be sent together")
        return self

    def orm_fields(self) -> dict:
        """The payment-context fields as the ORM stores them."""
        return {
            "merchant_id": self.merchant_id,
            "merchant_category": self.merchant_category,
            "currency": self.currency,
            "channel": self.channel.value if self.channel else None,
            "ip_address": str(self.ip_address) if self.ip_address else None,
            "card_token": self.card_token,
            "external_txn_id": self.external_txn_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }

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
    merchant_id: str | None = None
    merchant_category: str | None = None
    currency: str | None = None
    channel: str | None = None
    ip_address: str | None = None
    card_token: str | None = None
    external_txn_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
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
