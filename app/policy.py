"""Policy layer: turns a risk score into an action.

Contains no feature or risk computation - it only reads the score that
app/scoring.py produced. Keeping the two apart means thresholds can be retuned
without revalidating or redeploying the model (design principle P1), and it is
what makes STEP_UP possible as a real action rather than a second reject (P4).

Bands come from the policy section of config/rules.yaml (T-13).
"""
from dataclasses import dataclass
from enum import StrEnum

from .config import get_rules_config


class Decision(StrEnum):
    ALLOW   = "ALLOW"
    STEP_UP = "STEP_UP"     # new: request additional authentication
    REVIEW  = "REVIEW"
    REJECT  = "REJECT"


@dataclass(frozen=True)
class PolicyContext:
    amount: float
    customer_tier: str = "standard"
    account_age_days: int = 0


@dataclass(frozen=True)
class PolicyConfig:
    """Loaded from config, NOT hardcoded. Risk teams retune this without a deploy."""
    version: str
    allow_below: float
    step_up_below: float
    review_below: float
    high_value_amount: float   # above this, thresholds tighten
    # Multipliers applied to every band. Below 1.0 the bands shift down, so the
    # same score lands on a stricter action. The defaults are the values in force
    # before T-13 made them configurable, so audit rows written before then still
    # replay exactly.
    high_value_tightening: float = 0.8
    trusted_tier_loosening: float = 1.2


TRUSTED_TIER = "trusted"


def load_policy_config() -> PolicyConfig:
    """The policy section of config/rules.yaml, as currently loaded."""
    section = get_rules_config().policy
    return PolicyConfig(
        version=section.version,
        allow_below=section.allow_below,
        step_up_below=section.step_up_below,
        review_below=section.review_below,
        high_value_amount=section.high_value_amount,
        high_value_tightening=section.high_value_tightening,
        trusted_tier_loosening=section.trusted_tier_loosening,
    )


def _multiplier(ctx: PolicyContext, cfg: PolicyConfig) -> float:
    multiplier = 1.0
    if ctx.amount >= cfg.high_value_amount:
        multiplier *= cfg.high_value_tightening
    if ctx.customer_tier == TRUSTED_TIER:
        multiplier *= cfg.trusted_tier_loosening
    return multiplier


def decide(score: float, ctx: PolicyContext, cfg: PolicyConfig) -> Decision:
    """Maps a calibrated score plus business context to an action.
    Contains NO risk computation - it only reads the score."""
    multiplier = _multiplier(ctx, cfg)
    if score < cfg.allow_below * multiplier:
        return Decision.ALLOW
    if score < cfg.step_up_below * multiplier:
        return Decision.STEP_UP
    if score < cfg.review_below * multiplier:
        return Decision.REVIEW
    return Decision.REJECT
