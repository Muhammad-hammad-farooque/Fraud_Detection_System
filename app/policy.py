"""Policy layer: turns a risk score into an action.

Contains no feature or risk computation - it only reads the score that
app/scoring.py produced. Keeping the two apart means thresholds can be retuned
without revalidating or redeploying the model (design principle P1), and it is
what makes STEP_UP possible as a real action rather than a second reject (P4).
"""
import os
from dataclasses import dataclass
from enum import StrEnum

POLICY_VERSION = "policy-1"


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


# Multipliers applied to every threshold. Below 1.0 the bands shift down, so
# the same score lands on a stricter action.
HIGH_VALUE_TIGHTENING = 0.8
TRUSTED_TIER_LOOSENING = 1.2
TRUSTED_TIER = "trusted"


def load_policy_config() -> PolicyConfig:
    """Read thresholds from the environment, defaulting to the original bands.

    0.3 and 0.7 are preserved as the ALLOW and REJECT boundaries; the old
    single MANUAL_CHECK band between them is split into STEP_UP and REVIEW.
    T-13 moves this to config/rules.yaml.
    """
    return PolicyConfig(
        version=os.getenv("POLICY_VERSION", POLICY_VERSION),
        allow_below=float(os.getenv("POLICY_ALLOW_BELOW", "0.3")),
        step_up_below=float(os.getenv("POLICY_STEP_UP_BELOW", "0.5")),
        review_below=float(os.getenv("POLICY_REVIEW_BELOW", "0.7")),
        high_value_amount=float(os.getenv("POLICY_HIGH_VALUE_AMOUNT", "5000")),
    )


def _multiplier(ctx: PolicyContext, cfg: PolicyConfig) -> float:
    multiplier = 1.0
    if ctx.amount >= cfg.high_value_amount:
        multiplier *= HIGH_VALUE_TIGHTENING
    if ctx.customer_tier == TRUSTED_TIER:
        multiplier *= TRUSTED_TIER_LOOSENING
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
