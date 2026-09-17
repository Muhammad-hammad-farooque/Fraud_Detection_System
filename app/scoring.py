"""Rule evaluation and score aggregation.

Contains no thresholds and no decision vocabulary - turning a score into an
action is the policy layer's job (T-04). The output is a ScoreBreakdown, which
carries everything needed to reconstruct a decision for the audit log (T-11).
"""
from collections.abc import Callable
from dataclasses import dataclass, field

from .features import FeatureVector

HIGH_AMOUNT = 5000.0
HIGH_DEVIATION = 3.0
HIGH_VELOCITY = 5

# Weight split between the rules and the model. They sum to 1.0, so the blended
# score is in [0, 1] by construction and the model contribution can never be
# squeezed out by the rules (fixes A5).
W_RULES = 0.7
W_MODEL = 0.3


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    weight: float
    predicate: Callable[[FeatureVector], bool]


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    weight: float


@dataclass(frozen=True)
class ScoreBreakdown:
    """Everything needed to reconstruct the decision. Feeds the audit log in T-11."""
    rule_hits: list[RuleHit]
    rule_score: float          # normalised to [0, 1]
    model_probability: float   # uncalibrated until T-19, [0, 1]
    final_score: float         # [0, 1]
    model_version: str
    feature_vector: FeatureVector | None = field(default=None)


# A6: R1 (absolute amount) and R2 (amount relative to the user's own average)
# fire together on almost every large transaction. They used to carry 0.4 each,
# so the pair alone produced 0.8 and an immediate reject - a false-positive
# generator. They are kept separate, because an absolute threshold and a
# per-customer deviation are genuinely different signals, but each is re-weighted
# to 0.25 so that the correlated pair (0.5) no longer outweighs any single
# independent signal by more than a little. Merging them was rejected: it would
# hide which of the two actually fired from the audit trail.
RULES: list[Rule] = [
    Rule(
        id="R1_HIGH_AMOUNT",
        description=f"Amount above {HIGH_AMOUNT:.0f}",
        weight=0.25,
        predicate=lambda fv: fv.amount > HIGH_AMOUNT,
    ),
    Rule(
        id="R2_AMOUNT_DEVIATION",
        description=f"Amount more than {HIGH_DEVIATION:.0f}x the user's average",
        weight=0.25,
        predicate=lambda fv: fv.amount_deviation > HIGH_DEVIATION,
    ),
    Rule(
        id="R3_NEW_LOCATION",
        description="Location never seen for this user",
        weight=0.2,
        predicate=lambda fv: bool(fv.is_new_location),
    ),
    Rule(
        id="R4_FLAGGED_DEVICE",
        description="Device shared by several users",
        weight=0.3,
        predicate=lambda fv: bool(fv.is_flagged_device),
    ),
    Rule(
        id="R5_VELOCITY",
        description=f"{HIGH_VELOCITY}+ transactions in the last 2 minutes",
        weight=0.5,
        predicate=lambda fv: fv.velocity_2m >= HIGH_VELOCITY,
    ),
]

TOTAL_RULE_WEIGHT = sum(rule.weight for rule in RULES)


def evaluate_rules(fv: FeatureVector) -> list[RuleHit]:
    """Return one RuleHit per rule whose predicate fires, in RULES order."""
    return [RuleHit(rule.id, rule.weight) for rule in RULES if rule.predicate(fv)]


def combine(rule_hits: list[RuleHit], model_probability: float) -> float:
    """Normalise rule_score by the sum of ALL rule weights so it cannot exceed 1.0,
    then blend:  final = W_RULES * rule_score + W_MODEL * model_probability
    with W_RULES + W_MODEL == 1.0. No min() clipping - the arithmetic
    cannot exceed 1.0 by construction. Fixes A5."""
    return W_RULES * normalised_rule_score(rule_hits) + W_MODEL * model_probability


def normalised_rule_score(rule_hits: list[RuleHit]) -> float:
    """Fired weight as a fraction of the total available rule weight."""
    return sum(hit.weight for hit in rule_hits) / TOTAL_RULE_WEIGHT


def score(fv: FeatureVector, model_probability: float, model_version: str) -> ScoreBreakdown:
    """Evaluate every rule and blend the result with the model probability."""
    rule_hits = evaluate_rules(fv)
    return ScoreBreakdown(
        rule_hits=rule_hits,
        rule_score=normalised_rule_score(rule_hits),
        model_probability=model_probability,
        final_score=combine(rule_hits, model_probability),
        model_version=model_version,
        feature_vector=fv,
    )
