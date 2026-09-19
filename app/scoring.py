"""Rule evaluation and score aggregation.

Contains no decision vocabulary and no policy bands - turning a score into an
action is the policy layer's job (T-04). The output is a ScoreBreakdown, which
carries everything needed to reconstruct a decision for the audit log (T-11).

Rule logic lives here; each rule's weight and threshold come from
config/rules.yaml (T-13), so retuning never needs a code change.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import RulesConfig, ScoringSection, get_rules_config
from .features import FeatureVector


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
    model_probability: float   # [0, 1]; calibrated only if the active model is (see app/ML/models.py)
    final_score: float         # [0, 1]
    model_version: str
    feature_vector: FeatureVector | None = field(default=None)
    # The scoring parameters this score was computed with, so the audit row
    # replays against them rather than against whatever is configured later.
    scoring_params: dict | None = field(default=None)


# A6: R1 (absolute amount) and R2 (amount relative to the user's own average)
# fire together on almost every large transaction. They used to carry 0.4 each,
# so the pair alone produced 0.8 and an immediate reject - a false-positive
# generator. They are kept separate, because an absolute threshold and a
# per-customer deviation are genuinely different signals, but config/rules.yaml
# weights each at 0.25 so the correlated pair no longer outweighs any single
# independent signal by much. Merging them was rejected: it would hide which of
# the two actually fired from the audit trail.
def _build_rules(scoring: ScoringSection) -> list[Rule]:
    spec = scoring.rules
    high_amount = spec["R1_HIGH_AMOUNT"].threshold
    high_deviation = spec["R2_AMOUNT_DEVIATION"].threshold
    high_velocity = spec["R5_VELOCITY"].threshold
    return [
        Rule(
            id="R1_HIGH_AMOUNT",
            description=f"Amount above {high_amount:g}",
            weight=spec["R1_HIGH_AMOUNT"].weight,
            predicate=lambda fv: fv.amount > high_amount,
        ),
        Rule(
            id="R2_AMOUNT_DEVIATION",
            description=f"Amount more than {high_deviation:g}x the user's average",
            weight=spec["R2_AMOUNT_DEVIATION"].weight,
            predicate=lambda fv: fv.amount_deviation > high_deviation,
        ),
        Rule(
            id="R3_NEW_LOCATION",
            description="Location never seen for this user",
            weight=spec["R3_NEW_LOCATION"].weight,
            predicate=lambda fv: bool(fv.is_new_location),
        ),
        Rule(
            id="R4_FLAGGED_DEVICE",
            description="Device shared by several users",
            weight=spec["R4_FLAGGED_DEVICE"].weight,
            predicate=lambda fv: bool(fv.is_flagged_device),
        ),
        Rule(
            id="R5_VELOCITY",
            description=f"{high_velocity:g}+ transactions in the last 2 minutes",
            weight=spec["R5_VELOCITY"].weight,
            predicate=lambda fv: fv.velocity_2m >= high_velocity,
        ),
    ]


def active_rules(cfg: RulesConfig | None = None) -> list[Rule]:
    """The rules as currently configured, in a stable order."""
    return _build_rules((cfg or get_rules_config()).scoring)


def total_rule_weight(cfg: RulesConfig | None = None) -> float:
    return sum(rule.weight for rule in active_rules(cfg))


def evaluate_rules(fv: FeatureVector, cfg: RulesConfig | None = None) -> list[RuleHit]:
    """Return one RuleHit per rule whose predicate fires, in rule order."""
    return [RuleHit(rule.id, rule.weight) for rule in active_rules(cfg) if rule.predicate(fv)]


def normalised_rule_score(rule_hits: list[RuleHit], cfg: RulesConfig | None = None) -> float:
    """Fired weight as a fraction of the total available rule weight."""
    return sum(hit.weight for hit in rule_hits) / total_rule_weight(cfg)


def combine(rule_hits: list[RuleHit], model_probability: float, cfg: RulesConfig | None = None) -> float:
    """Normalise rule_score by the sum of ALL rule weights so it cannot exceed 1.0,
    then blend:  final = w_rules * rule_score + w_model * model_probability
    with w_rules + w_model == 1.0 (enforced when the config loads). No min()
    clipping - the arithmetic cannot exceed 1.0 by construction. Fixes A5."""
    cfg = cfg or get_rules_config()
    scoring = cfg.scoring
    return scoring.w_rules * normalised_rule_score(rule_hits, cfg) + scoring.w_model * model_probability


def score(
    fv: FeatureVector,
    model_probability: float,
    model_version: str,
    cfg: RulesConfig | None = None,
) -> ScoreBreakdown:
    """Evaluate every rule and blend the result with the model probability.

    One config snapshot is used throughout, so a reload mid-request cannot mix
    weights from two versions of the file.
    """
    cfg = cfg or get_rules_config()
    rule_hits = evaluate_rules(fv, cfg)
    return ScoreBreakdown(
        rule_hits=rule_hits,
        rule_score=normalised_rule_score(rule_hits, cfg),
        model_probability=model_probability,
        final_score=combine(rule_hits, model_probability, cfg),
        model_version=model_version,
        feature_vector=fv,
        scoring_params={
            "rules_version": cfg.version,
            "total_rule_weight": total_rule_weight(cfg),
            "w_rules": cfg.scoring.w_rules,
            "w_model": cfg.scoring.w_model,
        },
    )
