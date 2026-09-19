"""Decision audit trail: record every decision, and replay it from the record.

A regulator or an investigator must be able to take one stored row and get the
same score and the same action back, whatever has happened to the rules, the
thresholds or the model since. record_decision captures that; replay proves it.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from . import models, scoring
from .features import FeatureVector
from .policy import Decision, PolicyConfig, PolicyContext, decide
from .scoring import RuleHit, ScoreBreakdown


def record_decision(
    db: Session,
    transaction: models.Transaction,
    breakdown: ScoreBreakdown,
    decision: Decision,
    policy_cfg: PolicyConfig,
    policy_ctx: PolicyContext,
    now: datetime | None = None,
) -> models.DecisionAudit:
    """Add the audit row for a decision to the caller's database transaction.

    Never commits: the row must land atomically with the transaction it
    describes, or not at all.
    """
    if breakdown.feature_vector is None:
        raise ValueError("A decision cannot be audited without its feature vector")
    if breakdown.scoring_params is None:
        raise ValueError("A decision cannot be audited without the scoring parameters it used")

    audit = models.DecisionAudit(
        transaction_id=transaction.id,
        feature_vector=asdict(breakdown.feature_vector),
        rule_hits=[{"rule_id": hit.rule_id, "weight": hit.weight} for hit in breakdown.rule_hits],
        rule_score=breakdown.rule_score,
        model_prob=breakdown.model_probability,
        final_score=breakdown.final_score,
        decision=decision.value,
        model_version=breakdown.model_version,
        policy_version=policy_cfg.version,
        scoring_params=dict(breakdown.scoring_params),
        policy_config=asdict(policy_cfg),
        policy_context=asdict(policy_ctx),
        created_at=now or datetime.now(timezone.utc),
    )
    db.add(audit)
    return audit


@dataclass(frozen=True)
class ReplayResult:
    final_score: float
    decision: Decision
    matches: bool                   # score and decision both reproduce the stored ones
    rules_still_agree: bool         # today's rules fire the same way on the stored input


def replay(audit: models.DecisionAudit) -> ReplayResult:
    """Recompute a decision from its audit row alone.

    Uses the parameters stored on the row, not the ones in force today, and the
    same arithmetic in the same order as app/scoring.py, so a faithful record
    reproduces its score bit-for-bit.
    """
    params = audit.scoring_params
    hits = [RuleHit(h["rule_id"], h["weight"]) for h in audit.rule_hits]

    rule_score = sum(hit.weight for hit in hits) / params["total_rule_weight"]
    final_score = params["w_rules"] * rule_score + params["w_model"] * audit.model_prob

    replayed_decision = decide(
        final_score,
        PolicyContext(**audit.policy_context),
        PolicyConfig(**audit.policy_config),
    )

    fv = FeatureVector(**audit.feature_vector)
    current_hits = [hit.rule_id for hit in scoring.evaluate_rules(fv)]

    return ReplayResult(
        final_score=final_score,
        decision=replayed_decision,
        matches=final_score == audit.final_score and replayed_decision.value == audit.decision,
        rules_still_agree=current_hits == [hit.rule_id for hit in hits],
    )
