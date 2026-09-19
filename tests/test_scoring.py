"""
Unit tests for app/scoring.py — rule evaluation and score aggregation (T-03).
"""
import random

import pytest

from app.features import FeatureVector
from app.config import get_rules_config
from app.scoring import (
    RuleHit,
    active_rules,
    combine,
    evaluate_rules,
    normalised_rule_score,
    score,
    total_rule_weight,
)


def _weights():
    scoring = get_rules_config().scoring
    return scoring.w_rules, scoring.w_model

QUIET = FeatureVector(
    amount=100.0,
    amount_deviation=1.0,
    is_new_location=0,
    is_flagged_device=0,
    velocity_2m=0,
)

LOUD = FeatureVector(
    amount=9000.0,
    amount_deviation=90.0,
    is_new_location=1,
    is_flagged_device=1,
    velocity_2m=12,
)

# One feature vector per rule, firing that rule and no other.
SINGLE_RULE_VECTORS = {
    "R1_HIGH_AMOUNT": FeatureVector(5000.01, 1.0, 0, 0, 0),
    "R2_AMOUNT_DEVIATION": FeatureVector(100.0, 3.01, 0, 0, 0),
    "R3_NEW_LOCATION": FeatureVector(100.0, 1.0, 1, 0, 0),
    "R4_FLAGGED_DEVICE": FeatureVector(100.0, 1.0, 0, 1, 0),
    "R5_VELOCITY": FeatureVector(100.0, 1.0, 0, 0, 5),
}


def _ids(hits) -> list[str]:
    return [hit.rule_id for hit in hits]


# ── evaluate_rules ───────────────────────────────────────────────────────────

class TestEvaluateRules:
    def test_no_rules_fire_on_a_quiet_transaction(self):
        assert evaluate_rules(QUIET) == []

    def test_every_rule_fires_on_a_loud_transaction(self):
        assert _ids(evaluate_rules(LOUD)) == [rule.id for rule in active_rules()]

    @pytest.mark.parametrize("rule_id", list(SINGLE_RULE_VECTORS))
    def test_each_rule_fires_alone(self, rule_id):
        assert _ids(evaluate_rules(SINGLE_RULE_VECTORS[rule_id])) == [rule_id]

    def test_every_rule_has_a_single_rule_vector(self):
        """Guards against a rule being added without a test."""
        assert set(SINGLE_RULE_VECTORS) == {rule.id for rule in active_rules()}

    @pytest.mark.parametrize("amount, fires", [(4999.99, False), (5000.0, False), (5000.01, True)])
    def test_high_amount_boundary(self, amount, fires):
        hits = _ids(evaluate_rules(FeatureVector(amount, 1.0, 0, 0, 0)))
        assert ("R1_HIGH_AMOUNT" in hits) is fires

    @pytest.mark.parametrize("deviation, fires", [(2.99, False), (3.0, False), (3.01, True)])
    def test_deviation_boundary(self, deviation, fires):
        hits = _ids(evaluate_rules(FeatureVector(100.0, deviation, 0, 0, 0)))
        assert ("R2_AMOUNT_DEVIATION" in hits) is fires

    @pytest.mark.parametrize("velocity, fires", [(3, False), (4, False), (5, True), (6, True)])
    def test_velocity_boundary(self, velocity, fires):
        hits = _ids(evaluate_rules(FeatureVector(100.0, 1.0, 0, 0, velocity)))
        assert ("R5_VELOCITY" in hits) is fires


# ── combine ──────────────────────────────────────────────────────────────────

class TestCombine:
    def test_weights_split_the_whole_score(self):
        w_rules, w_model = _weights()
        assert w_rules + w_model == pytest.approx(1.0)

    def test_nothing_fires_and_model_is_certain_it_is_clean(self):
        assert combine([], 0.0) == pytest.approx(0.0)

    def test_everything_fires_and_model_is_certain_it_is_fraud(self):
        assert combine(evaluate_rules(LOUD), 1.0) == pytest.approx(1.0)

    def test_rule_score_is_normalised_by_total_weight(self):
        hits = evaluate_rules(SINGLE_RULE_VECTORS["R5_VELOCITY"])
        assert normalised_rule_score(hits) == pytest.approx(0.5 / total_rule_weight())

    def test_model_contribution_survives_every_rule_firing(self):
        """A5: with all rules fired, the model must still move the score."""
        all_hits = evaluate_rules(LOUD)
        _, w_model = _weights()
        assert combine(all_hits, 1.0) - combine(all_hits, 0.0) == pytest.approx(w_model)

    @pytest.mark.parametrize("rule_id", list(SINGLE_RULE_VECTORS))
    def test_model_contribution_survives_at_every_hit_count(self, rule_id):
        hits = evaluate_rules(SINGLE_RULE_VECTORS[rule_id])
        assert combine(hits, 1.0) > combine(hits, 0.0)

    def test_correlated_amount_rules_do_not_force_a_reject(self):
        """A6: R1 and R2 fire together on any large amount and used to total 0.8."""
        both = FeatureVector(9000.0, 90.0, 0, 0, 0)
        assert _ids(evaluate_rules(both)) == ["R1_HIGH_AMOUNT", "R2_AMOUNT_DEVIATION"]
        assert combine(evaluate_rules(both), 0.0) < 0.7

    def test_stays_in_range_for_random_vectors(self):
        rng = random.Random(0)
        for _ in range(10_000):
            fv = FeatureVector(
                amount=rng.uniform(0, 1_000_000),
                amount_deviation=rng.uniform(0, 5_000),
                is_new_location=rng.randint(0, 1),
                is_flagged_device=rng.randint(0, 1),
                velocity_2m=rng.randint(0, 500),
            )
            final = combine(evaluate_rules(fv), rng.random())
            assert 0.0 <= final <= 1.0

    def test_maximum_is_reachable_without_clamping(self):
        """The cap is a property of the arithmetic, not of a min() call."""
        assert normalised_rule_score(evaluate_rules(LOUD)) == pytest.approx(1.0)


# ── score ────────────────────────────────────────────────────────────────────

class TestScoreBreakdown:
    def test_breakdown_carries_the_full_trace(self):
        breakdown = score(LOUD, 0.9, "test-model-1")
        assert _ids(breakdown.rule_hits) == [rule.id for rule in active_rules()]
        assert breakdown.rule_score == pytest.approx(1.0)
        assert breakdown.model_probability == 0.9
        w_rules, w_model = _weights()
        assert breakdown.final_score == pytest.approx(w_rules + w_model * 0.9)
        assert breakdown.model_version == "test-model-1"
        assert breakdown.feature_vector == LOUD

    def test_breakdown_is_replayable(self):
        """T-11 depends on a stored breakdown reproducing its own score."""
        breakdown = score(LOUD, 0.42, "test-model-1")
        replayed = combine(
            [RuleHit(hit.rule_id, hit.weight) for hit in breakdown.rule_hits],
            breakdown.model_probability,
        )
        assert replayed == breakdown.final_score


class TestRulesFromConfig:
    """T-13: weights and thresholds come from config/rules.yaml, not from code."""

    def test_threshold_change_needs_no_code_change(self, rules_config):
        just_above = FeatureVector(4001.0, 1.0, 0, 0, 0)
        assert evaluate_rules(just_above) == []
        rules_config(scoring={"rules": {"R1_HIGH_AMOUNT": {"weight": 0.25, "threshold": 4000}}})
        assert [h.rule_id for h in evaluate_rules(just_above)] == ["R1_HIGH_AMOUNT"]

    def test_weight_change_rescales_the_rule_score(self, rules_config):
        hits_before = evaluate_rules(SINGLE_RULE_VECTORS["R5_VELOCITY"])
        before = normalised_rule_score(hits_before)
        rules_config(scoring={"rules": {"R5_VELOCITY": {"weight": 1.0, "threshold": 5}}})
        after = normalised_rule_score(evaluate_rules(SINGLE_RULE_VECTORS["R5_VELOCITY"]))
        assert after > before

    def test_blend_weights_come_from_config(self, rules_config):
        rules_config(scoring={"w_rules": 0.4, "w_model": 0.6})
        assert combine([], 1.0) == pytest.approx(0.6)

    def test_breakdown_records_the_parameters_it_used(self, rules_config):
        rules_config(version="rules-test", scoring={"w_rules": 0.4, "w_model": 0.6})
        breakdown = score(LOUD, 0.5, "m")
        assert breakdown.scoring_params == {
            "rules_version": "rules-test",
            "total_rule_weight": pytest.approx(1.5),
            "w_rules": 0.4,
            "w_model": 0.6,
        }
