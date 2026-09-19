"""
Unit tests for app/policy.py — score plus business context to an action (T-04).
"""
import inspect

import pytest

from app import policy
from app.policy import (
    TRUSTED_TIER,
    Decision,
    PolicyConfig,
    PolicyContext,
    decide,
    load_policy_config,
)

CFG = PolicyConfig(
    version="test-policy",
    allow_below=0.3,
    step_up_below=0.5,
    review_below=0.7,
    high_value_amount=5000.0,
)

SMALL = PolicyContext(amount=100.0)
LARGE = PolicyContext(amount=5000.0)
TRUSTED = PolicyContext(amount=100.0, customer_tier=TRUSTED_TIER)


class TestThresholds:
    @pytest.mark.parametrize("score, expected", [
        (0.0, Decision.ALLOW),
        (0.29, Decision.ALLOW),
        (0.3, Decision.STEP_UP),
        (0.49, Decision.STEP_UP),
        (0.5, Decision.REVIEW),
        (0.69, Decision.REVIEW),
        (0.7, Decision.REJECT),
        (1.0, Decision.REJECT),
    ])
    def test_every_boundary(self, score, expected):
        assert decide(score, SMALL, CFG) == expected

    def test_thresholds_come_from_config_not_code(self):
        loose = PolicyConfig("loose", allow_below=0.9, step_up_below=0.95,
                             review_below=0.99, high_value_amount=1e9)
        assert decide(0.85, SMALL, loose) == Decision.ALLOW
        assert decide(0.85, SMALL, CFG) == Decision.REJECT

    def test_defaults_preserve_the_original_bands(self):
        cfg = load_policy_config()
        assert cfg.allow_below == 0.3
        assert cfg.review_below == 0.7
        assert cfg.version

    def test_bands_come_from_the_rules_file(self, rules_config):
        rules_config(policy={"version": "policy-from-file", "allow_below": 0.42})
        cfg = load_policy_config()
        assert cfg.allow_below == 0.42
        assert cfg.version == "policy-from-file"

    def test_multipliers_come_from_the_rules_file(self, rules_config):
        rules_config(policy={"high_value_tightening": 0.5, "trusted_tier_loosening": 1.5})
        cfg = load_policy_config()
        assert (cfg.high_value_tightening, cfg.trusted_tier_loosening) == (0.5, 1.5)

    def test_rows_written_before_the_multipliers_were_configurable_still_load(self):
        """Pre-T-13 audit rows lack the multiplier fields; the defaults are the old constants."""
        cfg = PolicyConfig("old", 0.3, 0.5, 0.7, 5000.0)
        assert (cfg.high_value_tightening, cfg.trusted_tier_loosening) == (0.8, 1.2)


class TestBusinessContext:
    def test_high_value_tightens_thresholds(self):
        """The same score is treated more strictly on a large amount."""
        score = 0.29
        assert decide(score, SMALL, CFG) == Decision.ALLOW
        assert decide(score, LARGE, CFG) == Decision.STEP_UP

    def test_high_value_boundary_scales_by_the_documented_factor(self):
        edge = CFG.allow_below * CFG.high_value_tightening
        assert decide(edge - 0.001, LARGE, CFG) == Decision.ALLOW
        assert decide(edge, LARGE, CFG) == Decision.STEP_UP

    def test_trusted_tier_loosens_thresholds(self):
        score = 0.31
        assert decide(score, SMALL, CFG) == Decision.STEP_UP
        assert decide(score, TRUSTED, CFG) == Decision.ALLOW

    def test_trusted_boundary_scales_by_the_documented_factor(self):
        edge = CFG.allow_below * CFG.trusted_tier_loosening
        assert decide(edge - 0.001, TRUSTED, CFG) == Decision.ALLOW
        assert decide(edge, TRUSTED, CFG) == Decision.STEP_UP

    def test_standard_tier_is_unscaled(self):
        for score in (0.1, 0.35, 0.6, 0.9):
            assert decide(score, SMALL, CFG) == decide(score, PolicyContext(amount=100.0), CFG)

    def test_high_value_trusted_customer_combines_both(self):
        ctx = PolicyContext(amount=9000.0, customer_tier=TRUSTED_TIER)
        expected = CFG.allow_below * CFG.high_value_tightening * CFG.trusted_tier_loosening
        assert decide(expected - 0.001, ctx, CFG) == Decision.ALLOW
        assert decide(expected, ctx, CFG) == Decision.STEP_UP


class TestSeparationOfConcerns:
    def test_step_up_is_a_first_class_action(self):
        assert Decision.STEP_UP in set(Decision)
        assert decide(0.4, SMALL, CFG) == Decision.STEP_UP

    def test_policy_does_no_risk_computation(self):
        """P1: decide() reads a score; it must not reach for features or a model."""
        source = inspect.getsource(policy)
        for forbidden in ("compute_features", "predict_fraud", "evaluate_rules", "FeatureVector"):
            assert forbidden not in source

    def test_scoring_carries_no_decision_vocabulary(self):
        """The inverse: app/scoring.py must not name actions or thresholds bands."""
        from app import scoring

        source = inspect.getsource(scoring)
        for forbidden in ("ALLOW", "REJECT", "REVIEW", "STEP_UP", "LOW", "MEDIUM", "HIGH_RISK"):
            assert forbidden not in source
