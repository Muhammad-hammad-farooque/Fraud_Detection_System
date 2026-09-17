"""
Unit tests for scripts/monitor.py — reporting on confirmed outcomes (T-07).
"""
import inspect
from types import SimpleNamespace

import pytest

from scripts import monitor
from scripts.monitor import (
    MIN_LABELLED_FOR_METRICS,
    compute_business_metrics,
    compute_metrics,
    false_positive_ratio,
)


def _txn(txn_id, amount, decision="ALLOW", predicted_fraud=False):
    return SimpleNamespace(
        id=txn_id, amount=amount, decision=decision,
        predicted_fraud=predicted_fraud, risk_level="LOW", risk_score=0.1,
    )


TRANSACTIONS = [
    _txn(1, 100.0, "ALLOW"),
    _txn(2, 200.0, "ALLOW"),
    _txn(3, 300.0, "REVIEW"),
    _txn(4, 400.0, "REJECT", predicted_fraud=True),
]


class TestBusinessMetrics:
    def test_approval_rate_counts_allowed_transactions(self):
        metrics = compute_business_metrics(TRANSACTIONS, truth={})
        assert metrics["approval_rate"] == pytest.approx(0.5)

    def test_approval_rate_needs_no_labels(self):
        """Executives watch this one daily; it must not wait for confirmations."""
        assert compute_business_metrics(TRANSACTIONS, truth={})["approval_rate"] > 0

    def test_fraud_bps_is_share_of_labelled_value(self):
        # Labelled: txn 1 (100, legit) and txn 4 (400, fraud) -> 400/500 = 8000 bps
        metrics = compute_business_metrics(TRANSACTIONS, truth={1: False, 4: True})
        assert metrics["fraud_bps"] == pytest.approx(8000.0)

    def test_fraud_bps_is_none_without_labels(self):
        assert compute_business_metrics(TRANSACTIONS, truth={})["fraud_bps"] is None

    def test_coverage_reports_the_labelled_share(self):
        metrics = compute_business_metrics(TRANSACTIONS, truth={1: False, 4: True})
        assert metrics["labelled_count"] == 2
        assert metrics["total_count"] == 4
        assert metrics["coverage"] == pytest.approx(0.5)

    def test_empty_window_does_not_divide_by_zero(self):
        metrics = compute_business_metrics([], truth={})
        assert metrics["approval_rate"] == 0.0
        assert metrics["coverage"] == 0.0
        assert metrics["fraud_bps"] is None


class TestFalsePositiveRatio:
    def test_ratio_is_legit_blocked_per_fraud_caught(self):
        assert false_positive_ratio(tp=2, fp=16) == "8.0:1"

    def test_zero_false_positives(self):
        assert false_positive_ratio(tp=3, fp=0) == "0.0:1"

    def test_no_fraud_caught_is_not_a_division(self):
        assert "n/a" in false_positive_ratio(tp=0, fp=5)


class TestSuppression:
    def test_threshold_is_meaningful(self):
        assert MIN_LABELLED_FOR_METRICS >= 20

    def test_small_samples_are_suppressed_with_a_warning(self):
        source = inspect.getsource(monitor.run)
        assert "MIN_LABELLED_FOR_METRICS" in source
        assert "suppressed" in source

    def test_claim_status_influences_no_metric(self):
        """A3: claims are rejected for staleness and serial claiming, not fraud."""
        source = inspect.getsource(monitor)
        assert "APPROVED" not in source
        assert "claim.status" not in source
        assert "models.Claim" not in source


class TestDetectionMetrics:
    def test_confusion_matrix_maths_is_unchanged(self):
        m = compute_metrics(tp=8, fp=2, tn=80, fn=2)
        assert m["precision"] == pytest.approx(0.8)
        assert m["recall"] == pytest.approx(0.8)
        assert m["fpr"] == pytest.approx(2 / 82)

    def test_empty_counts_are_zero_not_errors(self):
        m = compute_metrics(tp=0, fp=0, tn=0, fn=0)
        assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
