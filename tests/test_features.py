"""
Unit tests for app/features.py — the single feature computation path (T-01).
"""
import inspect
import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from app import features
from app.features import (
    FEATURE_ORDER,
    FeatureVector,
    UserAggregates,
    aggregates_from_history,
    compute_features,
)
from app.fraud_detection import build_feature_vector
from scripts.retrain import build_features

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _aggs(**overrides) -> UserAggregates:
    base = dict(
        txn_count=4,
        avg_amount=100.0,
        known_locations=frozenset({"Lahore", "Karachi"}),
        count_last_2m=0,
        is_cold_start=False,
    )
    base.update(overrides)
    return UserAggregates(**base)


def _txn(amount, location, created_at):
    return SimpleNamespace(amount=amount, location=location, created_at=created_at)


# ── compute_features ─────────────────────────────────────────────────────────

class TestComputeFeatures:
    def test_cold_start_does_not_penalise_location(self):
        cold = aggregates_from_history([], NOW)
        assert cold.is_cold_start is True
        fv = compute_features(250.0, "Anywhere", cold, device_user_count=0)
        assert fv.is_new_location == 0
        assert fv.amount_deviation == 1.0

    def test_zero_average_amount_gives_neutral_deviation(self):
        fv = compute_features(500.0, "Lahore", _aggs(avg_amount=0.0), device_user_count=0)
        assert fv.amount_deviation == 1.0

    def test_amount_deviation_is_ratio_to_average(self):
        fv = compute_features(350.0, "Lahore", _aggs(avg_amount=100.0), device_user_count=0)
        assert fv.amount_deviation == pytest.approx(3.5)

    def test_unknown_location_is_new(self):
        fv = compute_features(100.0, "Quetta", _aggs(), device_user_count=0)
        assert fv.is_new_location == 1

    def test_known_location_is_not_new(self):
        fv = compute_features(100.0, "Karachi", _aggs(), device_user_count=0)
        assert fv.is_new_location == 0

    @pytest.mark.parametrize("count, flagged", [(0, 0), (2, 0), (3, 1), (4, 1)])
    def test_device_flag_threshold(self, count, flagged):
        fv = compute_features(100.0, "Lahore", _aggs(), device_user_count=count)
        assert fv.is_flagged_device == flagged

    def test_velocity_copied_from_aggregates(self):
        fv = compute_features(100.0, "Lahore", _aggs(count_last_2m=5), device_user_count=0)
        assert fv.velocity_2m == 5

    def test_deterministic(self):
        args = (1234.5, "Quetta", _aggs(count_last_2m=2), 3)
        assert compute_features(*args) == compute_features(*args)

    def test_takes_no_db_or_clock(self):
        params = set(inspect.signature(compute_features).parameters)
        assert params == {"amount", "location", "aggregates", "device_user_count"}
        with patch.object(features, "datetime", side_effect=AssertionError("clock used")):
            compute_features(100.0, "Lahore", _aggs(), 1)

    def test_to_frame_column_order(self):
        fv = compute_features(100.0, "Lahore", _aggs(), 0)
        frame = fv.to_frame()
        assert list(frame.columns) == FEATURE_ORDER
        assert frame.shape == (1, len(FEATURE_ORDER))

    def test_feature_order_matches_vector_fields(self):
        assert list(FeatureVector.__dataclass_fields__) == FEATURE_ORDER


# ── aggregates_from_history ──────────────────────────────────────────────────

class TestAggregatesFromHistory:
    def test_basic_aggregates(self):
        history = [
            _txn(100.0, "Lahore", NOW - timedelta(days=1)),
            _txn(300.0, "Karachi", NOW - timedelta(hours=1)),
        ]
        aggs = aggregates_from_history(history, NOW)
        assert aggs.txn_count == 2
        assert aggs.avg_amount == pytest.approx(200.0)
        assert aggs.known_locations == frozenset({"Lahore", "Karachi"})
        assert aggs.is_cold_start is False

    def test_velocity_window_boundary_is_inclusive(self):
        history = [
            _txn(1.0, "A", NOW - timedelta(seconds=120)),   # exactly on the edge: counted
            _txn(1.0, "A", NOW - timedelta(seconds=121)),   # just outside: not counted
            _txn(1.0, "A", NOW - timedelta(seconds=5)),
        ]
        assert aggregates_from_history(history, NOW).count_last_2m == 2

    def test_naive_timestamps_treated_as_utc(self):
        naive_recent = (NOW - timedelta(seconds=30)).replace(tzinfo=None)
        aggs = aggregates_from_history([_txn(1.0, "A", naive_recent)], NOW)
        assert aggs.count_last_2m == 1


# ── serving vs training parity ───────────────────────────────────────────────

class TestServingTrainingParity:
    """The live scoring path and the retraining path must yield identical vectors."""

    @pytest.mark.parametrize("seed", range(5))
    def test_identical_vectors_for_identical_input(self, seed):
        rng = random.Random(seed)
        t = NOW
        rows = []
        for i in range(60):
            t += timedelta(seconds=rng.randint(1, 200))
            rows.append({
                "id": i,
                "user_id": rng.randint(1, 3),
                "amount": rng.choice([0.0, 50.0, 120.0, 999.0, 6000.0]),
                "location": rng.choice(["Lahore", "Karachi", "Quetta"]),
                "device_id": rng.choice(["d1", "d2", "d3"]),
                "created_at": t,
                "fraud": 0,
            })
        df = pd.DataFrame(rows)
        device_user_counts = df.groupby("device_id")["user_id"].nunique().to_dict()

        trained = build_features(df, device_user_counts)

        for i, row in enumerate(rows):
            history = [
                _txn(r["amount"], r["location"], r["created_at"])
                for r in rows[:i] if r["user_id"] == row["user_id"]
            ]
            with patch(
                "app.fraud_detection.count_device_users",
                return_value=device_user_counts[row["device_id"]],
            ):
                served = build_feature_vector(
                    db=None,
                    transaction=SimpleNamespace(**row),
                    user_transactions=history,
                    now=row["created_at"],
                )
            train_row = trained.iloc[i]
            for name in FEATURE_ORDER:
                assert getattr(served, name) == pytest.approx(train_row[name]), (seed, i, name)
