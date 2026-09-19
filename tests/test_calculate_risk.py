"""
Direct unit tests for the scoring path — calculate_risk end to end (T-09).

test_scoring.py covers the rule arithmetic in isolation; these drive the whole
serving path, database reads included, the way the router does.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app import models
from app.fraud_detection import calculate_risk, score_transaction
from app.config import get_rules_config
from app.scoring import active_rules

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _candidate(amount=100.0, location="Lahore", device_id="d1"):
    return SimpleNamespace(amount=amount, location=location, device_id=device_id)


def _seed_user(db, user_id=1):
    db.add(models.User(id=user_id, name=f"U{user_id}", email=f"u{user_id}@t.com", hashed_password="x"))
    db.commit()


def _seed_history(db, rows, user_id=1):
    """rows: (amount, location, device_id, seconds_ago)."""
    for amount, location, device_id, ago in rows:
        db.add(models.Transaction(
            user_id=user_id, amount=amount, location=location, device_id=device_id,
            created_at=(NOW - timedelta(seconds=ago)).replace(tzinfo=None),
        ))
    db.commit()


def _hits(db, candidate, user_id=1):
    breakdown = score_transaction(db, candidate, user_id=user_id, now=NOW)
    return {hit.rule_id for hit in breakdown.rule_hits}


BASELINE = [(100.0, "Lahore", "d1", 3600)] * 3


class TestEachRuleThroughTheFullPath:
    def test_quiet_transaction_fires_nothing(self, db_session):
        _seed_user(db_session)
        _seed_history(db_session, BASELINE)
        assert _hits(db_session, _candidate(amount=110.0)) == set()

    def test_high_amount_alone(self, db_session):
        _seed_user(db_session)
        _seed_history(db_session, [(4000.0, "Lahore", "d1", 3600)] * 3)
        assert _hits(db_session, _candidate(amount=6000.0)) == {"R1_HIGH_AMOUNT"}

    def test_deviation_alone(self, db_session):
        _seed_user(db_session)
        _seed_history(db_session, BASELINE)
        assert _hits(db_session, _candidate(amount=400.0)) == {"R2_AMOUNT_DEVIATION"}

    def test_new_location_alone(self, db_session):
        _seed_user(db_session)
        _seed_history(db_session, BASELINE)
        assert _hits(db_session, _candidate(location="Quetta")) == {"R3_NEW_LOCATION"}

    def test_flagged_device_alone(self, db_session):
        _seed_user(db_session)
        for user_id in (2, 3):
            _seed_user(db_session, user_id)
            _seed_history(db_session, [(100.0, "Lahore", "shared", 3600)], user_id=user_id)
        _seed_history(db_session, [(100.0, "Lahore", "shared", 3600)])
        assert _hits(db_session, _candidate(device_id="shared")) == {"R4_FLAGGED_DEVICE"}

    def test_velocity_alone(self, db_session):
        _seed_user(db_session)
        _seed_history(db_session, [(100.0, "Lahore", "d1", 10)] * 5)
        assert _hits(db_session, _candidate()) == {"R5_VELOCITY"}

    def test_all_rules_together(self, db_session):
        _seed_user(db_session)
        for user_id in (2, 3):
            _seed_user(db_session, user_id)
            _seed_history(db_session, [(100.0, "Lahore", "shared", 3600)], user_id=user_id)
        _seed_history(db_session, [(100.0, "Lahore", "shared", 10)] * 5)
        hits = _hits(db_session, _candidate(amount=9000.0, location="Quetta", device_id="shared"))
        assert hits == {rule.id for rule in active_rules()}


class TestBoundaries:
    @pytest.mark.parametrize("amount, fires", [(5000.0, False), (5000.01, True)])
    def test_high_amount_threshold(self, db_session, amount, fires):
        _seed_user(db_session)
        _seed_history(db_session, [(5000.0, "Lahore", "d1", 3600)] * 3)
        assert ("R1_HIGH_AMOUNT" in _hits(db_session, _candidate(amount=amount))) is fires

    @pytest.mark.parametrize("count, fires", [(4, False), (5, True)])
    def test_velocity_threshold(self, db_session, count, fires):
        _seed_user(db_session)
        _seed_history(db_session, [(100.0, "Lahore", "d1", 10)] * count)
        assert ("R5_VELOCITY" in _hits(db_session, _candidate())) is fires

    @pytest.mark.parametrize("ago, counts", [(119, True), (121, False)])
    def test_velocity_window_edge(self, db_session, ago, counts):
        _seed_user(db_session)
        _seed_history(db_session, [(100.0, "Lahore", "d1", ago)] * 5)
        assert ("R5_VELOCITY" in _hits(db_session, _candidate())) is counts

    @pytest.mark.parametrize("users, fires", [(2, False), (3, True)])
    def test_device_sharing_threshold(self, db_session, users, fires):
        _seed_user(db_session)
        for user_id in range(2, 2 + users):
            _seed_user(db_session, user_id)
            _seed_history(db_session, [(100.0, "Lahore", "shared", 3600)], user_id=user_id)
        _seed_history(db_session, BASELINE)
        assert ("R4_FLAGGED_DEVICE" in _hits(db_session, _candidate(device_id="shared"))) is fires


class TestScoreProperties:
    def test_score_is_in_range(self, db_session):
        _seed_user(db_session)
        _seed_history(db_session, BASELINE)
        for amount in (0.01, 100.0, 5000.0, 1_000_000.0):
            score = calculate_risk(db_session, _candidate(amount=amount), user_id=1)
            assert 0.0 <= score <= 1.0

    def test_raising_the_amount_never_lowers_the_score(self, db_session):
        """Monotonicity: with everything else fixed, more money is never less risky."""
        _seed_user(db_session)
        _seed_history(db_session, BASELINE)
        scores = [
            calculate_risk(db_session, _candidate(amount=amount), user_id=1)
            for amount in (10.0, 100.0, 301.0, 1000.0, 5001.0, 20_000.0, 100_000.0)
        ]
        assert scores == sorted(scores), scores

    def test_model_moves_the_score_on_a_fully_fired_transaction(self, db_session):
        """A5 through the real path: the model is not squeezed out at the top."""
        _seed_user(db_session)
        for user_id in (2, 3):
            _seed_user(db_session, user_id)
            _seed_history(db_session, [(100.0, "Lahore", "shared", 3600)], user_id=user_id)
        _seed_history(db_session, [(100.0, "Lahore", "shared", 10)] * 5)
        breakdown = score_transaction(
            db_session, _candidate(amount=9000.0, location="Quetta", device_id="shared"),
            user_id=1, now=NOW,
        )
        assert breakdown.rule_score == pytest.approx(1.0)
        w_model = get_rules_config().scoring.w_model
        assert breakdown.final_score == pytest.approx(1.0 - w_model + w_model * breakdown.model_probability)
        assert breakdown.final_score <= 1.0

    def test_cold_start_user_is_not_penalised(self, db_session):
        _seed_user(db_session)
        assert _hits(db_session, _candidate(location="Somewhere New")) == set()
