"""
Unit tests for app/repositories/transaction_repo.py — bounded SQL reads (T-02).
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event

from app import models
from app.features import aggregates_from_history, compute_features
from app.fraud_detection import score_transaction
from app.repositories.transaction_repo import get_device_user_count, get_user_aggregates

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class QueryCounter:
    """Counts statements executed on a session's connection."""

    def __init__(self, db):
        self.engine = db.get_bind()
        self.count = 0

    def _on_execute(self, *args, **kwargs):
        self.count += 1

    def __enter__(self):
        event.listen(self.engine, "before_cursor_execute", self._on_execute)
        return self

    def __exit__(self, *exc):
        event.remove(self.engine, "before_cursor_execute", self._on_execute)


def _seed(db, rows):
    """rows: (user_id, amount, location, device_id, seconds_before_now)."""
    db.add(models.User(id=1, name="A", email="a@test.com", hashed_password="x"))
    db.add(models.User(id=2, name="B", email="b@test.com", hashed_password="x"))
    db.add(models.User(id=3, name="C", email="c@test.com", hashed_password="x"))
    for user_id, amount, location, device_id, ago in rows:
        db.add(models.Transaction(
            user_id=user_id,
            amount=amount,
            location=location,
            device_id=device_id,
            created_at=(NOW - timedelta(seconds=ago)).replace(tzinfo=None),
        ))
    db.commit()


HISTORY = [
    (1, 100.0, "Lahore", "d1", 3600),
    (1, 300.0, "Karachi", "d1", 90),
    (1, 200.0, "Lahore", "d2", 30),
    (2, 999.0, "Quetta", "d1", 10),
]


class TestGetUserAggregates:
    def test_counts_average_and_velocity(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 1, NOW, "Lahore")
        assert aggs.txn_count == 3
        assert aggs.avg_amount == pytest.approx(200.0)
        assert aggs.count_last_2m == 2        # 90s and 30s ago, not the 1h-old one
        assert aggs.is_cold_start is False

    def test_known_location_resolved_by_exists(self, db_session):
        _seed(db_session, HISTORY)
        assert "Lahore" in get_user_aggregates(db_session, 1, NOW, "Lahore").known_locations
        assert get_user_aggregates(db_session, 1, NOW, "Quetta").known_locations == frozenset()

    def test_other_users_history_is_excluded(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 2, NOW, "Quetta")
        assert aggs.txn_count == 1
        assert aggs.avg_amount == pytest.approx(999.0)

    def test_user_with_no_history_is_cold_start(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 3, NOW, "Lahore")
        assert aggs.is_cold_start is True
        assert aggs.txn_count == 0
        assert aggs.avg_amount == 0.0
        assert aggs.count_last_2m == 0

    def test_issues_at_most_two_queries(self, db_session):
        _seed(db_session, HISTORY)
        with QueryCounter(db_session) as counter:
            get_user_aggregates(db_session, 1, NOW, "Lahore")
        assert counter.count <= 2

    def test_matches_the_training_path(self, db_session):
        """SQL aggregates and the in-Python training aggregates must agree (A8 invariant)."""
        _seed(db_session, HISTORY)
        history = [r for r in HISTORY if r[0] == 1]
        expected = aggregates_from_history(
            [
                type("Row", (), {
                    "amount": amount,
                    "location": location,
                    "created_at": NOW - timedelta(seconds=ago),
                })()
                for _, amount, location, _, ago in history
            ],
            NOW,
        )
        actual = get_user_aggregates(db_session, 1, NOW, "Lahore")
        assert actual.txn_count == expected.txn_count
        assert actual.avg_amount == pytest.approx(expected.avg_amount)
        assert actual.count_last_2m == expected.count_last_2m
        assert actual.is_cold_start == expected.is_cold_start
        for location in {"Lahore", "Karachi", "Quetta"}:
            sql_aggs = get_user_aggregates(db_session, 1, NOW, location)
            assert compute_features(500.0, location, sql_aggs, 0) == compute_features(
                500.0, location, expected, 0
            )


class TestGetDeviceUserCount:
    def test_counts_distinct_users(self, db_session):
        _seed(db_session, HISTORY)
        assert get_device_user_count(db_session, "d1") == 2   # users 1 and 2
        assert get_device_user_count(db_session, "d2") == 1
        assert get_device_user_count(db_session, "unseen") == 0

    def test_single_query(self, db_session):
        _seed(db_session, HISTORY)
        with QueryCounter(db_session) as counter:
            get_device_user_count(db_session, "d1")
        assert counter.count == 1


class TestScoringIsBounded:
    @pytest.mark.parametrize("history_size", [1, 50, 400])
    def test_query_count_is_flat_regardless_of_history(self, db_session, history_size):
        """A7: scoring must not get more expensive as a customer transacts more."""
        _seed(db_session, [(1, 100.0 + i, f"city-{i % 5}", "d1", i + 200) for i in range(history_size)])
        candidate = type("Candidate", (), {
            "amount": 250.0, "location": "city-1", "device_id": "d1",
        })()
        with QueryCounter(db_session) as counter:
            score_transaction(db_session, candidate, user_id=1, now=NOW)
        assert counter.count <= 3

    def test_scoring_does_not_load_transaction_rows(self, db_session):
        """The scoring path must issue no SELECT of whole transaction rows."""
        _seed(db_session, HISTORY)
        statements = []
        engine = db_session.get_bind()

        def record(conn, cursor, statement, *args):
            statements.append(" ".join(statement.split()).lower())

        event.listen(engine, "before_cursor_execute", record)
        try:
            candidate = type("Candidate", (), {
                "amount": 250.0, "location": "Lahore", "device_id": "d1",
            })()
            score_transaction(db_session, candidate, user_id=1, now=NOW)
        finally:
            event.remove(engine, "before_cursor_execute", record)

        assert statements, "expected the scoring path to query the database"
        for statement in statements:
            assert "transactions.amount" not in statement or "avg(" in statement
            assert "select transactions.id, transactions.location" not in statement
