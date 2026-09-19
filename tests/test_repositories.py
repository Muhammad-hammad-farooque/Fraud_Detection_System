"""
Unit tests for app/repositories/transaction_repo.py — bounded, point-in-time SQL
aggregates (T-02, T-16).

Every expected value below is worked out by hand from the seeded history, so
these tests check the SQL itself, not one implementation against another.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from app import models
from app.features import TransactionInput
from app.fraud_detection import features_at, score_transaction
from app.repositories.transaction_repo import (
    get_device_aggregates,
    get_device_user_count,
    get_population_aggregates,
    get_user_aggregates,
)

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
SIGNUP = NOW - timedelta(days=100)


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


def _users(db):
    for user_id, name in ((1, "A"), (2, "B"), (3, "C")):
        db.add(models.User(id=user_id, name=name, email=f"{name.lower()}@test.com",
                           hashed_password="x", created_at=SIGNUP.replace(tzinfo=None)))


def _seed(db, rows):
    """rows: (user_id, amount, location, device_id, seconds_before_now, extras dict)."""
    _users(db)
    for user_id, amount, location, device_id, ago, extra in rows:
        db.add(models.Transaction(
            user_id=user_id, amount=amount, location=location, device_id=device_id,
            created_at=(NOW - timedelta(seconds=ago)).replace(tzinfo=None), **extra,
        ))
    db.commit()


# User 1's history, oldest first. UTC hours: NOW is 12:00.
HISTORY = [
    (1, 100.0, "Lahore",  "d1", 40 * 86400, {"merchant_id": "M5411-001", "merchant_category": "5411"}),  # 40 days ago, 12:00
    (1, 150.0, "Lahore",  "d1", 3 * 86400,  {"merchant_id": "M5411-001", "merchant_category": "5411"}),  # 3 days ago, 12:00
    (1, 300.0, "Karachi", "d1", 5 * 3600,   {"merchant_id": "M5812-002", "merchant_category": "5812"}),  # 07:00
    (1, 200.0, "Lahore",  "d2", 90,         {"latitude": 31.52, "longitude": 74.36}),                     # 11:58:30
    (2, 999.0, "Quetta",  "d1", 10,         {"merchant_id": "M5411-001"}),
]


def _candidate(**overrides):
    base = dict(amount=250.0, location="Lahore", at=NOW, device_id="d1",
                merchant_id="M5411-001", merchant_category="5411", channel="web",
                latitude=24.86, longitude=67.00)
    base.update(overrides)
    return TransactionInput(**base)


class TestUserAggregates:
    def test_counts_average_and_spread(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 1, _candidate())
        assert aggs.txn_count == 4
        assert aggs.avg_amount == pytest.approx(187.5)
        # population variance of 100,150,300,200: mean 187.5 -> 5468.75
        assert aggs.amount_std == pytest.approx(5468.75 ** 0.5)
        assert aggs.max_amount == 300.0
        assert aggs.is_cold_start is False

    def test_velocity_windows(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 1, _candidate())
        # windows: 1m, 5m, 1h, 24h, 7d, 30d
        assert aggs.window_counts == (0, 1, 1, 2, 3, 3)
        assert aggs.window_sums == (0.0, 200.0, 200.0, 500.0, 650.0, 650.0)
        assert aggs.count_last_2m == 1

    def test_candidate_relative_counts(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 1, _candidate())
        assert aggs.location_count == 3                  # three Lahore rows
        assert aggs.device_count == 3                    # three on d1
        assert aggs.merchant_count == 2
        assert aggs.category_count == 2
        assert aggs.known_locations == frozenset({"Lahore"})

    def test_unset_candidate_fields_count_as_zero(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 1, _candidate(device_id=None, merchant_id=None,
                                                             merchant_category=None))
        assert (aggs.device_count, aggs.merchant_count, aggs.category_count) == (0, 0, 0)

    def test_distinct_counts(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 1, _candidate())
        assert aggs.distinct_locations_24h == 2          # Karachi and Lahore within 24h
        assert aggs.distinct_devices == 2

    def test_typical_hour_and_amount_band(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 1, _candidate())
        # Candidate at 12:00 UTC; typical hours 10-14. Rows at 12:00, 12:00, 07:00, 11:58.
        assert aggs.typical_hour_count == 3
        # Candidate 250 -> band [100, 1000); all four amounts fall in it.
        assert aggs.amount_band_count == 4

    def test_previous_transaction_and_signup(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 1, _candidate())
        assert aggs.previous_at.replace(tzinfo=timezone.utc) == NOW - timedelta(seconds=90)
        assert (aggs.previous_latitude, aggs.previous_longitude) == (31.52, 74.36)
        assert aggs.account_created_at.replace(tzinfo=timezone.utc) == SIGNUP

    def test_other_customers_are_excluded(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 2, _candidate(location="Quetta"))
        assert aggs.txn_count == 1 and aggs.avg_amount == 999.0

    def test_no_history_is_cold_start(self, db_session):
        _seed(db_session, HISTORY)
        aggs = get_user_aggregates(db_session, 3, _candidate())
        assert aggs.is_cold_start is True
        assert aggs.txn_count == 0 and aggs.avg_amount == 0.0 and aggs.previous_at is None
        assert aggs.window_counts == (0,) * 6

    def test_issues_at_most_two_queries(self, db_session):
        _seed(db_session, HISTORY)
        with QueryCounter(db_session) as counter:
            get_user_aggregates(db_session, 1, _candidate())
        assert counter.count <= 2


class TestPointInTime:
    """Retraining evaluates these at historical timestamps; nothing later may leak in."""

    def test_rows_at_or_after_the_instant_are_invisible(self, db_session):
        _seed(db_session, HISTORY)
        earlier = _candidate(at=NOW - timedelta(days=2))
        aggs = get_user_aggregates(db_session, 1, earlier)
        assert aggs.txn_count == 2                         # only the 40d and 3d rows
        assert aggs.max_amount == 150.0

    def test_the_candidate_itself_is_excluded_at_its_own_timestamp(self, db_session):
        _seed(db_session, HISTORY)
        exactly = _candidate(at=NOW - timedelta(seconds=90))
        assert get_user_aggregates(db_session, 1, exactly).txn_count == 3

    def test_device_history_is_point_in_time(self, db_session):
        _seed(db_session, HISTORY)
        # At NOW user 2's d1 use (10s ago) counts; 60s earlier it had not happened.
        assert get_device_aggregates(db_session, "d1", NOW).user_count == 2
        assert get_device_aggregates(db_session, "d1", NOW - timedelta(seconds=60)).user_count == 1

    def test_labels_confirmed_later_are_invisible(self, db_session):
        _seed(db_session, HISTORY)
        first = db_session.query(models.Transaction).filter_by(amount=100.0).one()
        db_session.add(models.TransactionOutcome(
            transaction_id=first.id, is_fraud_confirmed=True, source="ANALYST",
            confirmed_at=(NOW - timedelta(days=1)).replace(tzinfo=None)))
        db_session.commit()
        before_label = get_population_aggregates(db_session, _candidate(at=NOW - timedelta(days=2)))
        after_label = get_population_aggregates(db_session, _candidate())
        assert (before_label.merchant_labelled, before_label.merchant_fraud) == (0, 0)
        assert (after_label.merchant_labelled, after_label.merchant_fraud) == (1, 1)


class TestDeviceAndPopulation:
    def test_device_aggregates(self, db_session):
        _seed(db_session, HISTORY)
        device = get_device_aggregates(db_session, "d1", NOW)
        assert device.user_count == 2
        assert device.first_seen_at.replace(tzinfo=timezone.utc) == NOW - timedelta(days=40)

    def test_unseen_and_missing_devices(self, db_session):
        _seed(db_session, HISTORY)
        assert get_device_aggregates(db_session, "never", NOW).user_count == 0
        assert get_device_aggregates(db_session, None, NOW).first_seen_at is None

    def test_population_percentile_is_as_of_the_start_of_the_day(self, db_session):
        """NOW is 12:00; only the 40-day and 3-day rows (100, 150) predate today."""
        _seed(db_session, HISTORY)
        assert get_population_aggregates(db_session, _candidate(amount=125.0)).amount_percentile == 0.5
        # 250 is below today's 300, but today's rows are not in the snapshot yet.
        assert get_population_aggregates(db_session, _candidate(amount=250.0)).amount_percentile == 1.0
        assert get_population_aggregates(db_session, _candidate(amount=50.0)).amount_percentile == 0.0

    def test_no_population_yet_is_none(self, db_session):
        _seed(db_session, HISTORY)
        before_everything = _candidate(at=NOW - timedelta(days=60))
        assert get_population_aggregates(db_session, before_everything).amount_percentile is None

    def test_the_day_snapshot_is_computed_once(self, db_session):
        _seed(db_session, HISTORY)
        get_population_aggregates(db_session, _candidate())
        with QueryCounter(db_session) as counter:
            get_population_aggregates(db_session, _candidate(amount=999.0))
        assert counter.count == 1                           # the merchant statement only

    def test_snapshot_percentiles_track_the_true_rank(self, db_session):
        """1,000 distinct amounts: the snapshot must agree with an exact count to 0.1%."""
        _users(db_session)
        for i in range(1000):
            db_session.add(models.Transaction(user_id=1, amount=float(i + 1), location="L", device_id="d",
                                              created_at=(NOW - timedelta(days=2, seconds=i)).replace(tzinfo=None)))
        db_session.commit()
        for amount in (1.0, 250.5, 500.0, 999.0):
            exact = sum(1 for i in range(1000) if i + 1 < amount) / 1000
            got = get_population_aggregates(db_session, _candidate(amount=amount, merchant_id=None)).amount_percentile
            assert got == pytest.approx(exact, abs=0.001)

    def test_population_without_a_merchant(self, db_session):
        _seed(db_session, HISTORY)
        population = get_population_aggregates(db_session, _candidate(merchant_id=None))
        assert (population.merchant_labelled, population.merchant_fraud) == (0, 0)

    def test_device_user_count_is_all_time(self, db_session):
        _seed(db_session, HISTORY)
        assert get_device_user_count(db_session, "d1") == 2


class TestBoundedScoring:
    @pytest.mark.parametrize("history_size", [1, 50, 400])
    def test_query_count_is_flat_regardless_of_history(self, db_session, history_size):
        """A7: scoring must not get more expensive as a customer transacts more."""
        _seed(db_session, [(1, 100.0 + i, f"city-{i % 5}", "d1", i + 200, {}) for i in range(history_size)])
        candidate = SimpleNamespace(amount=250.0, location="city-1", device_id="d1",
                                    merchant_id="M1", merchant_category="5411")
        score_transaction(db_session, candidate, user_id=1, now=NOW)       # builds today's snapshot
        with QueryCounter(db_session) as counter:
            score_transaction(db_session, candidate, user_id=1, now=NOW)
        assert counter.count <= 4

    def test_the_first_decision_of_a_day_pays_one_snapshot_query(self, db_session):
        _seed(db_session, [(1, 100.0 + i, "L", "d1", i + 200, {}) for i in range(50)])
        candidate = SimpleNamespace(amount=250.0, location="L", device_id="d1",
                                    merchant_id="M1", merchant_category="5411")
        with QueryCounter(db_session) as counter:
            score_transaction(db_session, candidate, user_id=1, now=NOW)
        assert counter.count <= 5

    def test_scoring_does_not_load_transaction_rows(self, db_session):
        """Aggregates only: no statement selects whole transaction rows."""
        _seed(db_session, HISTORY)
        statements = []
        engine = db_session.get_bind()

        def record(conn, cursor, statement, *args):
            statements.append(" ".join(statement.split()).lower())

        event.listen(engine, "before_cursor_execute", record)
        try:
            features_at(db_session, _candidate(), user_id=1)
        finally:
            event.remove(engine, "before_cursor_execute", record)

        assert statements
        snapshot = "select transactions.amount from transactions where transactions.created_at <"
        for statement in statements:
            assert "transactions.device_id, transactions.user_id" not in statement
            assert not statement.startswith("select transactions.id, transactions.location")
            if statement.startswith(snapshot):
                continue        # the day's population snapshot: one column, once per UTC day
            # The one per-row read is the single previous transaction, limited to one row.
            if "order by" in statement:
                assert "limit" in statement

        # The snapshot is not re-read for another decision on the same day.
        statements.clear()
        event.listen(engine, "before_cursor_execute", record)
        try:
            features_at(db_session, _candidate(amount=999.0), user_id=1)
        finally:
            event.remove(engine, "before_cursor_execute", record)
        assert not any(statement.startswith(snapshot) for statement in statements)
