"""
Unit tests for TransactionOutcome — ground truth decoupled from claims (T-05).
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app import models
from app.models import OutcomeSource
from scripts.monitor import get_ground_truth
from scripts.retrain import MIN_LABELLED_SAMPLES, load_labeled_data

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _user(db, user_id=1):
    db.add(models.User(id=user_id, name="A", email=f"a{user_id}@test.com", hashed_password="x"))
    db.commit()


def _transaction(db, predicted_fraud=False, ago_days=0, user_id=1):
    txn = models.Transaction(
        user_id=user_id,
        amount=100.0,
        location="Lahore",
        device_id="d1",
        predicted_fraud=predicted_fraud,
        created_at=(NOW - timedelta(days=ago_days)).replace(tzinfo=None),
    )
    db.add(txn)
    db.commit()
    return txn


def _outcome(db, txn, is_fraud_confirmed, source=OutcomeSource.ANALYST, confirmed_by=None):
    outcome = models.TransactionOutcome(
        transaction_id=txn.id,
        is_fraud_confirmed=is_fraud_confirmed,
        source=str(source),
        confirmed_by=confirmed_by,
        confirmed_at=NOW.replace(tzinfo=None),
    )
    db.add(outcome)
    db.commit()
    return outcome


def _claim(db, txn, status):
    claim = models.Claim(transaction_id=txn.id, reason="disputed", amount=50.0, status=status)
    db.add(claim)
    db.commit()
    return claim


class TestOutcomeModel:
    @pytest.mark.parametrize("source", list(OutcomeSource))
    def test_every_source_round_trips(self, db_session, source):
        _user(db_session)
        txn = _transaction(db_session)
        _outcome(db_session, txn, True, source=source)
        stored = db_session.query(models.TransactionOutcome).one()
        assert stored.source == str(source)
        assert stored.is_fraud_confirmed is True
        assert stored.transaction.id == txn.id

    def test_one_outcome_per_transaction(self, db_session):
        _user(db_session)
        txn = _transaction(db_session)
        _outcome(db_session, txn, True)
        with pytest.raises(IntegrityError):
            _outcome(db_session, txn, False, source=OutcomeSource.CHARGEBACK)
        db_session.rollback()

    def test_analyst_outcome_records_who_confirmed_it(self, db_session):
        _user(db_session)
        txn = _transaction(db_session)
        _outcome(db_session, txn, True, source=OutcomeSource.ANALYST, confirmed_by=1)
        assert db_session.query(models.TransactionOutcome).one().confirmed_by == 1


class TestRetrainLabels:
    def test_labels_come_only_from_outcomes(self, db_session):
        """A1/A4: an APPROVED claim is not a fraud label, and neither is a prediction."""
        _user(db_session)
        confirmed = _transaction(db_session)
        _outcome(db_session, confirmed, True, source=OutcomeSource.CHARGEBACK)

        claimed = _transaction(db_session)
        _claim(db_session, claimed, "APPROVED")

        predicted = _transaction(db_session, predicted_fraud=True)

        df = load_labeled_data(db_session)
        assert list(df["id"]) == [confirmed.id]
        assert list(df["fraud"]) == [1]
        assert claimed.id not in set(df["id"])
        assert predicted.id not in set(df["id"])

    def test_confirmed_legitimate_is_labelled_zero(self, db_session):
        _user(db_session)
        txn = _transaction(db_session, predicted_fraud=True)
        _outcome(db_session, txn, False, source=OutcomeSource.ANALYST)
        df = load_labeled_data(db_session)
        assert list(df["fraud"]) == [0]

    def test_no_outcomes_yields_an_empty_frame(self, db_session):
        _user(db_session)
        _transaction(db_session)
        assert load_labeled_data(db_session).empty

    def test_minimum_sample_gate_exists(self):
        assert MIN_LABELLED_SAMPLES >= 20


class TestMonitorGroundTruth:
    def test_ground_truth_ignores_claims(self, db_session):
        """A3: a REJECTED claim says nothing about whether the transaction was fraud."""
        _user(db_session)
        rejected = _transaction(db_session)
        _claim(db_session, rejected, "REJECTED")
        approved = _transaction(db_session)
        _claim(db_session, approved, "APPROVED")
        confirmed = _transaction(db_session)
        _outcome(db_session, confirmed, True)

        truth = get_ground_truth(db_session, NOW - timedelta(days=30))
        assert truth == {confirmed.id: True}

    def test_window_excludes_older_transactions(self, db_session):
        _user(db_session)
        old = _transaction(db_session, ago_days=10)
        _outcome(db_session, old, True)
        recent = _transaction(db_session, ago_days=1)
        _outcome(db_session, recent, False, source=OutcomeSource.CUSTOMER)

        truth = get_ground_truth(db_session, NOW - timedelta(days=3))
        assert truth == {recent.id: False}
