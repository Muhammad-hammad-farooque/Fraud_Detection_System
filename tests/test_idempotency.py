"""
Idempotency keys on POST /transactions/ (T-14).
"""
import pytest
from sqlalchemy.exc import IntegrityError

from app import models
from app.routers import transactions as transactions_router

BASE_TX = {"location": "Lahore", "amount": 100.0, "device_id": "device-001"}


def _post(client, headers, key=None, body=None):
    extra = {"Idempotency-Key": key} if key is not None else {}
    return client.post("/transactions/", json=body or BASE_TX, headers={**headers, **extra})


def _count(db, model):
    db.expire_all()
    return db.query(model).count()


class TestReplay:
    def test_same_key_twice_yields_one_transaction_and_identical_responses(self, client, auth_headers, db_session):
        first = _post(client, auth_headers, key="pay-001")
        second = _post(client, auth_headers, key="pay-001")
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert _count(db_session, models.Transaction) == 1

    def test_replay_is_marked(self, client, auth_headers):
        first = _post(client, auth_headers, key="pay-001")
        second = _post(client, auth_headers, key="pay-001")
        assert "Idempotent-Replayed" not in first.headers
        assert second.headers["Idempotent-Replayed"] == "true"

    def test_replay_does_not_rescore(self, client, auth_headers, monkeypatch):
        _post(client, auth_headers, key="pay-001")
        calls = []
        real = transactions_router.score_transaction
        monkeypatch.setattr(transactions_router, "score_transaction",
                            lambda *a, **k: calls.append(1) or real(*a, **k))
        _post(client, auth_headers, key="pay-001")
        assert calls == []

    def test_replay_writes_no_second_audit_row(self, client, auth_headers, db_session):
        _post(client, auth_headers, key="pay-001")
        _post(client, auth_headers, key="pay-001")
        assert _count(db_session, models.DecisionAudit) == 1

    def test_replay_opens_no_second_case(self, client, auth_headers, db_session, everything_reviews):
        first = _post(client, auth_headers, key="pay-001")
        assert first.json()["decision"] == "REVIEW"
        _post(client, auth_headers, key="pay-001")
        assert _count(db_session, models.Case) == 1

    def test_retries_do_not_inflate_velocity(self, client, auth_headers, db_session):
        """A15: five retries of one payment are one payment, not five."""
        for _ in range(6):
            _post(client, auth_headers, key="pay-001")
        after = _post(client, auth_headers, key="pay-002").json()

        db_session.expire_all()
        audit = db_session.query(models.DecisionAudit).filter_by(transaction_id=after["id"]).one()
        assert audit.feature_vector["velocity_2m"] == 1
        assert "R5_VELOCITY" not in {hit["rule_id"] for hit in audit.rule_hits}

    def test_without_the_fix_the_same_retries_would_trip_velocity(self, client, auth_headers, db_session):
        """Control: the same six sends without a key are six payments."""
        for _ in range(6):
            _post(client, auth_headers)
        after = _post(client, auth_headers).json()
        db_session.expire_all()
        audit = db_session.query(models.DecisionAudit).filter_by(transaction_id=after["id"]).one()
        assert audit.feature_vector["velocity_2m"] == 6
        assert "R5_VELOCITY" in {hit["rule_id"] for hit in audit.rule_hits}


class TestKeyScope:
    def test_different_keys_are_different_payments(self, client, auth_headers, db_session):
        a = _post(client, auth_headers, key="pay-001").json()
        b = _post(client, auth_headers, key="pay-002").json()
        assert a["id"] != b["id"]
        assert _count(db_session, models.Transaction) == 2

    def test_requests_without_a_key_are_unaffected(self, client, auth_headers, db_session):
        _post(client, auth_headers)
        _post(client, auth_headers)
        assert _count(db_session, models.Transaction) == 2

    def test_keys_are_scoped_to_the_customer(self, client, auth_headers, second_auth_headers, db_session):
        mine = _post(client, auth_headers, key="shared-key").json()
        theirs = _post(client, second_auth_headers, key="shared-key").json()
        assert mine["id"] != theirs["id"]
        assert mine["user_id"] != theirs["user_id"]
        assert _count(db_session, models.Transaction) == 2


class TestMisuse:
    @pytest.mark.parametrize("field, value", [("amount", 250.0), ("location", "Karachi"), ("device_id", "other")])
    def test_same_key_with_a_different_payment_is_refused(self, client, auth_headers, db_session, field, value):
        _post(client, auth_headers, key="pay-001")
        resp = _post(client, auth_headers, key="pay-001", body={**BASE_TX, field: value})
        assert resp.status_code == 422
        assert "different request" in resp.json()["detail"]
        assert _count(db_session, models.Transaction) == 1

    @pytest.mark.parametrize("key", ["", "x" * 256])
    def test_malformed_keys_are_rejected(self, client, auth_headers, db_session, key):
        assert _post(client, auth_headers, key=key).status_code == 422
        assert _count(db_session, models.Transaction) == 0

    def test_longest_allowed_key_works(self, client, auth_headers):
        assert _post(client, auth_headers, key="k" * 255).status_code == 200


class TestConcurrency:
    def test_a_racing_duplicate_returns_the_winner(self, client, auth_headers, db_session, monkeypatch):
        """Both requests missed the lookup; the constraint decides, the loser replays."""
        winner = _post(client, auth_headers, key="pay-001").json()

        # Make the second request's pre-check miss, as if it had run before the
        # first one committed, so it reaches the insert and hits the constraint.
        real_find = transactions_router.find_by_idempotency_key
        calls = []

        def racing_find(db, user_id, key):
            calls.append(key)
            return None if len(calls) == 1 else real_find(db, user_id, key)

        monkeypatch.setattr(transactions_router, "find_by_idempotency_key", racing_find)
        loser = _post(client, auth_headers, key="pay-001")
        assert len(calls) == 2          # the pre-check missed; the recovery lookup found the winner

        assert loser.status_code == 200
        assert loser.json() == winner
        assert loser.headers["Idempotent-Replayed"] == "true"
        assert _count(db_session, models.Transaction) == 1
        assert _count(db_session, models.DecisionAudit) == 1     # the loser's audit row rolled back

    def test_the_database_enforces_one_row_per_customer_and_key(self, client, auth_headers, db_session):
        user_id = _post(client, auth_headers, key="pay-001").json()["user_id"]
        db_session.add(models.Transaction(user_id=user_id, amount=1.0, location="L", device_id="d",
                                          idempotency_key="pay-001"))
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_a_collision_with_no_winner_to_return_still_surfaces(self, client, auth_headers, monkeypatch):
        """Defensive path: the constraint fired but the winning row cannot be found."""
        _post(client, auth_headers, key="pay-001")
        monkeypatch.setattr(transactions_router, "find_by_idempotency_key", lambda db, user_id, key: None)
        with pytest.raises(IntegrityError):
            _post(client, auth_headers, key="pay-001")

    def test_integrity_errors_unrelated_to_the_key_still_surface(self, client, auth_headers, monkeypatch):
        """Only a key collision is turned into a replay; anything else is a real error."""
        def broken_commit(self):
            raise IntegrityError("INSERT", {}, Exception("some other constraint"))

        monkeypatch.setattr("sqlalchemy.orm.Session.commit", broken_commit)
        with pytest.raises(IntegrityError):
            _post(client, auth_headers)
