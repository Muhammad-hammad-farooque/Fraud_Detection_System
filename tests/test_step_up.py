"""
Step-up authentication — challenge, verify, expire, escalate (T-14b).
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import OperationalError

from app import models
from app.models import ChallengeStatus
from app.services import step_up
from app.services.step_up import expire_stale
from scripts.set_role import set_role

BASE_TX = {"location": "Lahore", "amount": 100.0, "device_id": "device-001"}


class CapturingSender:
    def __init__(self):
        self.sent = []

    def send(self, user, challenge, code):
        self.sent.append({"email": user.email, "challenge_id": challenge.id, "code": code})

    @property
    def last_code(self):
        return self.sent[-1]["code"]


@pytest.fixture
def sender():
    capture = CapturingSender()
    previous = step_up.set_sender(capture)
    yield capture
    step_up.set_sender(previous)


@pytest.fixture
def everything_steps_up(rules_config):
    return rules_config(policy={"allow_below": 0, "step_up_below": 1.01, "review_below": 1.01})


@pytest.fixture
def stepped(client, auth_headers, sender, everything_steps_up):
    txn = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
    assert txn["decision"] == "STEP_UP"
    return txn


def _verify(client, headers, txn_id, code):
    return client.post(f"/transactions/{txn_id}/step-up", json={"code": code}, headers=headers)


def _wrong(code):
    return "000000" if code != "000000" else "111111"


def _challenge(db, txn_id):
    db.expire_all()
    return db.query(models.StepUpChallenge).filter_by(transaction_id=txn_id).one()


# ── Issuing ──────────────────────────────────────────────────────────────────

class TestIssue:
    def test_step_up_decision_issues_a_pending_challenge(self, stepped):
        info = stepped["step_up"]
        assert info["status"] == "PENDING"
        assert info["method"] == "OTP"
        assert info["attempts_remaining"] == 3
        assert stepped["resolved_decision"] is None

    def test_code_is_delivered_once_and_is_six_digits(self, stepped, sender):
        assert len(sender.sent) == 1
        assert sender.sent[0]["challenge_id"] == stepped["step_up"]["id"]
        assert len(sender.last_code) == 6 and sender.last_code.isdigit()

    def test_code_is_not_in_the_response_by_default(self, stepped):
        assert stepped["step_up"]["dev_code"] is None

    def test_dev_echo_puts_the_code_in_the_response(self, client, auth_headers, sender,
                                                    everything_steps_up, monkeypatch):
        monkeypatch.setenv("STEP_UP_DEV_ECHO", "true")
        txn = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        assert txn["step_up"]["dev_code"] == sender.last_code

    def test_code_is_stored_only_as_a_keyed_hash(self, stepped, sender, db_session):
        challenge = _challenge(db_session, stepped["id"])
        assert sender.last_code not in challenge.code_hash
        assert len(challenge.code_hash) == 64

    def test_other_decisions_issue_no_challenge(self, client, auth_headers, sender, db_session):
        txn = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        assert txn["decision"] == "ALLOW"
        assert txn["step_up"] is None
        assert sender.sent == []

    def test_expiry_comes_from_config(self, client, auth_headers, sender, everything_steps_up,
                                      monkeypatch, db_session):
        monkeypatch.setenv("STEP_UP_TTL_SECONDS", "60")
        txn = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        challenge = _challenge(db_session, txn["id"])
        lifetime = challenge.expires_at - challenge.created_at
        assert lifetime == timedelta(seconds=60)

    def test_nothing_is_sent_when_the_payment_rolls_back(self, client, auth_headers, sender,
                                                        everything_steps_up, monkeypatch, db_session):
        def failing_commit(self):
            raise OperationalError("COMMIT", {}, Exception("database went away"))

        monkeypatch.setattr("sqlalchemy.orm.Session.commit", failing_commit)
        with pytest.raises(OperationalError):
            client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        monkeypatch.undo()
        assert sender.sent == []
        db_session.expire_all()
        assert db_session.query(models.StepUpChallenge).count() == 0


# ── Verifying ────────────────────────────────────────────────────────────────

class TestVerify:
    def test_right_code_allows_the_payment(self, client, auth_headers, stepped, sender):
        resp = _verify(client, auth_headers, stepped["id"], sender.last_code)
        assert resp.status_code == 200
        body = resp.json()
        assert body["step_up"]["status"] == "PASSED"
        assert body["resolved_decision"] == "ALLOW"
        assert body["decision"] == "STEP_UP"          # the engine's decision is never rewritten

    def test_wrong_code_spends_an_attempt(self, client, auth_headers, stepped, sender):
        body = _verify(client, auth_headers, stepped["id"], _wrong(sender.last_code)).json()
        assert body["step_up"]["status"] == "PENDING"
        assert body["step_up"]["attempts_remaining"] == 2
        assert body["resolved_decision"] is None

    def test_right_code_after_a_wrong_one_still_passes(self, client, auth_headers, stepped, sender):
        _verify(client, auth_headers, stepped["id"], _wrong(sender.last_code))
        body = _verify(client, auth_headers, stepped["id"], sender.last_code).json()
        assert body["step_up"]["status"] == "PASSED"

    def test_exhausting_attempts_holds_the_payment_for_an_analyst(self, client, auth_headers, stepped,
                                                                   sender, db_session):
        for _ in range(3):
            body = _verify(client, auth_headers, stepped["id"], _wrong(sender.last_code)).json()
        assert body["step_up"]["status"] == "FAILED"
        assert body["resolved_decision"] is None       # held, not silently rejected

        db_session.expire_all()
        case = db_session.query(models.Case).filter_by(transaction_id=stepped["id"]).one()
        assert case.source == "STEP_UP_FAILED"
        assert case.status == "OPEN"

    def test_an_analyst_settles_a_failed_challenge(self, client, auth_headers, stepped, sender, db_session):
        for _ in range(3):
            _verify(client, auth_headers, stepped["id"], _wrong(sender.last_code))
        client.post("/auth/register", json={"name": "A", "email": "analyst@bank.com", "password": "pw"})
        set_role(db_session, "analyst@bank.com", "ANALYST")
        token = client.post("/auth/login", json={"email": "analyst@bank.com", "password": "pw"}).json()["access_token"]
        analyst = {"Authorization": f"Bearer {token}"}

        case_id = client.get("/analyst/cases", headers=analyst).json()[0]["id"]
        client.post(f"/analyst/cases/{case_id}/assign", headers=analyst)
        client.post(f"/analyst/cases/{case_id}/resolve", json={"is_fraud_confirmed": True}, headers=analyst)

        txn = client.get(f"/transactions/{stepped['id']}", headers=auth_headers).json()
        assert txn["resolved_decision"] == "REJECT"

    @pytest.mark.parametrize("finish", ["pass", "fail"])
    def test_a_settled_challenge_cannot_be_answered_again(self, client, auth_headers, stepped, sender, finish):
        if finish == "pass":
            _verify(client, auth_headers, stepped["id"], sender.last_code)
        else:
            for _ in range(3):
                _verify(client, auth_headers, stepped["id"], _wrong(sender.last_code))
        resp = _verify(client, auth_headers, stepped["id"], sender.last_code)
        assert resp.status_code == 409

    def test_no_path_writes_a_training_label(self, client, auth_headers, sender, everything_steps_up, db_session):
        """Passing an OTP is not proof of legitimacy; failing one is not proof of fraud."""
        passed = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        _verify(client, auth_headers, passed["id"], sender.last_code)
        failed = client.post("/transactions/", json={**BASE_TX, "amount": 101.0}, headers=auth_headers).json()
        for _ in range(3):
            _verify(client, auth_headers, failed["id"], _wrong(sender.last_code))
        db_session.expire_all()
        assert db_session.query(models.TransactionOutcome).count() == 0


# ── Expiry ───────────────────────────────────────────────────────────────────

def _backdate(db, txn_id, seconds_ago=1):
    challenge = _challenge(db, txn_id)
    challenge.expires_at = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    db.commit()


class TestExpiry:
    def test_answering_after_the_deadline_rejects(self, client, auth_headers, stepped, sender, db_session):
        _backdate(db_session, stepped["id"])
        body = _verify(client, auth_headers, stepped["id"], sender.last_code).json()
        assert body["step_up"]["status"] == "EXPIRED"
        assert body["resolved_decision"] == "REJECT"

    def test_the_sweep_settles_abandoned_challenges(self, client, auth_headers, sender,
                                                   everything_steps_up, db_session):
        stale = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        fresh = client.post("/transactions/", json={**BASE_TX, "amount": 101.0}, headers=auth_headers).json()
        _backdate(db_session, stale["id"])

        assert expire_stale(db_session) == 1
        db_session.commit()
        assert _challenge(db_session, stale["id"]).status == ChallengeStatus.EXPIRED.value
        assert _challenge(db_session, fresh["id"]).status == ChallengeStatus.PENDING.value
        stale_txn = client.get(f"/transactions/{stale['id']}", headers=auth_headers).json()
        assert stale_txn["resolved_decision"] == "REJECT"

    def test_the_sweep_leaves_settled_challenges_alone(self, client, auth_headers, stepped, sender, db_session):
        _verify(client, auth_headers, stepped["id"], sender.last_code)
        _backdate(db_session, stepped["id"])
        assert expire_stale(db_session) == 0
        assert _challenge(db_session, stepped["id"]).status == ChallengeStatus.PASSED.value

    def test_the_sweep_script(self, client, auth_headers, stepped, db_session, monkeypatch):
        from scripts import expire_challenges
        from tests.conftest import TestingSessionLocal

        _backdate(db_session, stepped["id"])
        monkeypatch.setattr(expire_challenges, "SessionLocal", TestingSessionLocal)
        assert expire_challenges.run() == 1
        assert _challenge(db_session, stepped["id"]).status == ChallengeStatus.EXPIRED.value


# ── Access and interplay ─────────────────────────────────────────────────────

class TestAccess:
    def test_another_customer_cannot_answer(self, client, stepped, sender, second_auth_headers):
        assert _verify(client, second_auth_headers, stepped["id"], sender.last_code).status_code == 404

    def test_transaction_without_a_challenge_is_404(self, client, auth_headers, sender):
        txn = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        assert _verify(client, auth_headers, txn["id"], "123456").status_code == 404

    def test_staff_cannot_use_the_customer_route(self, client, stepped, sender, db_session):
        client.post("/auth/register", json={"name": "A", "email": "a@bank.com", "password": "pw"})
        set_role(db_session, "a@bank.com", "ANALYST")
        token = client.post("/auth/login", json={"email": "a@bank.com", "password": "pw"}).json()["access_token"]
        resp = _verify(client, {"Authorization": f"Bearer {token}"}, stepped["id"], sender.last_code)
        assert resp.status_code == 403

    def test_anonymous_is_refused(self, client, stepped, sender):
        assert client.post(f"/transactions/{stepped['id']}/step-up", json={"code": "1"}).status_code == 401

    def test_empty_code_is_rejected(self, client, auth_headers, stepped):
        assert _verify(client, auth_headers, stepped["id"], "").status_code == 422

    def test_an_idempotent_replay_sends_no_second_code(self, client, auth_headers, sender,
                                                       everything_steps_up, monkeypatch, db_session):
        monkeypatch.setenv("STEP_UP_DEV_ECHO", "true")
        headers = {**auth_headers, "Idempotency-Key": "pay-001"}
        first = client.post("/transactions/", json=BASE_TX, headers=headers).json()
        replay = client.post("/transactions/", json=BASE_TX, headers=headers).json()
        assert len(sender.sent) == 1
        assert replay["step_up"]["id"] == first["step_up"]["id"]
        assert replay["step_up"]["dev_code"] is None     # the code is only ever echoed once
        db_session.expire_all()
        assert db_session.query(models.StepUpChallenge).count() == 1
