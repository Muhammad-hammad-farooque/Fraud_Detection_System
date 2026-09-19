"""
Analyst case queue — open, assign, resolve, label (T-12).
"""
from datetime import datetime, timedelta, timezone

import pytest

from app import models
from app.models import CaseSource, CaseStatus, OutcomeSource
from app.services import case_service
from app.services.case_service import CaseConfig, is_sla_breached, list_queue, open_case
from scripts.retrain import load_labeled_data
from scripts.set_role import set_role

BASE_TX = {"location": "Lahore", "amount": 100.0, "device_id": "device-001"}


def _account(client, db, email, role="CUSTOMER"):
    resp = client.post("/auth/register", json={"name": email, "email": email, "password": "pw"})
    assert resp.status_code == 201
    if role != "CUSTOMER":
        set_role(db, email, role)
    token = client.post("/auth/login", json={"email": email, "password": "pw"}).json()["access_token"]
    return {"id": resp.json()["id"], "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def everything_reviews(monkeypatch):
    """Widen the REVIEW band to cover every score, via the policy's own config."""
    monkeypatch.setenv("POLICY_ALLOW_BELOW", "0")
    monkeypatch.setenv("POLICY_STEP_UP_BELOW", "0")
    monkeypatch.setenv("POLICY_REVIEW_BELOW", "1.01")


@pytest.fixture
def people(client, db_session):
    return {
        "customer": _account(client, db_session, "customer@bank.com"),
        "analyst": _account(client, db_session, "analyst@bank.com", "ANALYST"),
        "analyst2": _account(client, db_session, "analyst2@bank.com", "ANALYST"),
        "admin": _account(client, db_session, "admin@bank.com", "ADMIN"),
    }


@pytest.fixture
def reviewed(client, people, everything_reviews):
    """A customer transaction the policy sent to REVIEW, and its case."""
    txn = client.post("/transactions/", json=BASE_TX, headers=people["customer"]["headers"]).json()
    assert txn["decision"] == "REVIEW"
    cases = client.get("/analyst/cases", headers=people["analyst"]["headers"]).json()
    assert len(cases) == 1
    return {"txn": txn, "case": cases[0]}


def _assign(client, people, case_id, who="analyst"):
    return client.post(f"/analyst/cases/{case_id}/assign", headers=people[who]["headers"])


def _resolve(client, people, case_id, is_fraud, who="analyst", notes="checked"):
    return client.post(f"/analyst/cases/{case_id}/resolve",
                       json={"is_fraud_confirmed": is_fraud, "notes": notes},
                       headers=people[who]["headers"])


# ── Case creation ────────────────────────────────────────────────────────────

class TestCaseCreation:
    def test_review_decision_opens_a_case(self, reviewed):
        case = reviewed["case"]
        assert case["transaction_id"] == reviewed["txn"]["id"]
        assert case["source"] == "POLICY_REVIEW"
        assert case["status"] == "OPEN"
        assert case["sla_breached"] is False

    def test_allowed_transaction_opens_no_case(self, client, people):
        txn = client.post("/transactions/", json=BASE_TX, headers=people["customer"]["headers"]).json()
        assert txn["decision"] == "ALLOW"
        assert client.get("/analyst/cases", headers=people["analyst"]["headers"]).json() == []

    def test_manual_review_claim_opens_a_case(self, client, people, db_session):
        customer = people["customer"]["headers"]
        txn = client.post("/transactions/", json=BASE_TX, headers=customer).json()
        first = client.post("/claims/", json={"transaction_id": txn["id"], "reason": "a", "amount": 10.0},
                            headers=customer).json()
        second = client.post("/claims/", json={"transaction_id": txn["id"], "reason": "b", "amount": 10.0},
                             headers=customer).json()
        assert first["status"] == "APPROVED"          # clean first claim, no case needed
        assert second["status"] == "MANUAL_REVIEW"    # a repeat claim needs a human

        cases = client.get("/analyst/cases", headers=people["analyst"]["headers"]).json()
        assert len(cases) == 1
        assert cases[0]["source"] == "CLAIM_REVIEW"
        assert cases[0]["claim_id"] == second["id"]

    def test_claim_on_a_reviewed_transaction_joins_the_existing_case(self, client, people, reviewed, db_session):
        customer = people["customer"]["headers"]
        txn_id = reviewed["txn"]["id"]
        client.post("/claims/", json={"transaction_id": txn_id, "reason": "a", "amount": 10.0}, headers=customer)
        claim = client.post("/claims/", json={"transaction_id": txn_id, "reason": "b", "amount": 10.0},
                            headers=customer).json()
        assert claim["status"] == "MANUAL_REVIEW"

        cases = client.get("/analyst/cases", headers=people["analyst"]["headers"]).json()
        assert len(cases) == 1
        assert cases[0]["source"] == "POLICY_REVIEW"
        assert cases[0]["claim_id"] == claim["id"]

    @pytest.mark.parametrize("threshold, expected", [("0", "HIGH"), ("1.1", "NORMAL")])
    def test_priority_follows_risk_score(self, client, people, everything_reviews, monkeypatch, threshold, expected):
        monkeypatch.setenv("CASE_HIGH_PRIORITY_SCORE", threshold)
        client.post("/transactions/", json=BASE_TX, headers=people["customer"]["headers"])
        case = client.get("/analyst/cases", headers=people["analyst"]["headers"]).json()[0]
        assert case["priority"] == expected

    def test_high_priority_gets_the_tighter_sla(self, db_session):
        cfg = CaseConfig(high_priority_score=0.6, high_priority_sla_hours=4, normal_sla_hours=24)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        db_session.add(models.User(id=1, name="c", email="c@b.com", hashed_password="x"))
        high = models.Transaction(user_id=1, amount=1, location="L", device_id="d", risk_score=0.9, decision="REVIEW")
        low = models.Transaction(user_id=1, amount=1, location="L", device_id="d", risk_score=0.2, decision="REVIEW")
        db_session.add_all([high, low])
        db_session.flush()
        high_case = open_case(db_session, high, CaseSource.POLICY_REVIEW, now=now, cfg=cfg)
        low_case = open_case(db_session, low, CaseSource.POLICY_REVIEW, now=now, cfg=cfg)
        assert high_case.sla_due_at == now + timedelta(hours=4)
        assert low_case.sla_due_at == now + timedelta(hours=24)


# ── The queue ────────────────────────────────────────────────────────────────

@pytest.fixture
def seeded_queue(db_session):
    """Four cases with controlled risk scores and ages."""
    owner = models.User(name="queue owner", email="queue@b.com", hashed_password="x")
    db_session.add(owner)
    db_session.flush()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    specs = [  # (risk_score, hours after base, priority, status)
        (0.50, 0, "NORMAL", CaseStatus.OPEN),
        (0.90, 5, "HIGH", CaseStatus.OPEN),
        (0.90, 1, "HIGH", CaseStatus.ASSIGNED),
        (0.65, 2, "HIGH", CaseStatus.RESOLVED),
    ]
    ids = []
    for score, hours, priority, status in specs:
        txn = models.Transaction(user_id=owner.id, amount=1, location="L", device_id="d",
                                 risk_score=score, decision="REVIEW")
        db_session.add(txn)
        db_session.flush()
        case = models.Case(transaction_id=txn.id, source="POLICY_REVIEW", status=status.value,
                           priority=priority, created_at=base + timedelta(hours=hours),
                           sla_due_at=base + timedelta(hours=hours + 4))
        db_session.add(case)
        db_session.flush()
        ids.append(case.id)
    db_session.commit()
    return ids


class TestQueue:
    def test_sorted_by_risk_then_age(self, db_session, seeded_queue):
        ordered = [c.id for c in list_queue(db_session)]
        # 0.90 (older) , 0.90 (newer) , 0.65 , 0.50
        assert ordered == [seeded_queue[2], seeded_queue[1], seeded_queue[3], seeded_queue[0]]

    def test_filter_by_status(self, db_session, seeded_queue):
        open_ids = {c.id for c in list_queue(db_session, status=CaseStatus.OPEN)}
        assert open_ids == {seeded_queue[0], seeded_queue[1]}

    def test_filter_by_priority(self, db_session, seeded_queue):
        assert {c.id for c in list_queue(db_session, priority="NORMAL")} == {seeded_queue[0]}

    def test_pagination(self, db_session, seeded_queue):
        first = [c.id for c in list_queue(db_session, limit=2, offset=0)]
        second = [c.id for c in list_queue(db_session, limit=2, offset=2)]
        assert len(first) == 2 and len(second) == 2
        assert not set(first) & set(second)

    def test_api_paginates_and_rejects_bad_limits(self, client, people, seeded_queue):
        headers = people["analyst"]["headers"]
        assert len(client.get("/analyst/cases?limit=3", headers=headers).json()) == 3
        assert client.get("/analyst/cases?limit=0", headers=headers).status_code == 422
        assert client.get("/analyst/cases?limit=201", headers=headers).status_code == 422

    def test_sla_breach_is_flagged(self, db_session, seeded_queue):
        case = db_session.get(models.Case, seeded_queue[0])
        due = case.sla_due_at.replace(tzinfo=timezone.utc)
        assert is_sla_breached(case, now=due - timedelta(minutes=1)) is False
        assert is_sla_breached(case, now=due + timedelta(minutes=1)) is True

    def test_resolved_cases_are_never_in_breach(self, db_session, seeded_queue):
        resolved = db_session.get(models.Case, seeded_queue[3])
        assert is_sla_breached(resolved, now=datetime(2030, 1, 1, tzinfo=timezone.utc)) is False

    def test_api_reports_the_breach(self, client, people, seeded_queue):
        """Seeded deadlines are in 2026-01, long past, so open ones are breached."""
        cases = client.get("/analyst/cases", headers=people["analyst"]["headers"]).json()
        by_status = {c["status"]: c["sla_breached"] for c in cases}
        assert by_status["OPEN"] is True
        assert by_status["RESOLVED"] is False


# ── Lifecycle ────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_full_lifecycle_writes_a_label(self, client, people, reviewed, db_session):
        """Create → assign → resolve → the label is exactly what retraining reads."""
        case_id = reviewed["case"]["id"]

        assigned = _assign(client, people, case_id).json()
        assert assigned["status"] == "ASSIGNED"
        assert assigned["assigned_to"] == people["analyst"]["id"]

        resp = _resolve(client, people, case_id, is_fraud=True, notes="card reported stolen")
        assert resp.status_code == 200
        outcome = resp.json()
        assert outcome["source"] == "ANALYST"
        assert outcome["is_fraud_confirmed"] is True
        assert outcome["confirmed_by"] == people["analyst"]["id"]

        case = client.get(f"/analyst/cases/{case_id}", headers=people["analyst"]["headers"]).json()
        assert case["status"] == "RESOLVED"
        assert case["resolved_by"] == people["analyst"]["id"]
        assert case["notes"] == "card reported stolen"

        db_session.expire_all()
        outcomes = db_session.query(models.TransactionOutcome).all()
        assert len(outcomes) == 1
        labels = load_labeled_data(db_session)
        assert list(labels["id"]) == [reviewed["txn"]["id"]]
        assert list(labels["fraud"]) == [1]

    def test_resolution_is_visible_to_the_customer(self, client, people, reviewed):
        case_id = reviewed["case"]["id"]
        _assign(client, people, case_id)
        _resolve(client, people, case_id, is_fraud=False)
        txn = client.get(f"/transactions/{reviewed['txn']['id']}", headers=people["customer"]["headers"]).json()
        assert txn["decision"] == "REVIEW"            # the engine's decision is never rewritten
        assert txn["resolved_decision"] == "ALLOW"    # the human outcome sits beside it

    def test_confirmed_fraud_resolves_to_reject(self, client, people, reviewed):
        case_id = reviewed["case"]["id"]
        _assign(client, people, case_id)
        _resolve(client, people, case_id, is_fraud=True)
        txn = client.get(f"/transactions/{reviewed['txn']['id']}", headers=people["customer"]["headers"]).json()
        assert txn["resolved_decision"] == "REJECT"

    @pytest.mark.parametrize("is_fraud, claim_status", [(True, "APPROVED"), (False, "REJECTED")])
    def test_attached_claim_is_settled(self, client, people, reviewed, is_fraud, claim_status):
        customer = people["customer"]["headers"]
        txn_id = reviewed["txn"]["id"]
        client.post("/claims/", json={"transaction_id": txn_id, "reason": "a", "amount": 10.0}, headers=customer)
        claim = client.post("/claims/", json={"transaction_id": txn_id, "reason": "b", "amount": 10.0},
                            headers=customer).json()
        assert claim["status"] == "MANUAL_REVIEW"

        case_id = reviewed["case"]["id"]
        _assign(client, people, case_id)
        _resolve(client, people, case_id, is_fraud=is_fraud)
        settled = client.get(f"/claims/{claim['id']}", headers=customer).json()
        assert settled["status"] == claim_status


class TestLifecycleRules:
    def test_another_analyst_cannot_take_an_assigned_case(self, client, people, reviewed):
        case_id = reviewed["case"]["id"]
        _assign(client, people, case_id)
        assert _assign(client, people, case_id, who="analyst2").status_code == 409

    def test_reassigning_to_yourself_is_a_no_op(self, client, people, reviewed):
        case_id = reviewed["case"]["id"]
        first = _assign(client, people, case_id).json()
        second = _assign(client, people, case_id)
        assert second.status_code == 200
        assert second.json()["assigned_at"] == first["assigned_at"]

    def test_analyst_must_hold_the_case_to_resolve_it(self, client, people, reviewed):
        assert _resolve(client, people, reviewed["case"]["id"], is_fraud=True).status_code == 409

    def test_admin_can_resolve_without_assignment(self, client, people, reviewed):
        resp = _resolve(client, people, reviewed["case"]["id"], is_fraud=False, who="admin")
        assert resp.status_code == 200
        assert resp.json()["confirmed_by"] == people["admin"]["id"]

    def test_a_case_resolves_exactly_once(self, client, people, reviewed, db_session):
        case_id = reviewed["case"]["id"]
        _assign(client, people, case_id)
        assert _resolve(client, people, case_id, is_fraud=True).status_code == 200
        assert _resolve(client, people, case_id, is_fraud=False).status_code == 409
        db_session.expire_all()
        assert db_session.query(models.TransactionOutcome).count() == 1

    def test_resolved_case_cannot_be_reassigned(self, client, people, reviewed):
        case_id = reviewed["case"]["id"]
        _assign(client, people, case_id)
        _resolve(client, people, case_id, is_fraud=True)
        assert _assign(client, people, case_id, who="analyst2").status_code == 409

    def test_existing_outcome_is_never_overwritten(self, client, people, reviewed, db_session):
        """A chargeback that landed first stands; the case cannot relabel it."""
        db_session.add(models.TransactionOutcome(
            transaction_id=reviewed["txn"]["id"], is_fraud_confirmed=True,
            source=OutcomeSource.CHARGEBACK.value,
        ))
        db_session.commit()
        case_id = reviewed["case"]["id"]
        _assign(client, people, case_id)
        resp = _resolve(client, people, case_id, is_fraud=False)
        assert resp.status_code == 409
        assert "CHARGEBACK" in resp.json()["detail"]

    def test_missing_case_is_404(self, client, people):
        headers = people["analyst"]["headers"]
        assert client.get("/analyst/cases/999", headers=headers).status_code == 404
        assert client.post("/analyst/cases/999/assign", headers=headers).status_code == 404
        assert _resolve(client, people, 999, is_fraud=True).status_code == 404


class TestAccess:
    @pytest.mark.parametrize("method, path", [
        ("GET", "/analyst/cases"),
        ("GET", "/analyst/cases/{case_id}"),
        ("POST", "/analyst/cases/{case_id}/assign"),
        ("POST", "/analyst/cases/{case_id}/resolve"),
    ])
    def test_customers_are_refused(self, client, people, reviewed, method, path):
        url = path.format(case_id=reviewed["case"]["id"])
        body = {"is_fraud_confirmed": True} if url.endswith("resolve") else None
        resp = client.request(method, url, json=body, headers=people["customer"]["headers"])
        assert resp.status_code == 403
        anonymous = client.request(method, url, json=body)
        assert anonymous.status_code == 401
