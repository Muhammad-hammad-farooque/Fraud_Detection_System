"""
Immutable decision audit trail — record, replay, refuse to change (T-11).
"""
import pathlib
import random
import re

import pytest

from app import models
from app.audit import replay
from app.features import FEATURE_ORDER, FeatureVector
from app.models import AuditImmutableError
from app.routers import transactions as transactions_router
from scripts.set_role import set_role

REPO = pathlib.Path(__file__).resolve().parent.parent


def _account(client, db, email, role="CUSTOMER"):
    resp = client.post("/auth/register", json={"name": email, "email": email, "password": "pw"})
    if role != "CUSTOMER":
        set_role(db, email, role)
    token = client.post("/auth/login", json={"email": email, "password": "pw"}).json()["access_token"]
    return {"id": resp.json()["id"], "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def people(client, db_session):
    return {
        "customer": _account(client, db_session, "customer@bank.com"),
        "analyst": _account(client, db_session, "analyst@bank.com", "ANALYST"),
    }


def _pay(client, people, amount=100.0, location="Lahore", device="device-001"):
    resp = client.post("/transactions/", json={"location": location, "amount": amount, "device_id": device},
                       headers=people["customer"]["headers"])
    assert resp.status_code == 200
    return resp.json()


def _audits(db, txn_id=None):
    db.expire_all()
    query = db.query(models.DecisionAudit)
    if txn_id is not None:
        query = query.filter(models.DecisionAudit.transaction_id == txn_id)
    return query.all()


# ── Recording ────────────────────────────────────────────────────────────────

class TestRecording:
    def test_every_scored_transaction_gets_exactly_one_row(self, client, people, db_session):
        txns = [_pay(client, people, amount) for amount in (10.0, 100.0, 9000.0)]
        for txn in txns:
            assert len(_audits(db_session, txn["id"])) == 1
        assert len(_audits(db_session)) == len(txns)

    def test_row_matches_the_transaction(self, client, people, db_session):
        txn = _pay(client, people, 9000.0, location="Quetta")
        audit = _audits(db_session, txn["id"])[0]
        assert audit.final_score == txn["risk_score"]
        assert audit.decision == txn["decision"]
        assert audit.policy_version == txn["policy_version"]
        assert audit.feature_vector["amount"] == 9000.0

    def test_row_carries_everything_needed_to_replay(self, client, people, db_session):
        txn = _pay(client, people, 9000.0)
        audit = _audits(db_session, txn["id"])[0]
        assert list(audit.feature_vector) == FEATURE_ORDER      # the full vector, all 42 features
        assert all(set(hit) == {"rule_id", "weight"} for hit in audit.rule_hits)
        assert set(audit.scoring_params) == {"rules_version", "total_rule_weight", "w_rules", "w_model"}
        assert {"allow_below", "step_up_below", "review_below", "high_value_amount"} <= set(audit.policy_config)
        assert audit.policy_context["amount"] == 9000.0
        assert audit.model_version

    def test_a_decision_without_its_inputs_cannot_be_audited(self, db_session):
        from app.audit import record_decision
        from app.policy import Decision, PolicyContext, load_policy_config
        from app.scoring import ScoreBreakdown

        blind = ScoreBreakdown(rule_hits=[], rule_score=0.0, model_probability=0.0,
                               final_score=0.0, model_version="m", feature_vector=None,
                               scoring_params={"w_rules": 0.7})
        with pytest.raises(ValueError, match="feature vector"):
            record_decision(db_session, models.Transaction(id=1), blind, Decision.ALLOW,
                            load_policy_config(), PolicyContext(amount=1.0))

    def test_audit_and_transaction_commit_together(self, client, people, db_session, monkeypatch,
                                                  everything_reviews):
        """If anything after scoring fails, neither the decision nor its record survives."""

        def explode(*args, **kwargs):
            raise RuntimeError("case store unavailable")

        monkeypatch.setattr(transactions_router, "open_case", explode)
        with pytest.raises(RuntimeError):
            _pay(client, people)

        db_session.expire_all()
        assert db_session.query(models.Transaction).count() == 0
        assert db_session.query(models.DecisionAudit).count() == 0


# ── Replay ───────────────────────────────────────────────────────────────────

class TestReplay:
    def test_replay_reproduces_the_score_bit_for_bit(self, client, people, db_session):
        txn = _pay(client, people, 9000.0, location="Quetta")
        audit = _audits(db_session, txn["id"])[0]
        result = replay(audit)
        assert result.final_score == audit.final_score      # exact, not approximate
        assert result.decision.value == audit.decision
        assert result.matches is True
        assert result.rules_still_agree is True

    def test_replay_holds_across_many_decisions(self, client, people, db_session):
        rng = random.Random(11)
        for _ in range(40):
            _pay(client, people,
                 amount=rng.choice([5.0, 120.0, 800.0, 5000.01, 25_000.0]),
                 location=rng.choice(["Lahore", "Karachi", "Quetta", "Multan"]),
                 device=rng.choice(["d1", "d2"]))
        audits = _audits(db_session)
        assert len(audits) == 40
        assert {a.decision for a in audits} >= {"ALLOW"}
        for audit in audits:
            result = replay(audit)
            assert result.final_score == audit.final_score, audit.id
            assert result.decision.value == audit.decision, audit.id

    def test_replay_uses_the_thresholds_in_force_at_the_time(self, client, people, db_session, rules_config):
        """Retuning the policy afterwards must not rewrite history."""
        txn = _pay(client, people, 100.0)
        assert txn["decision"] == "ALLOW"

        # Everything would now REJECT, and every rule weighs differently.
        rules_config(
            policy={"allow_below": 0, "step_up_below": 0, "review_below": 0},
            scoring={"w_rules": 0.5, "w_model": 0.5},
        )

        result = replay(_audits(db_session, txn["id"])[0])
        assert result.decision.value == "ALLOW"
        assert result.matches is True

    def test_rows_written_before_t16_still_replay(self, client, people, db_session):
        """Pre-T-16 audit rows stored only the five baseline features."""
        txn = _pay(client, people, 9000.0)
        audit = _audits(db_session, txn["id"])[0]
        db_session.expunge(audit)
        audit.feature_vector = {k: audit.feature_vector[k] for k in FEATURE_ORDER[:5]}
        result = replay(audit)
        assert result.matches is True
        assert result.rules_still_agree is True

    def test_replay_notices_a_tampered_score(self, client, people, db_session):
        txn = _pay(client, people, 9000.0)
        audit = _audits(db_session, txn["id"])[0]
        db_session.expunge(audit)              # work on a detached copy, never the stored row
        audit.final_score = audit.final_score + 1e-12
        assert replay(audit).matches is False

    def test_replay_flags_rules_that_changed_since(self, client, people, db_session):
        txn = _pay(client, people, 9000.0)
        audit = _audits(db_session, txn["id"])[0]
        db_session.expunge(audit)
        audit.rule_hits = [{"rule_id": "R_RETIRED", "weight": 0.25}]
        assert replay(audit).rules_still_agree is False


# ── Append-only ──────────────────────────────────────────────────────────────

class TestAppendOnly:
    def test_update_is_refused(self, client, people, db_session):
        _pay(client, people)
        audit = _audits(db_session)[0]
        audit.decision = "ALLOW_OVERRIDE"
        with pytest.raises(AuditImmutableError):
            db_session.commit()
        db_session.rollback()

    def test_delete_is_refused(self, client, people, db_session):
        _pay(client, people)
        audit = _audits(db_session)[0]
        db_session.delete(audit)
        with pytest.raises(AuditImmutableError):
            db_session.commit()
        db_session.rollback()
        assert len(_audits(db_session)) == 1

    def test_no_update_or_delete_path_exists_in_the_codebase(self):
        """Nothing in app/ or scripts/ writes to DecisionAudit except record_decision."""
        offenders = []
        for path in list((REPO / "app").rglob("*.py")) + list((REPO / "scripts").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"DecisionAudit[^\n]*\.(update|delete)\(", text):
                offenders.append(f"{path.name}: {match.group(0)}")
            if re.search(r"(UPDATE|DELETE FROM)\s+decision_audits", text, re.IGNORECASE):
                offenders.append(f"{path.name}: raw SQL on decision_audits")
        assert offenders == []

    def test_migration_adds_a_database_level_guard(self):
        sql = (REPO / "migrations" / "004_decision_audits.sql").read_text(encoding="utf-8")
        assert "BEFORE UPDATE OR DELETE ON decision_audits" in sql


# ── Endpoint ─────────────────────────────────────────────────────────────────

class TestAuditEndpoint:
    def test_analyst_sees_the_record_and_its_replay(self, client, people):
        txn = _pay(client, people, 9000.0)
        resp = client.get(f"/analyst/transactions/{txn['id']}/audit", headers=people["analyst"]["headers"])
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["final_score"] == txn["risk_score"]
        assert rows[0]["replay_matches"] is True
        assert rows[0]["rules_still_agree"] is True

    def test_customers_cannot_read_audit_records(self, client, people):
        txn = _pay(client, people)
        resp = client.get(f"/analyst/transactions/{txn['id']}/audit", headers=people["customer"]["headers"])
        assert resp.status_code == 403

    def test_anonymous_callers_are_refused(self, client, people):
        txn = _pay(client, people)
        assert client.get(f"/analyst/transactions/{txn['id']}/audit").status_code == 401

    def test_missing_transaction_is_404(self, client, people):
        resp = client.get("/analyst/transactions/999/audit", headers=people["analyst"]["headers"])
        assert resp.status_code == 404
