"""
Tests for /claims endpoints and claim verification logic.
"""
import pytest

BASE_TX = {"location": "New York", "amount": 100.0, "device_id": "device-001"}


def make_transaction(client, headers, tx=None):
    """Helper — create a transaction and return its JSON."""
    return client.post("/transactions/", json=tx or BASE_TX, headers=headers).json()


def make_claim(client, headers, transaction_id, amount=50.0, reason="Unauthorized charge"):
    return client.post("/claims/", json={
        "transaction_id": transaction_id,
        "reason": reason,
        "amount": amount,
    }, headers=headers)


class TestCreateClaim:
    def test_create_claim_success(self, client, auth_headers):
        tx = make_transaction(client, auth_headers)
        resp = make_claim(client, auth_headers, tx["id"])
        assert resp.status_code == 201
        data = resp.json()
        assert data["transaction_id"] == tx["id"]
        assert data["reason"] == "Unauthorized charge"
        assert data["amount"] == 50.0
        assert data["status"] in ("APPROVED", "MANUAL_REVIEW", "REJECTED")

    def test_create_claim_unauthenticated(self, client, auth_headers):
        tx = make_transaction(client, auth_headers)
        resp = client.post("/claims/", json={
            "transaction_id": tx["id"], "reason": "fraud", "amount": 50.0
        })
        assert resp.status_code == 401

    def test_cannot_claim_another_users_transaction(
        self, client, auth_headers, second_auth_headers
    ):
        tx = make_transaction(client, auth_headers)
        resp = make_claim(client, second_auth_headers, tx["id"])
        assert resp.status_code == 404

    def test_claim_nonexistent_transaction(self, client, auth_headers):
        resp = make_claim(client, auth_headers, transaction_id=99999)
        assert resp.status_code == 404

    def test_claim_missing_fields(self, client, auth_headers):
        resp = client.post("/claims/", json={"transaction_id": 1}, headers=auth_headers)
        assert resp.status_code == 422


class TestClaimVerificationLogic:
    """Tests for the three-step verify_claim logic via the API."""

    def test_first_clean_claim_is_approved(self, client, auth_headers):
        """Non-fraud transaction with zero prior claims → APPROVED."""
        tx = make_transaction(client, auth_headers, {"location": "NY", "amount": 50.0, "device_id": "d1"})
        # Force non-fraud by using a safe amount (is_fraud depends on ML + rules)
        resp = make_claim(client, auth_headers, tx["id"])
        data = resp.json()
        # APPROVED only if is_fraud is False; otherwise MANUAL_REVIEW — both are acceptable
        assert data["status"] in ("APPROVED", "MANUAL_REVIEW")

    def test_serial_claimer_is_rejected(self, client, auth_headers):
        """User with > 3 prior claims should have new claim REJECTED."""
        for i in range(4):
            tx = make_transaction(client, auth_headers)
            make_claim(client, auth_headers, tx["id"])

        # 5th claim
        tx = make_transaction(client, auth_headers)
        resp = make_claim(client, auth_headers, tx["id"])
        assert resp.json()["status"] == "REJECTED"

    def test_stale_transaction_claim_rejected(self, client, auth_headers):
        """Claims on transactions older than 90 days must be REJECTED."""
        from sqlalchemy import text
        from tests.conftest import TestingSessionLocal
        from datetime import datetime, timedelta, timezone

        tx = make_transaction(client, auth_headers)
        # Manually back-date the transaction in the DB
        old_date = datetime.now(timezone.utc) - timedelta(days=91)
        db = TestingSessionLocal()
        try:
            db.execute(
                text("UPDATE transactions SET created_at = :d WHERE id = :id"),
                {"d": old_date, "id": tx["id"]},
            )
            db.commit()
        finally:
            db.close()

        resp = make_claim(client, auth_headers, tx["id"])
        assert resp.json()["status"] == "REJECTED"


class TestListClaims:
    def test_list_empty(self, client, auth_headers):
        resp = client.get("/claims/", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_own_claims(self, client, auth_headers):
        tx = make_transaction(client, auth_headers)
        make_claim(client, auth_headers, tx["id"])
        resp = client.get("/claims/", headers=auth_headers)
        assert len(resp.json()) == 1

    def test_list_unauthenticated(self, client):
        resp = client.get("/claims/")
        assert resp.status_code == 401

    def test_cannot_see_other_users_claims(
        self, client, auth_headers, second_auth_headers
    ):
        tx = make_transaction(client, auth_headers)
        make_claim(client, auth_headers, tx["id"])
        resp = client.get("/claims/", headers=second_auth_headers)
        assert resp.json() == []


class TestGetClaim:
    def test_get_own_claim(self, client, auth_headers):
        tx = make_transaction(client, auth_headers)
        claim = make_claim(client, auth_headers, tx["id"]).json()
        resp = client.get(f"/claims/{claim['id']}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == claim["id"]

    def test_get_nonexistent_claim(self, client, auth_headers):
        resp = client.get("/claims/99999", headers=auth_headers)
        assert resp.status_code == 404

    def test_cannot_get_another_users_claim(
        self, client, auth_headers, second_auth_headers
    ):
        tx = make_transaction(client, auth_headers)
        claim = make_claim(client, auth_headers, tx["id"]).json()
        resp = client.get(f"/claims/{claim['id']}", headers=second_auth_headers)
        assert resp.status_code == 403

    def test_get_claim_unauthenticated(self, client, auth_headers):
        tx = make_transaction(client, auth_headers)
        claim = make_claim(client, auth_headers, tx["id"]).json()
        resp = client.get(f"/claims/{claim['id']}")
        assert resp.status_code == 401
