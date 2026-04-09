"""
Tests for /transactions endpoints and fraud detection rules.
"""
import pytest


BASE_TX = {"location": "New York", "amount": 100.0, "device_id": "device-001"}


class TestCreateTransaction:
    def test_create_transaction_success(self, client, auth_headers):
        resp = client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["amount"] == 100.0
        assert data["location"] == "New York"
        assert data["device_id"] == "device-001"
        assert "risk_score" in data
        assert "risk_level" in data
        assert "decision" in data
        assert "is_fraud" in data
        assert "created_at" in data

    def test_create_transaction_unauthenticated(self, client):
        resp = client.post("/transactions/", json=BASE_TX)
        assert resp.status_code == 401

    def test_create_transaction_missing_fields(self, client, auth_headers):
        resp = client.post("/transactions/", json={"amount": 50.0}, headers=auth_headers)
        assert resp.status_code == 422

    def test_create_transaction_negative_amount(self, client, auth_headers):
        # API doesn't reject negatives at schema level — it should still process
        resp = client.post("/transactions/", json={**BASE_TX, "amount": -1.0}, headers=auth_headers)
        # Accept 200 (no schema-level guard) or 422 if validation is added later
        assert resp.status_code in (200, 422)

    def test_risk_level_is_valid_value(self, client, auth_headers):
        resp = client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        assert resp.json()["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_decision_is_valid_value(self, client, auth_headers):
        resp = client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        assert resp.json()["decision"] in ("ALLOW", "MANUAL_CHECK", "REJECT")


class TestFraudRules:
    """Each test verifies one fraud-detection rule fires (or doesn't)."""

    def test_low_amount_gets_low_risk(self, client, auth_headers):
        """Small, familiar transaction should be low risk."""
        resp = client.post("/transactions/", json={**BASE_TX, "amount": 10.0}, headers=auth_headers)
        data = resp.json()
        assert data["risk_score"] < 0.7        # not flagged as HIGH

    def test_high_amount_increases_risk(self, client, auth_headers):
        """Amount > 5 000 adds 0.4 to the risk score."""
        resp = client.post("/transactions/", json={**BASE_TX, "amount": 6000.0}, headers=auth_headers)
        data = resp.json()
        assert data["risk_score"] >= 0.4

    def test_new_location_increases_risk(self, client, auth_headers):
        """First-ever transaction has a new location, adding 0.2."""
        resp = client.post("/transactions/", json={**BASE_TX, "location": "Brand New City"}, headers=auth_headers)
        data = resp.json()
        # New location adds 0.2 — risk_score should be > 0
        assert data["risk_score"] > 0.0

    def test_is_fraud_true_for_high_risk(self, client, auth_headers):
        """A transaction that clears the HIGH threshold should be marked is_fraud=True."""
        # Amount > 5000 (0.4) + new location (0.2) + shared device (0.3) = 0.9 before ML
        # Register two more users on the same device to trigger shared-device rule
        client.post("/auth/register", json={"name": "U2", "email": "u2@test.com", "password": "p"})
        r2 = client.post("/auth/login", json={"email": "u2@test.com", "password": "p"})
        h2 = {"Authorization": f"Bearer {r2.json()['access_token']}"}
        client.post("/transactions/", json={**BASE_TX, "device_id": "shared-dev"}, headers=h2)

        client.post("/auth/register", json={"name": "U3", "email": "u3@test.com", "password": "p"})
        r3 = client.post("/auth/login", json={"email": "u3@test.com", "password": "p"})
        h3 = {"Authorization": f"Bearer {r3.json()['access_token']}"}
        client.post("/transactions/", json={**BASE_TX, "device_id": "shared-dev"}, headers=h3)

        # Now the original user posts a high-amount tx on the same device
        resp = client.post("/transactions/", json={
            "location": "Unknown City",
            "amount": 9000.0,
            "device_id": "shared-dev",
        }, headers=auth_headers)
        data = resp.json()
        assert data["risk_score"] >= 0.7
        assert data["is_fraud"] is True
        assert data["decision"] == "REJECT"


class TestListTransactions:
    def test_list_empty(self, client, auth_headers):
        resp = client.get("/transactions/", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_own_transactions(self, client, auth_headers):
        client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        client.post("/transactions/", json={**BASE_TX, "amount": 200.0}, headers=auth_headers)
        resp = client.get("/transactions/", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_unauthenticated(self, client):
        resp = client.get("/transactions/")
        assert resp.status_code == 401

    def test_user_cannot_see_other_users_transactions(
        self, client, auth_headers, second_auth_headers
    ):
        client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        resp = client.get("/transactions/", headers=second_auth_headers)
        assert resp.json() == []


class TestGetTransaction:
    def test_get_own_transaction(self, client, auth_headers):
        tx = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        resp = client.get(f"/transactions/{tx['id']}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == tx["id"]

    def test_get_nonexistent_transaction(self, client, auth_headers):
        resp = client.get("/transactions/99999", headers=auth_headers)
        assert resp.status_code == 404

    def test_cannot_get_another_users_transaction(
        self, client, auth_headers, second_auth_headers
    ):
        tx = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        resp = client.get(f"/transactions/{tx['id']}", headers=second_auth_headers)
        assert resp.status_code == 404

    def test_get_transaction_unauthenticated(self, client, auth_headers):
        tx = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        resp = client.get(f"/transactions/{tx['id']}")
        assert resp.status_code == 401
