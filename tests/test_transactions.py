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
        assert "predicted_fraud" in data
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
        assert resp.json()["decision"] in ("ALLOW", "STEP_UP", "REVIEW", "REJECT")


class TestFraudRules:
    """Each test verifies one fraud-detection rule fires (or doesn't)."""

    def test_low_amount_gets_low_risk(self, client, auth_headers):
        """Small, familiar transaction should be low risk."""
        resp = client.post("/transactions/", json={**BASE_TX, "amount": 10.0}, headers=auth_headers)
        data = resp.json()
        assert data["risk_score"] < 0.7        # not flagged as HIGH

    def test_high_amount_increases_risk(self, client, auth_headers):
        """Amount > 5 000 fires R1_HIGH_AMOUNT and must raise the score.

        T-03 normalises the rule score by the total rule weight, so an absolute
        threshold is asserted against a comparable baseline rather than against
        the old raw 0.4 weight.
        """
        low = client.post("/transactions/", json={**BASE_TX, "amount": 10.0}, headers=auth_headers)
        high = client.post("/transactions/", json={**BASE_TX, "amount": 6000.0}, headers=auth_headers)
        assert high.json()["risk_score"] > low.json()["risk_score"]

    def test_new_location_increases_risk(self, client, auth_headers):
        """A location unseen in the user's history fires R3_NEW_LOCATION.

        Compared against the same transaction from a known location, because
        T-03 normalises rule weights instead of adding a raw 0.2.
        """
        client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        known = client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        new_city = client.post("/transactions/", json={**BASE_TX, "location": "Brand New City"}, headers=auth_headers)
        assert new_city.json()["risk_score"] > known.json()["risk_score"]

    def test_first_transaction_location_not_penalised(self, client, auth_headers):
        """Cold start (A16): a user's first transaction has no history, so its location is not 'new'."""
        resp = client.post("/transactions/", json={**BASE_TX, "location": "Brand New City"}, headers=auth_headers)
        assert resp.json()["risk_score"] < 0.2

    def test_predicted_fraud_true_for_high_risk(self, client, auth_headers):
        """A transaction that clears the HIGH threshold should be marked predicted_fraud=True."""
        # Fires all five rules: R1 (amount > 5000), R2 (90x the user's average),
        # R3 (new location), R4 (shared device) and R5 (five payments in two
        # minutes). The rule half of the score is then exactly 0.7, which clears
        # HIGH whatever the model says. Until T-18 this test fired four rules and
        # relied on the model rating a large amount as fraud; the model trained on
        # realistic data no longer treats amount alone as proof, and this test is
        # about the policy, not the model.
        # The user needs prior history: a first-ever transaction gets no deviation or
        # new-location penalty (cold start, A16).
        for _ in range(5):
            client.post("/transactions/", json=BASE_TX, headers=auth_headers)
        # Register three more users on the same device to trigger the shared-device rule
        client.post("/auth/register", json={"name": "U2", "email": "u2@test.com", "password": "p"})
        r2 = client.post("/auth/login", json={"email": "u2@test.com", "password": "p"})
        h2 = {"Authorization": f"Bearer {r2.json()['access_token']}"}
        client.post("/transactions/", json={**BASE_TX, "device_id": "shared-dev"}, headers=h2)

        client.post("/auth/register", json={"name": "U3", "email": "u3@test.com", "password": "p"})
        r3 = client.post("/auth/login", json={"email": "u3@test.com", "password": "p"})
        h3 = {"Authorization": f"Bearer {r3.json()['access_token']}"}
        client.post("/transactions/", json={**BASE_TX, "device_id": "shared-dev"}, headers=h3)

        client.post("/auth/register", json={"name": "U4", "email": "u4@test.com", "password": "p"})
        r4 = client.post("/auth/login", json={"email": "u4@test.com", "password": "p"})
        h4 = {"Authorization": f"Bearer {r4.json()['access_token']}"}
        client.post("/transactions/", json={**BASE_TX, "device_id": "shared-dev"}, headers=h4)

        # Now the original user posts a high-amount tx on the same device
        resp = client.post("/transactions/", json={
            "location": "Unknown City",
            "amount": 9000.0,
            "device_id": "shared-dev",
        }, headers=auth_headers)
        data = resp.json()
        assert data["risk_score"] >= 0.7
        assert data["predicted_fraud"] is True
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
