"""
Payment context on transactions — schema, validation, generated data (T-15).
"""
import hashlib
import ipaddress
import json
import random
from collections import Counter

import pytest

import populate_db
from app import models
from app.routers.transactions import request_fingerprint
from app.schemas import TransactionCreate

BASE_TX = {"location": "Lahore", "amount": 100.0, "device_id": "device-001"}
CONTEXT = {
    "merchant_id": "M5411-007",
    "merchant_category": "5411",
    "currency": "PKR",
    "channel": "pos",
    "ip_address": "72.221.14.9",
    "card_token": "tok_9f2c61a0b3d4e5f6",
    "external_txn_id": "psp_00a1b2c3d4e5f60718293a4b",
    "latitude": 31.5204,
    "longitude": 74.3587,
}


def _post(client, headers, **overrides):
    return client.post("/transactions/", json={**BASE_TX, **overrides}, headers=headers)


class TestAcceptance:
    def test_every_new_field_is_accepted_and_returned(self, client, auth_headers):
        resp = _post(client, auth_headers, **CONTEXT)
        assert resp.status_code == 200
        body = resp.json()
        for field, value in CONTEXT.items():
            assert body[field] == value, field

    def test_every_new_field_is_stored(self, client, auth_headers, db_session):
        txn_id = _post(client, auth_headers, **CONTEXT).json()["id"]
        db_session.expire_all()
        row = db_session.get(models.Transaction, txn_id)
        for field, value in CONTEXT.items():
            assert getattr(row, field) == value, field

    def test_all_new_fields_are_optional(self, client, auth_headers):
        body = _post(client, auth_headers).json()
        assert all(body[field] is None for field in CONTEXT)

    def test_ipv6_is_accepted(self, client, auth_headers):
        body = _post(client, auth_headers, ip_address="2001:db8::1").json()
        assert body["ip_address"] == "2001:db8::1"

    @pytest.mark.parametrize("channel", ["web", "mobile", "pos", "atm"])
    def test_every_channel(self, client, auth_headers, channel):
        assert _post(client, auth_headers, channel=channel).json()["channel"] == channel


INVALID = [
    ("merchant_category", "54a1"),
    ("merchant_category", "541"),
    ("currency", "pkr"),
    ("currency", "PKRS"),
    ("channel", "fax"),
    ("ip_address", "999.1.1.1"),
    ("ip_address", "not-an-ip"),
    ("latitude", 91.0),
    ("longitude", -181.0),
    ("merchant_id", ""),
    ("card_token", "x" * 65),
]


class TestValidation:
    @pytest.mark.parametrize("field, value", INVALID, ids=[f"{f}={v!r}"[:40] for f, v in INVALID])
    def test_malformed_values_are_rejected(self, client, auth_headers, field, value):
        extra = {field: value}
        if field in ("latitude", "longitude"):
            extra = {"latitude": 0.0, "longitude": 0.0, field: value}
        assert _post(client, auth_headers, **extra).status_code == 422

    @pytest.mark.parametrize("coordinates", [{"latitude": 31.5}, {"longitude": 74.3}])
    def test_coordinates_must_come_in_pairs(self, client, auth_headers, coordinates):
        resp = _post(client, auth_headers, **coordinates)
        assert resp.status_code == 422
        assert "together" in json.dumps(resp.json())

    @pytest.mark.parametrize("pan", ["4111111111111111", "4111 1111 1111 1111", "5555-5555-5555-4444",
                                     "378282246310005"])
    def test_a_raw_card_number_is_refused(self, client, auth_headers, db_session, pan):
        """PCI DSS: a primary account number must never be stored."""
        resp = _post(client, auth_headers, card_token=pan)
        assert resp.status_code == 422
        assert "not a card number" in json.dumps(resp.json())
        db_session.expire_all()
        assert db_session.query(models.Transaction).count() == 0

    def test_digits_that_fail_luhn_are_a_legitimate_token(self, client, auth_headers):
        assert _post(client, auth_headers, card_token="4111111111111112").status_code == 200


class TestIdempotencyCompatibility:
    def test_old_shape_requests_keep_their_pre_t15_fingerprint(self):
        """A retry spanning the deploy must still replay, not be refused as a different request."""
        pre_t15 = hashlib.sha256(
            json.dumps(BASE_TX, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert request_fingerprint(TransactionCreate(**BASE_TX)) == pre_t15

    def test_a_retry_with_the_new_fields_replays(self, client, auth_headers):
        headers = {**auth_headers, "Idempotency-Key": "pay-ctx"}
        first = client.post("/transactions/", json={**BASE_TX, **CONTEXT}, headers=headers)
        second = client.post("/transactions/", json={**BASE_TX, **CONTEXT}, headers=headers)
        assert second.headers["Idempotent-Replayed"] == "true"
        assert first.json() == second.json()

    def test_changing_a_new_field_under_the_same_key_is_refused(self, client, auth_headers):
        headers = {**auth_headers, "Idempotency-Key": "pay-ctx"}
        client.post("/transactions/", json={**BASE_TX, **CONTEXT}, headers=headers)
        other = client.post("/transactions/", json={**BASE_TX, **CONTEXT, "merchant_id": "M9999-001"},
                            headers=headers)
        assert other.status_code == 422


# ── Generated data ───────────────────────────────────────────────────────────

def _generate(n=2000, fraud_rate=0.3, seed=7):
    rng = random.Random(seed)
    home = "Lahore"
    profile = populate_db.build_payment_profile(rng, home)
    rows = []
    for _ in range(n):
        is_fraud = rng.random() < fraud_rate
        location = rng.choice(populate_db.LOCATIONS) if is_fraud else home
        rows.append((is_fraud, location, populate_db.payment_context(rng, profile, location, is_fraud)))
    return profile, rows


class TestGeneratedData:
    def test_every_generated_row_passes_the_api_validation(self):
        _, rows = _generate()
        for _, location, context in rows:
            TransactionCreate(**BASE_TX, **context)       # raises on anything invalid

    def test_every_new_field_is_populated(self):
        _, rows = _generate(200)
        for _, _, context in rows:
            assert set(context) == set(CONTEXT)
            assert all(value is not None for value in context.values())

    def test_every_city_has_coordinates_and_a_currency(self):
        assert set(populate_db.CITY_INFO) == set(populate_db.LOCATIONS)

    def test_fraud_concentrates_in_high_risk_categories_and_card_not_present(self):
        _, rows = _generate()
        fraud = [c for is_fraud, _, c in rows if is_fraud]
        legit = [c for is_fraud, _, c in rows if not is_fraud]
        assert all(c["merchant_category"] in populate_db.HIGH_RISK_MCCS for c in fraud)
        assert not any(c["merchant_category"] in populate_db.HIGH_RISK_MCCS for c in legit)
        fraud_cnp = sum(c["channel"] in ("web", "mobile") for c in fraud) / len(fraud)
        legit_cnp = sum(c["channel"] in ("web", "mobile") for c in legit) / len(legit)
        assert fraud_cnp > 0.85 and legit_cnp < 0.55

    def test_legitimate_behaviour_is_consistent_per_customer(self):
        profile, rows = _generate()
        legit = [c for is_fraud, _, c in rows if not is_fraud]
        assert len({c["ip_address"] for c in legit}) == 1               # one home network
        favourites = {m for _, m in profile["favourite_merchants"]}
        share_at_favourites = sum(c["merchant_id"] in favourites for c in legit) / len(legit)
        assert share_at_favourites > 0.8
        assert {c["currency"] for c in legit} == {"PKR"}               # Lahore prices in rupees
        assert {c["card_token"] for c in legit} <= set(profile["cards"])

    def test_fraud_reuses_proxy_infrastructure(self):
        _, rows = _generate()
        prefixes = Counter(".".join(c["ip_address"].split(".")[:2]) for is_fraud, _, c in rows if is_fraud)
        assert set(prefixes) <= set(populate_db.PROXY_PREFIXES)

    def test_coordinates_sit_near_the_city(self):
        _, rows = _generate(300)
        for _, location, context in rows:
            lat, lon, _ = populate_db.CITY_INFO[location]
            assert abs(context["latitude"] - lat) <= 0.08 + 1e-9
            assert abs(context["longitude"] - lon) <= 0.08 + 1e-9

    def test_generated_ips_are_public_looking_addresses(self):
        _, rows = _generate(300)
        for _, _, context in rows:
            assert ipaddress.ip_address(context["ip_address"]).version == 4


class TestPopulateScript:
    def test_a_small_seed_run_fills_every_new_column(self, db_session, monkeypatch):
        from tests.conftest import TestingSessionLocal

        monkeypatch.setattr(populate_db, "SessionLocal", TestingSessionLocal)
        monkeypatch.setattr(populate_db, "TOTAL_USERS", 5)
        monkeypatch.setattr(populate_db, "TOTAL_TRANSACTIONS", 80)
        populate_db.populate()

        db_session.expire_all()
        rows = db_session.query(models.Transaction).all()
        assert len(rows) == 80
        for row in rows:
            for field in CONTEXT:
                assert getattr(row, field) is not None, field

    def test_refuses_to_reseed_over_an_audit_trail(self, client, auth_headers, db_session, monkeypatch):
        from tests.conftest import TestingSessionLocal

        _post(client, auth_headers)                      # writes a decision_audits row
        monkeypatch.setattr(populate_db, "SessionLocal", TestingSessionLocal)
        with pytest.raises(SystemExit, match="append-only"):
            populate_db.populate()
        db_session.expire_all()
        assert db_session.query(models.Transaction).count() == 1
