"""
Unit tests for app/services/fraud_services.py and app/auth.py (pure functions).
No HTTP layer — tests the business logic directly.
"""
import pytest
from unittest.mock import MagicMock, patch
from app.services.fraud_services import get_risk_level, get_decision, detect_device_fraud, verify_claim
from app.auth import hash_password, verify_password, create_access_token, decode_access_token


# ── get_risk_level ────────────────────────────────────────────────────────────

class TestGetRiskLevel:
    def test_below_0_3_is_low(self):
        assert get_risk_level(0.0) == "LOW"
        assert get_risk_level(0.29) == "LOW"

    def test_boundary_0_3_is_medium(self):
        assert get_risk_level(0.3) == "MEDIUM"

    def test_mid_range_is_medium(self):
        assert get_risk_level(0.5) == "MEDIUM"
        assert get_risk_level(0.69) == "MEDIUM"

    def test_boundary_0_7_is_high(self):
        assert get_risk_level(0.7) == "HIGH"

    def test_above_0_7_is_high(self):
        assert get_risk_level(0.9) == "HIGH"
        assert get_risk_level(1.0) == "HIGH"


# ── get_decision ─────────────────────────────────────────────────────────────

class TestGetDecision:
    def test_low_allows(self):
        assert get_decision("LOW") == "ALLOW"

    def test_medium_manual_check(self):
        assert get_decision("MEDIUM") == "MANUAL_CHECK"

    def test_high_rejects(self):
        assert get_decision("HIGH") == "REJECT"


# ── detect_device_fraud ───────────────────────────────────────────────────────

class TestDetectDeviceFraud:
    def _make_db(self, distinct_user_ids):
        """Return a mock DB that simulates distinct() returning user_id tuples."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.distinct.return_value = mock_query
        mock_query.all.return_value = [(uid,) for uid in distinct_user_ids]
        db = MagicMock()
        db.query.return_value = mock_query
        return db

    def test_fewer_than_3_users_not_flagged(self):
        db = self._make_db([1, 2])
        assert detect_device_fraud(db, "dev-x") is False

    def test_exactly_3_users_flagged(self):
        db = self._make_db([1, 2, 3])
        assert detect_device_fraud(db, "dev-x") is True

    def test_more_than_3_users_flagged(self):
        db = self._make_db([1, 2, 3, 4])
        assert detect_device_fraud(db, "dev-x") is True

    def test_single_user_not_flagged(self):
        db = self._make_db([1])
        assert detect_device_fraud(db, "dev-x") is False


# ── verify_claim ──────────────────────────────────────────────────────────────

class TestVerifyClaim:
    from datetime import datetime, timezone

    def _make_db(self, prior_claims_count):
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = prior_claims_count
        db = MagicMock()
        db.query.return_value = mock_query
        return db

    def _fresh_transaction(self, is_fraud=False):
        from datetime import datetime, timezone
        tx = MagicMock()
        tx.user_id = 1
        tx.is_fraud = is_fraud
        tx.created_at = datetime.now(timezone.utc)
        return tx

    def _old_transaction(self):
        from datetime import datetime, timedelta, timezone
        tx = MagicMock()
        tx.user_id = 1
        tx.is_fraud = False
        tx.created_at = datetime.now(timezone.utc) - timedelta(days=100)
        return tx

    def test_first_clean_claim_approved(self):
        db = self._make_db(0)
        tx = self._fresh_transaction(is_fraud=False)
        result = verify_claim(db, MagicMock(), tx)
        assert result == "APPROVED"

    def test_serial_claimer_rejected(self):
        db = self._make_db(4)           # more than 3
        tx = self._fresh_transaction()
        result = verify_claim(db, MagicMock(), tx)
        assert result == "REJECTED"

    def test_stale_claim_rejected(self):
        db = self._make_db(0)
        tx = self._old_transaction()    # > 90 days old
        result = verify_claim(db, MagicMock(), tx)
        assert result == "REJECTED"

    def test_fraud_transaction_manual_review(self):
        db = self._make_db(0)
        tx = self._fresh_transaction(is_fraud=True)
        result = verify_claim(db, MagicMock(), tx)
        assert result == "MANUAL_REVIEW"

    def test_repeat_claimer_manual_review(self):
        db = self._make_db(2)           # 2 prior claims, ≤ 3
        tx = self._fresh_transaction(is_fraud=False)
        result = verify_claim(db, MagicMock(), tx)
        assert result == "MANUAL_REVIEW"


# ── JWT / password helpers ────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_is_not_plain(self):
        h = hash_password("mypassword")
        assert h != "mypassword"

    def test_verify_correct_password(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("mypassword")
        assert verify_password("wrong", h) is False

    def test_two_hashes_of_same_password_differ(self):
        # bcrypt uses random salt
        h1 = hash_password("mypassword")
        h2 = hash_password("mypassword")
        assert h1 != h2


class TestJWT:
    def test_create_and_decode(self):
        token = create_access_token(42)
        assert decode_access_token(token) == 42

    def test_invalid_token_returns_none(self):
        assert decode_access_token("not.a.valid.token") is None

    def test_tampered_token_returns_none(self):
        token = create_access_token(1)
        tampered = token[:-5] + "XXXXX"
        assert decode_access_token(tampered) is None
