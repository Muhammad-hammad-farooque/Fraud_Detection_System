"""
The Streamlit client's payment call — retries carry one Idempotency-Key (T-14).
"""
import importlib
import pathlib
import sys
import types

import pytest
import requests

from app import models

FRONTEND = pathlib.Path(__file__).resolve().parent.parent / "frontend"


@pytest.fixture
def api_client(monkeypatch):
    """frontend/api.py, imported with Streamlit stubbed out.

    The client only reads st.secrets, and Streamlit lives in the frontend's own
    image, so the API's test environment does not need it installed.
    """
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(secrets={}))
    monkeypatch.syspath_prepend(str(FRONTEND))
    sys.modules.pop("api", None)
    module = importlib.import_module("api")
    yield module
    sys.modules.pop("api", None)


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def _token(auth_headers):
    return auth_headers["Authorization"].split(" ", 1)[1]


class TestRetries:
    def test_every_attempt_carries_the_same_key(self, api_client, monkeypatch):
        seen = []

        def flaky_post(url, json, headers, timeout):
            seen.append(headers["Idempotency-Key"])
            if len(seen) < 3:
                raise requests.ConnectionError("dropped")
            return _Response(200, {"id": 1})

        monkeypatch.setattr(api_client.requests, "post", flaky_post)
        status, body = api_client.create_transaction("tok", "Lahore", 100.0, "d1")
        assert (status, body) == (200, {"id": 1})
        assert len(seen) == 3 and len(set(seen)) == 1

    def test_each_payment_gets_its_own_key(self, api_client, monkeypatch):
        seen = []
        monkeypatch.setattr(api_client.requests, "post",
                            lambda url, json, headers, timeout: seen.append(headers["Idempotency-Key"])
                            or _Response(200, {}))
        api_client.create_transaction("tok", "Lahore", 100.0, "d1")
        api_client.create_transaction("tok", "Lahore", 100.0, "d1")
        assert seen[0] != seen[1]

    def test_giving_up_says_the_outcome_is_unknown(self, api_client, monkeypatch):
        def down(url, json, headers, timeout):
            raise requests.Timeout("no answer")

        monkeypatch.setattr(api_client.requests, "post", down)
        status, body = api_client.create_transaction("tok", "Lahore", 100.0, "d1", retries=1)
        assert status == 0
        assert "may or may not" in body["detail"]


class TestLostResponse:
    def test_a_lost_response_does_not_charge_twice(self, api_client, client, auth_headers, db_session, monkeypatch):
        """A15 end to end: the API commits, the reply is lost, the client retries."""
        attempts = []

        def through_the_real_api(url, json, headers, timeout):
            resp = client.post("/transactions/", json=json, headers=headers)
            attempts.append(resp)
            if len(attempts) == 1:
                raise requests.ConnectionError("response lost after the server committed")
            return resp

        monkeypatch.setattr(api_client.requests, "post", through_the_real_api)
        status, body = api_client.create_transaction(_token(auth_headers), "Lahore", 100.0, "d1")

        assert status == 200
        assert body["id"] == attempts[0].json()["id"]
        assert attempts[1].headers["Idempotent-Replayed"] == "true"
        db_session.expire_all()
        assert db_session.query(models.Transaction).count() == 1
        assert db_session.query(models.DecisionAudit).count() == 1
