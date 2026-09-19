"""
Role-based access control — every role against every endpoint (T-10).
"""
import pytest

from app import models
from app.models import Role
from scripts.set_role import set_role

CUSTOMER, ANALYST, ADMIN = Role.CUSTOMER, Role.ANALYST, Role.ADMIN
BASE_TX = {"location": "Lahore", "amount": 100.0, "device_id": "device-001"}


def _account(client, db, email, role):
    """Register, promote directly in the database, and log in."""
    resp = client.post("/auth/register", json={"name": email, "email": email, "password": "pw"})
    assert resp.status_code == 201
    if role != CUSTOMER:
        set_role(db, email, role.value)
    token = client.post("/auth/login", json={"email": email, "password": "pw"}).json()["access_token"]
    return {"id": resp.json()["id"], "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture
def world(client, db_session):
    """One account per role, plus a customer transaction and claim to aim at."""
    accounts = {
        CUSTOMER: _account(client, db_session, "customer@bank.com", CUSTOMER),
        ANALYST: _account(client, db_session, "analyst@bank.com", ANALYST),
        ADMIN: _account(client, db_session, "admin@bank.com", ADMIN),
    }
    customer = accounts[CUSTOMER]["headers"]
    txn = client.post("/transactions/", json=BASE_TX, headers=customer).json()
    claim = client.post("/claims/", json={
        "transaction_id": txn["id"], "reason": "not me", "amount": 50.0,
    }, headers=customer).json()
    return {"accounts": accounts, "txn_id": txn["id"], "claim_id": claim["id"]}


def _call(client, world, role, method, path_template, body_factory=None):
    account = world["accounts"][role]
    path = path_template.format(
        txn_id=world["txn_id"],
        claim_id=world["claim_id"],
        own_id=account["id"],
        customer_id=world["accounts"][CUSTOMER]["id"],
    )
    body = body_factory(world) if body_factory else None
    return client.request(method, path, json=body, headers=account["headers"])


# (method, path, body, {role: expected status})
MATRIX = [
    ("GET",   "/auth/me",                     None, {CUSTOMER: 200, ANALYST: 200, ADMIN: 200}),
    ("GET",   "/users/me",                    None, {CUSTOMER: 200, ANALYST: 200, ADMIN: 200}),
    ("GET",   "/users/{own_id}",              None, {CUSTOMER: 200, ANALYST: 200, ADMIN: 200}),
    ("POST",  "/transactions/",               lambda w: BASE_TX,
                                                    {CUSTOMER: 200, ANALYST: 403, ADMIN: 403}),
    ("GET",   "/transactions/",               None, {CUSTOMER: 200, ANALYST: 403, ADMIN: 403}),
    ("GET",   "/transactions/{txn_id}",       None, {CUSTOMER: 200, ANALYST: 403, ADMIN: 403}),
    ("POST",  "/claims/",                     lambda w: {"transaction_id": w["txn_id"], "reason": "x", "amount": 10.0},
                                                    {CUSTOMER: 201, ANALYST: 403, ADMIN: 403}),
    ("GET",   "/claims/",                     None, {CUSTOMER: 200, ANALYST: 403, ADMIN: 403}),
    ("GET",   "/claims/{claim_id}",           None, {CUSTOMER: 200, ANALYST: 403, ADMIN: 403}),
    ("GET",   "/analyst/transactions/{txn_id}", None, {CUSTOMER: 403, ANALYST: 200, ADMIN: 200}),
    ("GET",   "/analyst/cases",               None, {CUSTOMER: 403, ANALYST: 200, ADMIN: 200}),
    ("GET",   "/analyst/transactions/{txn_id}/audit", None, {CUSTOMER: 403, ANALYST: 200, ADMIN: 200}),
    ("GET",   "/admin/users",                 None, {CUSTOMER: 403, ANALYST: 403, ADMIN: 200}),
    ("PATCH", "/admin/users/{customer_id}/role", lambda w: {"role": "CUSTOMER"},
                                                    {CUSTOMER: 403, ANALYST: 403, ADMIN: 200}),
]

CASES = [
    pytest.param(method, path, body, role, expected, id=f"{role.value}-{method}-{path}")
    for method, path, body, expectations in MATRIX
    for role, expected in expectations.items()
]


@pytest.mark.parametrize("method, path, body, role, expected", CASES)
def test_role_matrix(client, world, method, path, body, role, expected):
    assert _call(client, world, role, method, path, body).status_code == expected


@pytest.mark.parametrize("method, path, body, _", [
    pytest.param(m, p, b, e, id=f"{m}-{p}") for m, p, b, e in MATRIX
])
def test_every_protected_endpoint_rejects_anonymous_callers(client, world, method, path, body, _):
    resolved = path.format(txn_id=world["txn_id"], claim_id=world["claim_id"],
                           own_id=1, customer_id=1)
    resp = client.request(method, resolved, json=body(world) if body else None)
    assert resp.status_code == 401


class TestRoleAssignment:
    def test_new_accounts_are_customers(self, client):
        resp = client.post("/auth/register", json={"name": "N", "email": "n@bank.com", "password": "pw"})
        assert resp.json()["role"] == "CUSTOMER"

    def test_role_cannot_be_chosen_at_registration(self, client):
        resp = client.post("/auth/register", json={
            "name": "Sneaky", "email": "sneaky@bank.com", "password": "pw", "role": "ADMIN",
        })
        assert resp.status_code == 201
        assert resp.json()["role"] == "CUSTOMER"

    def test_admin_can_promote_a_customer(self, client, world):
        customer_id = world["accounts"][CUSTOMER]["id"]
        resp = client.patch(f"/admin/users/{customer_id}/role", json={"role": "ANALYST"},
                            headers=world["accounts"][ADMIN]["headers"])
        assert resp.status_code == 200
        assert resp.json()["role"] == "ANALYST"

    def test_role_change_takes_effect_on_the_existing_token(self, client, world):
        """Demotion must not wait for the token to expire."""
        analyst = world["accounts"][ANALYST]
        path = f"/analyst/transactions/{world['txn_id']}"
        assert client.get(path, headers=analyst["headers"]).status_code == 200

        client.patch(f"/admin/users/{analyst['id']}/role", json={"role": "CUSTOMER"},
                     headers=world["accounts"][ADMIN]["headers"])
        assert client.get(path, headers=analyst["headers"]).status_code == 403

    def test_admin_cannot_demote_themselves(self, client, world):
        admin = world["accounts"][ADMIN]
        resp = client.patch(f"/admin/users/{admin['id']}/role", json={"role": "CUSTOMER"},
                            headers=admin["headers"])
        assert resp.status_code == 400

    def test_unknown_role_is_rejected(self, client, world):
        customer_id = world["accounts"][CUSTOMER]["id"]
        resp = client.patch(f"/admin/users/{customer_id}/role", json={"role": "SUPERUSER"},
                            headers=world["accounts"][ADMIN]["headers"])
        assert resp.status_code == 422

    def test_unknown_user_is_404(self, client, world):
        resp = client.patch("/admin/users/99999/role", json={"role": "ANALYST"},
                            headers=world["accounts"][ADMIN]["headers"])
        assert resp.status_code == 404


class TestScoping:
    def test_analyst_view_reaches_across_customers(self, client, world):
        """The analyst route is the one place cross-customer reads are intended."""
        resp = client.get(f"/analyst/transactions/{world['txn_id']}",
                          headers=world["accounts"][ANALYST]["headers"])
        assert resp.json()["user_id"] == world["accounts"][CUSTOMER]["id"]

    def test_customer_routes_stay_scoped_to_the_caller(self, client, world, db_session):
        """A second customer sees none of the first customer's records."""
        other = _account(client, db_session, "other@bank.com", CUSTOMER)
        assert client.get("/transactions/", headers=other["headers"]).json() == []
        assert client.get(f"/transactions/{world['txn_id']}", headers=other["headers"]).status_code == 404
        assert client.get("/claims/", headers=other["headers"]).json() == []

    def test_analyst_view_404s_on_a_missing_transaction(self, client, world):
        resp = client.get("/analyst/transactions/99999", headers=world["accounts"][ANALYST]["headers"])
        assert resp.status_code == 404


class TestSetRoleScript:
    def test_promotes_by_email(self, client, db_session):
        client.post("/auth/register", json={"name": "B", "email": "boot@bank.com", "password": "pw"})
        user = set_role(db_session, "boot@bank.com", "admin")
        assert user.role == "ADMIN"

    def test_unknown_role_raises(self, client, db_session):
        client.post("/auth/register", json={"name": "B", "email": "boot@bank.com", "password": "pw"})
        with pytest.raises(ValueError, match="Unknown role"):
            set_role(db_session, "boot@bank.com", "ROOT")

    def test_unknown_email_raises(self, db_session):
        with pytest.raises(ValueError, match="No user"):
            set_role(db_session, "nobody@bank.com", "ADMIN")
