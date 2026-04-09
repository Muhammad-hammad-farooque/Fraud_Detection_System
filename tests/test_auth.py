"""
Tests for /auth/register, /auth/login, /auth/me
"""
import pytest


class TestRegister:
    def test_register_success(self, client):
        resp = client.post("/auth/register", json={
            "name": "Alice",
            "email": "alice@example.com",
            "password": "secret123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "alice@example.com"
        assert data["name"] == "Alice"
        assert "id" in data
        assert "password" not in data          # password must never be returned
        assert "hashed_password" not in data

    def test_register_duplicate_email(self, client, registered_user):
        resp = client.post("/auth/register", json={
            "name": "Duplicate",
            "email": registered_user["email"],
            "password": "doesntmatter",
        })
        assert resp.status_code == 400
        assert "already registered" in resp.json()["detail"].lower()

    def test_register_invalid_email(self, client):
        resp = client.post("/auth/register", json={
            "name": "Bad",
            "email": "not-an-email",
            "password": "secret",
        })
        assert resp.status_code == 422

    def test_register_missing_fields(self, client):
        resp = client.post("/auth/register", json={"name": "Incomplete"})
        assert resp.status_code == 422


class TestLogin:
    def test_login_success(self, client, registered_user):
        resp = client.post("/auth/login", json={
            "email": registered_user["email"],
            "password": registered_user["password"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, registered_user):
        resp = client.post("/auth/login", json={
            "email": registered_user["email"],
            "password": "wrong-password",
        })
        assert resp.status_code == 401

    def test_login_unknown_email(self, client):
        resp = client.post("/auth/login", json={
            "email": "nobody@example.com",
            "password": "whatever",
        })
        assert resp.status_code == 401

    def test_login_missing_fields(self, client):
        resp = client.post("/auth/login", json={"email": "test@example.com"})
        assert resp.status_code == 422


class TestGetMe:
    def test_get_me_authenticated(self, client, registered_user, auth_headers):
        resp = client.get("/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == registered_user["email"]
        assert data["name"] == registered_user["name"]

    def test_get_me_no_token(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_get_me_invalid_token(self, client):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert resp.status_code == 401

    def test_get_me_malformed_header(self, client):
        resp = client.get("/auth/me", headers={"Authorization": "NotBearer sometoken"})
        assert resp.status_code == 401
