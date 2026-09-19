"""
Shared fixtures for all test modules.

Uses a genuinely in-memory SQLite database - StaticPool is required, because
without it every connection gets its own empty database and the tables created
on one are invisible to the next. The suite leaves no test.db file behind.
"""
import copy
import itertools
import os

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set required env vars BEFORE importing anything from app
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.database import Base
from app.main import app
from app.dependencies import get_db
from app import auth

# Real bcrypt, at its minimum cost factor. Production keeps passlib's default
# (12 rounds); at that cost the role matrix alone, which registers three
# accounts per test, turns a one-minute suite into a four-minute one.
auth.pwd_context.update(bcrypt__rounds=4)

SQLITE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,      # required, or each connection gets a fresh empty DB
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def reset_db():
    """Drop and recreate all tables before each test for full isolation."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    """A direct session against the test database, for repository-level tests."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def registered_user(client):
    """Register a user and return their credentials + response data."""
    payload = {"name": "Test User", "email": "test@example.com", "password": "password123"}
    resp = client.post("/auth/register", json=payload)
    assert resp.status_code == 201
    return {**payload, "id": resp.json()["id"]}


@pytest.fixture
def auth_headers(client, registered_user):
    """Login and return Authorization headers for the registered user."""
    resp = client.post("/auth/login", json={
        "email": registered_user["email"],
        "password": registered_user["password"],
    })
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def second_user(client):
    """A second distinct user."""
    payload = {"name": "Other User", "email": "other@example.com", "password": "other123"}
    resp = client.post("/auth/register", json=payload)
    assert resp.status_code == 201
    return {**payload, "id": resp.json()["id"]}


@pytest.fixture
def second_auth_headers(client, second_user):
    resp = client.post("/auth/login", json={
        "email": second_user["email"],
        "password": second_user["password"],
    })
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


_config_counter = itertools.count()


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@pytest.fixture
def rules_config(tmp_path, monkeypatch):
    """Write a rules file derived from the shipped one, and point the app at it.

    Call with nested overrides, e.g. rules_config(policy={"allow_below": 0}).
    Each call writes a fresh file, so the config cache always sees a change.
    """
    from app.config import DEFAULT_PATH

    shipped = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))

    def write(raw: dict | None = None, **overrides) -> "os.PathLike":
        data = raw if raw is not None else _deep_merge(shipped, overrides)
        path = tmp_path / f"rules-{next(_config_counter)}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        monkeypatch.setenv("RULES_CONFIG_PATH", str(path))
        return path

    return write


@pytest.fixture
def everything_reviews(rules_config):
    """Widen the REVIEW band to cover every score, through the rules file."""
    return rules_config(policy={"allow_below": 0, "step_up_below": 0, "review_below": 1.01})
