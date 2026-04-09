"""
Shared fixtures for all test modules.
Uses an in-memory SQLite database so no real PostgreSQL is needed.
"""
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set required env vars BEFORE importing anything from app
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from app.database import Base
from app.main import app
from app.dependencies import get_db

SQLITE_URL = "sqlite:///./test.db"
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
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
