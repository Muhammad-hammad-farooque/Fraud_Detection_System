import requests
import os

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Auth ──────────────────────────────────────────────────────────────────────

def register(name: str, email: str, password: str):
    r = requests.post(
        f"{BASE_URL}/auth/register",
        json={"name": name, "email": email, "password": password},
    )
    return r.status_code, r.json()


def login(email: str, password: str):
    r = requests.post(
        f"{BASE_URL}/auth/login",
        json={"email": email, "password": password},
    )
    return r.status_code, r.json()


def get_me(token: str):
    r = requests.get(f"{BASE_URL}/auth/me", headers=_headers(token))
    return r.status_code, r.json()


# ── Transactions ──────────────────────────────────────────────────────────────

def list_transactions(token: str):
    r = requests.get(f"{BASE_URL}/transactions/", headers=_headers(token))
    return r.status_code, r.json()


def create_transaction(token: str, location: str, amount: float, device_id: str):
    r = requests.post(
        f"{BASE_URL}/transactions/",
        json={"location": location, "amount": amount, "device_id": device_id},
        headers=_headers(token),
    )
    return r.status_code, r.json()


# ── Claims ────────────────────────────────────────────────────────────────────

def list_claims(token: str):
    r = requests.get(f"{BASE_URL}/claims/", headers=_headers(token))
    return r.status_code, r.json()


def create_claim(token: str, transaction_id: int, reason: str, amount: float):
    r = requests.post(
        f"{BASE_URL}/claims/",
        json={"transaction_id": transaction_id, "reason": reason, "amount": amount},
        headers=_headers(token),
    )
    return r.status_code, r.json()
