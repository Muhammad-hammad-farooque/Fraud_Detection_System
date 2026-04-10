import requests
import os
import streamlit as st

BASE_URL = (
    st.secrets.get("API_BASE_URL")
    or os.getenv("API_BASE_URL", "http://localhost:8000")
)


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _parse(r: requests.Response):
    """Return (status_code, json_body). If response is not JSON, return a safe error dict."""
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"detail": f"Server returned non-JSON response (HTTP {r.status_code}). Check API logs."}


# ── Auth ──────────────────────────────────────────────────────────────────────

def register(name: str, email: str, password: str):
    r = requests.post(
        f"{BASE_URL}/auth/register",
        json={"name": name, "email": email, "password": password},
    )
    return _parse(r)


def login(email: str, password: str):
    r = requests.post(
        f"{BASE_URL}/auth/login",
        json={"email": email, "password": password},
    )
    return _parse(r)


def get_me(token: str):
    r = requests.get(f"{BASE_URL}/auth/me", headers=_headers(token))
    return _parse(r)


# ── Transactions ──────────────────────────────────────────────────────────────

def list_transactions(token: str):
    r = requests.get(f"{BASE_URL}/transactions/", headers=_headers(token))
    return _parse(r)


def create_transaction(token: str, location: str, amount: float, device_id: str):
    r = requests.post(
        f"{BASE_URL}/transactions/",
        json={"location": location, "amount": amount, "device_id": device_id},
        headers=_headers(token),
    )
    return _parse(r)


# ── Claims ────────────────────────────────────────────────────────────────────

def list_claims(token: str):
    r = requests.get(f"{BASE_URL}/claims/", headers=_headers(token))
    return _parse(r)


def create_claim(token: str, transaction_id: int, reason: str, amount: float):
    r = requests.post(
        f"{BASE_URL}/claims/",
        json={"transaction_id": transaction_id, "reason": reason, "amount": amount},
        headers=_headers(token),
    )
    return _parse(r)
