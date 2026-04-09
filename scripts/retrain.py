"""
Retraining Pipeline
-------------------
Loads labeled transactions from the database, recomputes features,
trains a new RandomForest model, evaluates it against the current
deployed model, and replaces model.pkl only if the new model wins.

Usage:
    python -m scripts.retrain
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)

# Make sure the project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app import models

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "ML", "model.pkl")
LOG_PATH   = os.path.join(os.path.dirname(__file__), "..", "logs", "retrain.log")


def log(msg: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_labeled_data(db) -> pd.DataFrame:
    """
    Build a labeled dataset from the database.

    Label logic (ground truth):
      - APPROVED claim on a transaction  →  confirmed fraud (1)
      - No claim + model said NOT fraud  →  confirmed legitimate (0)
      - Everything else (MANUAL_REVIEW, REJECTED claim) → skipped (ambiguous)
    """
    transactions = db.query(models.Transaction).all()
    claims       = db.query(models.Claim).all()

    # Map transaction_id → claim status
    claim_map = {c.transaction_id: c.status for c in claims}

    rows = []
    for tx in transactions:
        status = claim_map.get(tx.id)

        if status == "APPROVED":
            confirmed_fraud = 1
        elif status is None and tx.is_fraud is False:
            confirmed_fraud = 0
        else:
            continue  # skip ambiguous labels

        rows.append({
            "id":         tx.id,
            "user_id":    tx.user_id,
            "amount":     tx.amount,
            "location":   tx.location,
            "device_id":  tx.device_id,
            "created_at": tx.created_at,
            "fraud":      confirmed_fraud,
        })

    return pd.DataFrame(rows)


def build_features(df: pd.DataFrame, device_user_counts: dict) -> pd.DataFrame:
    """
    Recompute the same 5 features used during training, preserving
    the temporal order so we never use future information.
    """
    df = df.sort_values("created_at").reset_index(drop=True)

    amount_deviation  = []
    is_new_location   = []
    is_flagged_device = []
    velocity          = []

    for i, row in df.iterrows():
        history = df[(df["user_id"] == row["user_id"]) & (df.index < i)]

        # Feature 1: amount deviation from user's own average
        if len(history) > 0 and history["amount"].mean() > 0:
            dev = row["amount"] / history["amount"].mean()
        else:
            dev = 1.0
        amount_deviation.append(dev)

        # Feature 2: new location for this user
        seen_locations = set(history["location"].tolist())
        is_new_location.append(0 if row["location"] in seen_locations else 1)

        # Feature 3: device shared by 3+ distinct users (full dataset view)
        is_flagged_device.append(1 if device_user_counts.get(row["device_id"], 0) >= 3 else 0)

        # Feature 4: transaction velocity (same user, last 120 seconds)
        tx_time = row["created_at"]
        if tx_time.tzinfo is None:
            tx_time = tx_time.replace(tzinfo=timezone.utc)
        recent = history[
            history["created_at"].apply(
                lambda t: (t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t)
            ) >= tx_time - pd.Timedelta(seconds=120)
        ]
        velocity.append(len(recent))

    df["amount_deviation"]  = amount_deviation
    df["is_new_location"]   = is_new_location
    df["is_flagged_device"] = is_flagged_device
    df["velocity"]          = velocity

    return df


def evaluate(model, X_test, y_test, label: str) -> float:
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_prob)
    cm          = confusion_matrix(y_test, y_pred)

    log(f"--- {label} ---")
    log(f"AUC-ROC: {auc:.4f}")
    log(f"Confusion Matrix:\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}")
    log(f"Report:\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")
    return auc


def run():
    log("=" * 60)
    log("Retraining pipeline started")

    db = SessionLocal()
    try:
        # ── 1. Load labeled data ──────────────────────────────────
        df = load_labeled_data(db)
        log(f"Labeled samples: {len(df)}  (fraud={df['fraud'].sum()}, legit={(df['fraud']==0).sum()})")

        if len(df) < 20:
            log("Not enough labeled data to retrain (need at least 20 samples). Exiting.")
            return

        if df["fraud"].nunique() < 2:
            log("Only one class in labeled data — cannot train. Exiting.")
            return

        # ── 2. Build features ─────────────────────────────────────
        device_user_counts = (
            df.groupby("device_id")["user_id"].nunique().to_dict()
        )
        df = build_features(df, device_user_counts)

        FEATURES = ["amount", "amount_deviation", "is_new_location", "is_flagged_device", "velocity"]
        X = df[FEATURES].values
        y = df["fraud"].values

        # ── 3. Train / test split ─────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        log(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

        # ── 4. Train new model ────────────────────────────────────
        new_model = RandomForestClassifier(n_estimators=100, random_state=42)
        new_model.fit(X_train, y_train)
        new_auc = evaluate(new_model, X_test, y_test, "New Model")

        # ── 5. Evaluate current deployed model ────────────────────
        if os.path.exists(MODEL_PATH):
            old_model = joblib.load(MODEL_PATH)
            try:
                old_auc = evaluate(old_model, X_test, y_test, "Current Model")
            except Exception:
                log("Current model could not be evaluated (feature mismatch?). Replacing.")
                old_auc = 0.0
        else:
            log("No existing model.pkl found. Deploying new model directly.")
            old_auc = 0.0

        # ── 6. Champion / challenger decision ─────────────────────
        if new_auc > old_auc:
            joblib.dump(new_model, MODEL_PATH)
            log(f"New model deployed. AUC {new_auc:.4f} > {old_auc:.4f}")
        else:
            log(f"Current model retained. New AUC {new_auc:.4f} <= current {old_auc:.4f}")

    finally:
        db.close()
        log("Retraining pipeline complete")
        log("=" * 60)


if __name__ == "__main__":
    run()
