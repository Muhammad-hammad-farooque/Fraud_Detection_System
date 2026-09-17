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
from dataclasses import asdict
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
from app.features import FEATURE_ORDER, aggregates_from_history, compute_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "ML", "model.pkl")
LOG_PATH   = os.path.join(os.path.dirname(__file__), "..", "logs", "retrain.log")


def log(msg: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


MIN_LABELLED_SAMPLES = 20


def load_labeled_data(db) -> pd.DataFrame:
    """
    Build a labeled dataset from confirmed outcomes only.

    Ground truth comes exclusively from TransactionOutcome - an investigator
    determination, a chargeback, a customer report, or an exploration holdout.
    Claim status is not a fraud judgement, and the model's own predicted_fraud
    output must never become its training label (A1, A4).
    """
    rows = (
        db.query(models.Transaction, models.TransactionOutcome)
        .join(
            models.TransactionOutcome,
            models.TransactionOutcome.transaction_id == models.Transaction.id,
        )
        .all()
    )

    return pd.DataFrame([
        {
            "id":         tx.id,
            "user_id":    tx.user_id,
            "amount":     tx.amount,
            "location":   tx.location,
            "device_id":  tx.device_id,
            "created_at": tx.created_at,
            "source":     outcome.source,
            "fraud":      int(bool(outcome.is_fraud_confirmed)),
        }
        for tx, outcome in rows
    ])


def build_features(df: pd.DataFrame, device_user_counts: dict) -> pd.DataFrame:
    """
    Recompute features through app.features, preserving temporal order so
    each row only sees the user's strictly earlier transactions.
    """
    df = df.sort_values("created_at").reset_index(drop=True)

    vectors = []
    for i, row in df.iterrows():
        history = df[(df["user_id"] == row["user_id"]) & (df.index < i)]
        aggregates = aggregates_from_history(history.itertuples(index=False), row["created_at"])
        fv = compute_features(
            amount=row["amount"],
            location=row["location"],
            aggregates=aggregates,
            device_user_count=device_user_counts.get(row["device_id"], 0),
        )
        vectors.append(asdict(fv))

    features = pd.DataFrame(vectors, columns=FEATURE_ORDER, index=df.index)
    return pd.concat([df.drop(columns=["amount"]), features], axis=1)


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
        if df.empty:
            log("No confirmed outcomes in transaction_outcomes — nothing to train on. Exiting.")
            log("Labels come from analyst review, chargebacks or exploration holdouts, never from claims.")
            return

        log(f"Labeled samples: {len(df)}  (fraud={df['fraud'].sum()}, legit={(df['fraud']==0).sum()})")
        log(f"Label sources: {df['source'].value_counts().to_dict()}")

        if len(df) < MIN_LABELLED_SAMPLES:
            log(f"Only {len(df)} confirmed outcomes, need at least {MIN_LABELLED_SAMPLES}. Exiting.")
            return

        if df["fraud"].nunique() < 2:
            log("Only one class in labeled data — cannot train. Exiting.")
            return

        # ── 2. Build features ─────────────────────────────────────
        device_user_counts = (
            df.groupby("device_id")["user_id"].nunique().to_dict()
        )
        df = build_features(df, device_user_counts)

        X = df[FEATURE_ORDER].values
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
