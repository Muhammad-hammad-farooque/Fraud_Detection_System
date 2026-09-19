"""
Retraining Pipeline
-------------------
Loads labeled transactions from the database, recomputes features,
trains a new RandomForest model, evaluates it against the current
deployed model, and registers and activates the new model only if it wins.

Usage:
    python -m scripts.retrain
"""

import os
import sys
from dataclasses import asdict
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)

# Make sure the project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app import models
from app.features import FEATURE_ORDER, TransactionInput
from app.fraud_detection import features_at
from app.ML.registry import RegistryError, activate, load_model, save_model

LOG_PATH   = os.path.join(os.path.dirname(__file__), "..", "logs", "retrain.log")


def log(msg: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


MIN_LABELLED_SAMPLES = 20
LABEL_MATURITY_DAYS = 90
TEST_FRACTION = 0.2


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
            "merchant_id":       tx.merchant_id,
            "merchant_category": tx.merchant_category,
            "channel":           tx.channel,
            "latitude":          tx.latitude,
            "longitude":         tx.longitude,
            "created_at": tx.created_at,
            "source":     outcome.source,
            "fraud":      int(bool(outcome.is_fraud_confirmed)),
        }
        for tx, outcome in rows
    ])


def build_features(db, df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute each labelled transaction's features exactly as serving computed
    them, by calling the same repository queries "as of" the transaction's own
    timestamp.

    History comes from every transaction in the database, not just the
    labelled ones, and nothing at or after the row's timestamp is visible -
    no feature can see the future. Before T-16 training rebuilt history from
    labelled rows only and counted device sharing across the whole dataset,
    which served different features than it trained on.
    """
    df = df.sort_values("created_at").reset_index(drop=True)

    vectors = []
    for row in df.itertuples(index=False):
        candidate = TransactionInput.from_request(row, at=row.created_at)
        vectors.append(asdict(features_at(db, candidate, row.user_id)))

    features = pd.DataFrame(vectors, columns=FEATURE_ORDER, index=df.index)
    return pd.concat([df.drop(columns=["amount"]), features], axis=1)


def apply_label_maturity(df: pd.DataFrame, maturity_days: int = LABEL_MATURITY_DAYS) -> pd.DataFrame:
    """Drop transactions newer than maturity_days. Their labels have not ripened -
    a chargeback may still arrive. Including them understates the fraud rate."""
    if df.empty:
        return df
    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=maturity_days)
    created_at = _as_utc_series(df["created_at"])
    return df[created_at <= cutoff].reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by created_at, cut at the (1 - test_fraction) quantile.
    All training rows strictly precede all test rows."""
    df = df.sort_values("created_at").reset_index(drop=True)
    if df.empty:
        return df, df

    created_at = _as_utc_series(df["created_at"])
    cut_index = int(len(df) * (1 - test_fraction))
    cut_index = min(max(cut_index, 1), len(df) - 1) if len(df) > 1 else len(df)
    # Cut on the timestamp, not the row index: rows sharing the boundary
    # timestamp all belong to the test side, so the two sets never overlap in time.
    cutoff = created_at.iloc[cut_index]
    train = df[created_at < cutoff].reset_index(drop=True)
    test = df[created_at >= cutoff].reset_index(drop=True)
    return train, test


def _as_utc_series(created_at: pd.Series) -> pd.Series:
    """Timestamps arrive naive from SQLite and aware from Postgres."""
    series = pd.to_datetime(created_at, utc=True)
    return series


def _date_range(df: pd.DataFrame) -> str:
    created_at = _as_utc_series(df["created_at"])
    return f"{created_at.min():%Y-%m-%d %H:%M} .. {created_at.max():%Y-%m-%d %H:%M} UTC"


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

        # ── 2. Drop labels that have not ripened yet ──────────────
        before = len(df)
        df = apply_label_maturity(df)
        log(f"Label maturity: dropped {before - len(df)} transaction(s) newer than "
            f"{LABEL_MATURITY_DAYS} days; {len(df)} remain")

        if len(df) < MIN_LABELLED_SAMPLES or df["fraud"].nunique() < 2:
            log("Not enough mature labelled data after the maturity cutoff. Exiting.")
            return

        # ── 3. Build features ─────────────────────────────────────
        df = build_features(db, df)

        # ── 4. Chronological split ────────────────────────────────
        # A random split on time-ordered data trains on the future to predict
        # the past, which inflates every metric. The cut is on time instead.
        train_df, test_df = chronological_split(df)
        if train_df.empty or test_df.empty:
            log("Chronological split left one side empty. Exiting.")
            return

        # DataFrames, not arrays: the model keeps its feature names, which the
        # registry checks against FEATURE_ORDER before it will store or load it.
        X_train, y_train = train_df[FEATURE_ORDER], train_df["fraud"].values
        X_test, y_test = test_df[FEATURE_ORDER], test_df["fraud"].values
        log(f"Train size: {len(X_train)}  ({_date_range(train_df)})")
        log(f"Test  size: {len(X_test)}  ({_date_range(test_df)})")

        if len(set(y_test)) < 2:
            log("Test window contains a single class — AUC is undefined. Exiting.")
            return

        # ── 5. Train new model ────────────────────────────────────
        new_model = RandomForestClassifier(n_estimators=100, random_state=42)
        new_model.fit(X_train, y_train)
        new_auc = evaluate(new_model, X_test, y_test, "New Model")

        # ── 6. Evaluate the active model on the same test window ──
        try:
            old_model, old_manifest = load_model()
            log(f"Active model: {old_manifest.version}")
            # The champion may use a subset of the features; give it its own.
            old_auc = evaluate(old_model, X_test[old_manifest.feature_order], y_test, "Current Model")
        except RegistryError as exc:
            log(f"No loadable active model ({exc}). The challenger wins by default.")
            old_auc = 0.0

        # ── 7. Champion / challenger decision ─────────────────────
        if new_auc > old_auc:
            manifest = save_model(
                new_model,
                algorithm="RandomForestClassifier",
                training_rows=len(train_df),
                metrics={"auc": new_auc, "test_rows": len(test_df), "champion_auc": old_auc},
            )
            activate(manifest.version)
            log(f"New model {manifest.version} registered and activated. "
                f"AUC {new_auc:.4f} > {old_auc:.4f}. Restart the API to serve it.")
        else:
            log(f"Current model retained. New AUC {new_auc:.4f} <= current {old_auc:.4f}")

    finally:
        db.close()
        log("Retraining pipeline complete")
        log("=" * 60)


if __name__ == "__main__":
    run()
