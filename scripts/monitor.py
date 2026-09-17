"""
Model Performance Monitor
--------------------------
Queries the last N days of transactions, joins with confirmed outcomes
from transaction_outcomes, and reports daily fraud detection metrics.

Logs are written to logs/monitor_YYYY-MM-DD.log

Usage:
    python -m scripts.monitor           # last 1 day (default)
    python -m scripts.monitor --days 7  # last 7 days
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app import models

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

# Below this many confirmed outcomes the detection metrics are noise, so they
# are suppressed rather than reported with false precision.
MIN_LABELLED_FOR_METRICS = 20
ALLOWED_DECISIONS = {"ALLOW"}


def log(msg: str, log_file):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line)
    log_file.write(line + "\n")


def get_ground_truth(db, since: datetime) -> dict:
    """
    Returns a dict:  transaction_id → actual_fraud (bool)

    Ground truth comes only from TransactionOutcome. Claim status is not a fraud
    judgement: claims are rejected for staleness and serial claiming, neither of
    which says anything about whether the transaction was fraudulent (A3).
    """
    outcomes = (
        db.query(models.TransactionOutcome)
        .join(
            models.Transaction,
            models.Transaction.id == models.TransactionOutcome.transaction_id,
        )
        .filter(models.Transaction.created_at >= since)
        .all()
    )
    return {o.transaction_id: bool(o.is_fraud_confirmed) for o in outcomes}


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false positive rate

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "fpr":       fpr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def compute_business_metrics(transactions, truth: dict) -> dict:
    """The operating metrics from the industry reference (§10.5).

    `truth` maps transaction_id to confirmed fraud. Value-based figures count
    only transactions that have a confirmed outcome; volume-based ones count
    everything, since approval rate does not need labels.
    """
    total_count = len(transactions)
    total_value = sum(tx.amount for tx in transactions)
    approved = [tx for tx in transactions if tx.decision in ALLOWED_DECISIONS]

    confirmed_fraud_value = sum(
        tx.amount for tx in transactions if truth.get(tx.id) is True
    )
    labelled_value = sum(tx.amount for tx in transactions if tx.id in truth)

    return {
        "approval_rate": len(approved) / total_count if total_count else 0.0,
        "fraud_bps": (confirmed_fraud_value / labelled_value * 10_000) if labelled_value else None,
        "labelled_count": len(truth),
        "total_count": total_count,
        "total_value": total_value,
        "coverage": len(truth) / total_count if total_count else 0.0,
    }


def false_positive_ratio(tp: int, fp: int) -> str:
    """Good transactions blocked per fraud caught, e.g. '8:1'."""
    if tp == 0:
        return "n/a (no fraud caught in this window)"
    return f"{fp / tp:.1f}:1"


def run(days: int = 1):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"monitor_{date_str}.log")

    os.makedirs(LOG_DIR, exist_ok=True)

    db = SessionLocal()
    try:
        with open(log_path, "a") as log_file:
            log("=" * 60, log_file)
            log(f"Monitor run — last {days} day(s) since {since.strftime('%Y-%m-%d %H:%M UTC')}", log_file)

            # ── Load transactions in window ───────────────────────
            transactions = db.query(models.Transaction).filter(
                models.Transaction.created_at >= since
            ).all()

            if not transactions:
                log("No transactions found in this window.", log_file)
                return

            log(f"Transactions in window: {len(transactions)}", log_file)

            # ── Get ground truth labels from confirmed outcomes ───
            truth = get_ground_truth(db, since)
            labeled = [(tx, truth[tx.id]) for tx in transactions if tx.id in truth]

            business = compute_business_metrics(transactions, truth)
            log(f"Labelled coverage: {len(labeled)} of {len(transactions)} transactions "
                f"have confirmed outcomes ({business['coverage']:.1%})", log_file)

            if len(labeled) < MIN_LABELLED_FOR_METRICS:
                log("", log_file)
                log(f"WARNING: fewer than {MIN_LABELLED_FOR_METRICS} confirmed outcomes — "
                    "detection metrics suppressed, they would be noise at this volume.", log_file)
                log("Ground truth comes from transaction_outcomes (analyst review, chargebacks).", log_file)
            else:
                # ── Confusion matrix ──────────────────────────────
                tp = fp = tn = fn = 0
                for tx, actual_fraud in labeled:
                    predicted_fraud = tx.predicted_fraud
                    if predicted_fraud and actual_fraud:
                        tp += 1
                    elif predicted_fraud and not actual_fraud:
                        fp += 1  # blocked a legit transaction
                    elif not predicted_fraud and actual_fraud:
                        fn += 1  # missed fraud (most dangerous)
                    else:
                        tn += 1

                m = compute_metrics(tp, fp, tn, fn)

                log("", log_file)
                log("── Confusion Matrix ─────────────────────", log_file)
                log(f"  True  Positives (caught fraud):       {m['tp']}", log_file)
                log(f"  False Positives (blocked legit):      {m['fp']}", log_file)
                log(f"  True  Negatives (allowed legit):      {m['tn']}", log_file)
                log(f"  False Negatives (missed fraud):       {m['fn']}", log_file)
                log("", log_file)
                log("── Performance Metrics ──────────────────", log_file)
                log(f"  Precision            : {m['precision']:.2%}  (of flagged, how many were real fraud)", log_file)
                log(f"  Recall               : {m['recall']:.2%}  (of real fraud, how many we caught)", log_file)
                log(f"  F1 Score             : {m['f1']:.2%}", log_file)
                log(f"  False Positive Rate  : {m['fpr']:.2%}  (legit transactions wrongly blocked)", log_file)

                # ── Health warnings ───────────────────────────────
                log("", log_file)
                log("── Health Check ─────────────────────────", log_file)
                if m["recall"] < 0.80:
                    log("  WARNING: Recall below 80% — model is missing too much fraud. Consider retraining.", log_file)
                else:
                    log("  OK: Recall is acceptable.", log_file)

                if m["fpr"] > 0.10:
                    log("  WARNING: False positive rate above 10% — too many legit users being blocked.", log_file)
                else:
                    log("  OK: False positive rate is acceptable.", log_file)

                if m["fn"] > 0:
                    log(f"  ALERT: {m['fn']} fraudulent transaction(s) were missed (false negatives).", log_file)

                log("", log_file)
                log("── Business Metrics ─────────────────────", log_file)
                log(f"  Approval rate        : {business['approval_rate']:.2%}  (share of transactions allowed)", log_file)
                log(f"  False positive ratio : {false_positive_ratio(m['tp'], m['fp'])}  (legit blocked per fraud caught)", log_file)
                if business["fraud_bps"] is not None:
                    log(f"  Fraud loss           : {business['fraud_bps']:.1f} bps of labelled volume", log_file)

            # ── Overall volume stats (no labels needed) ───────────
            log("", log_file)
            log("── Volume Stats ─────────────────────────", log_file)

            log(f"  Approval rate    : {business['approval_rate']:.2%}", log_file)

            by_decision = defaultdict(int)
            by_risk     = defaultdict(int)
            flagged     = 0

            for tx in transactions:
                by_decision[tx.decision] += 1
                by_risk[tx.risk_level]   += 1
                if tx.predicted_fraud:
                    flagged += 1

            log(f"  Flagged as fraud : {flagged} / {len(transactions)}", log_file)
            log(f"  By decision      : {dict(by_decision)}", log_file)
            log(f"  By risk level    : {dict(by_risk)}", log_file)

            # Average risk score
            avg_risk = sum(tx.risk_score for tx in transactions) / len(transactions)
            log(f"  Avg risk score   : {avg_risk:.4f}", log_file)

            log("=" * 60, log_file)

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor fraud model performance")
    parser.add_argument("--days", type=int, default=1, help="Look-back window in days (default: 1)")
    args = parser.parse_args()
    run(days=args.days)
