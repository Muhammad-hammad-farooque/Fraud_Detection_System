"""
Retraining Pipeline
-------------------
Loads labeled transactions from the database, recomputes features,
trains a calibrated LightGBM challenger, evaluates it against the current
deployed model, and registers and activates the new model only if it wins.

run() takes any database session, so the same pipeline trains the shipped
model on synthetic data (app/ML/train_model.py) and retrains on real labels.

Usage:
    python -m scripts.retrain
"""

import os
import sys
from dataclasses import asdict
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
import time

# Make sure the project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app import models
from app.features import FEATURE_ORDER, TransactionInput
from app.fraud_detection import features_at
from app.ML.registry import ModelManifest, RegistryError, activate, load_model, save_model
from app.ML.scorer import build_scorer
from app.features import FeatureVector

LOG_PATH   = os.path.join(os.path.dirname(__file__), "..", "logs", "retrain.log")


def log(msg: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
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


IMPORTANCE_REPEATS = 3
CALIBRATION_FRACTION = 0.2          # the latest slice of the training window
LATENCY_SAMPLES = 300

# A higher amount must never lower the score (§5.7). The model is constrained
# to be non-decreasing in every feature that rises with the amount...
MONOTONE_INCREASING = (
    "amount",
    "amount_deviation",
    "amount_zscore",
    "amount_population_percentile",
    "ratio_to_lifetime_max",
)
# ...and left without the two amount features that do not rise with it:
# is_round_amount flips 1-0-1 across 500, 501, 600, and amount_band_share
# jumps between order-of-magnitude bands. A model using either cannot be
# monotone in amount whatever its constraints. Serving still computes and
# audits both; the model simply does not read them.
NOT_MONOTONE_IN_AMOUNT = ("is_round_amount", "amount_band_share")
MODEL_FEATURES = [name for name in FEATURE_ORDER if name not in NOT_MONOTONE_IN_AMOUNT]
# Isotonic regression overfits small calibration sets - scikit-learn advises
# against it below roughly a thousand samples - and with a few dozen fraud
# cases it collapses every probability onto a handful of steps, which throws
# away the ranking PR-AUC depends on. Below this many fraud cases in the
# calibration window, Platt scaling (a sigmoid) is used instead. It is
# decided by rule, before evaluation, so the test window never chooses it.
ISOTONIC_MIN_POSITIVES = 1000

# Promotion (§11.2b, convention 20). Fraud is rare, and ROC AUC barely moves
# between models that differ a lot where it matters, so the challenger must win
# on PR-AUC and may not give up more than this much ROC AUC doing it.
MAX_AUC_REGRESSION = 0.01


# Candidate capacities, chosen between on the training window only. The first
# is the spec's contract. With a few hundred fraud cases each weighted ~150x,
# it memorises the positives: on the 50k synthetic data it lost to the forest
# (PR-AUC 0.850 vs 0.871). The others trade capacity for regularisation. The
# test window never takes part in this choice.
CANDIDATES = (
    {"num_leaves": 31, "min_child_samples": 20},
    {"num_leaves": 15, "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8},
    {"num_leaves": 7, "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
     "reg_lambda": 1.0},
)
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 100


def make_booster(scale_pos_weight: float, n_estimators: int = MAX_TREES, **capacity) -> lgb.LGBMClassifier:
    """The gradient-boosted challenger (T-19). scale_pos_weight rebalances the
    rare fraud class from the actual ratio; the monotone constraints make a
    larger amount never less risky."""
    params = {"num_leaves": 31, "min_child_samples": 20, **capacity}
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=[1 if name in MONOTONE_INCREASING else 0 for name in MODEL_FEATURES],
        monotone_constraints_method="advanced",
        random_state=42,
        verbose=-1,
        **params,
    )


def select_booster(X_fit, y_fit, X_val, y_val, scale_pos_weight: float) -> tuple[lgb.LGBMClassifier, dict]:
    """Fit every candidate with early stopping on the validation slice and keep
    the one with the best validation PR-AUC. Returns it with its settings."""
    best, best_info = None, None
    for capacity in CANDIDATES:
        booster = make_booster(scale_pos_weight, **capacity)
        booster.fit(X_fit, y_fit, eval_X=(X_val,), eval_y=(y_val,), eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        score = float(average_precision_score(y_val, booster.predict_proba(X_val)[:, 1]))
        if best_info is None or score > best_info["validation_average_precision"]:
            best, best_info = booster, {**capacity, "trees": int(booster.best_iteration_ or MAX_TREES),
                                        "validation_average_precision": score}
    return best, best_info


def train_challenger(X_train, y_train) -> tuple[CalibratedClassifierCV, dict]:
    """Fit the booster on the earlier 80% of the training window and calibrate
    it on the later 20%.

    Calibrating on rows the booster trained on would only learn its
    overconfidence, and the test window must stay untouched for evaluation, so
    calibration gets its own slice - the latest, like the test window, so the
    split stays chronological. scale_pos_weight inflates probabilities by
    design; isotonic calibration maps them back to observed fraud rates, and
    being monotone it preserves both the ranking and the amount guarantee.
    """
    cut = int(len(X_train) * (1 - CALIBRATION_FRACTION))
    X_fit, y_fit = X_train.iloc[:cut], y_train[:cut]
    X_cal, y_cal = X_train.iloc[cut:], y_train[cut:]
    positives = int(y_fit.sum())
    scale_pos_weight = (len(y_fit) - positives) / max(positives, 1)
    method = "isotonic" if int(y_cal.sum()) >= ISOTONIC_MIN_POSITIVES else "sigmoid"

    # The calibration slice doubles as the early-stopping and selection set: it
    # is the only data after the fit window that is not the test window.
    booster, selection = select_booster(X_fit, y_fit, X_cal, y_cal, scale_pos_weight)
    calibrated = CalibratedClassifierCV(FrozenEstimator(booster), method=method).fit(X_cal, y_cal)
    if method == "sigmoid" and calibrated.calibrated_classifiers_[0].calibrators[0].a_ > 0:
        # A decreasing sigmoid would invert the ranking and break the amount guarantee.
        raise RuntimeError("Platt scaling fitted a decreasing curve; refusing an inverted model")
    info = {
        "scale_pos_weight": scale_pos_weight,
        "selection": selection,
        "calibration_method": method,
        "fit_rows": len(X_fit),
        "calibration_rows": len(X_cal),
        "calibration_fraud": int(y_cal.sum()),
    }
    return calibrated, info


def challenger_wins(new: dict, old: dict) -> tuple[bool, str]:
    """PR-AUC decides; ROC AUC may not fall by more than MAX_AUC_REGRESSION."""
    new_ap, old_ap = new["average_precision"], old.get("average_precision", 0.0)
    new_auc, old_auc = new["auc"], old.get("auc", 0.0)
    if new_ap <= old_ap:
        return False, f"challenger PR-AUC {new_ap:.4f} <= champion {old_ap:.4f}"
    if new_auc < old_auc - MAX_AUC_REGRESSION:
        return False, (f"challenger PR-AUC {new_ap:.4f} > {old_ap:.4f}, but ROC AUC fell "
                       f"{old_auc - new_auc:.4f} (limit {MAX_AUC_REGRESSION})")
    return True, f"challenger PR-AUC {new_ap:.4f} > champion {old_ap:.4f}; ROC AUC {new_auc:.4f} vs {old_auc:.4f}"


def reliability(y_true, y_prob, bins: int = 10) -> list[dict]:
    """Predicted versus observed fraud rate in equal-width probability bins,
    with how many transactions each bin holds - most sit near zero."""
    rows = []
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        mask = (y_prob >= low) & ((y_prob < high) if i < bins - 1 else (y_prob <= high))
        count = int(mask.sum())
        if count:
            rows.append({"low": low, "high": high, "count": count,
                         "predicted": float(y_prob[mask].mean()), "observed": float(y_true[mask].mean())})
    return rows


def expected_calibration_error(curve: list[dict]) -> float:
    total = sum(row["count"] for row in curve)
    return sum(row["count"] / total * abs(row["predicted"] - row["observed"]) for row in curve) if total else 0.0


def measure_latency(model, X, samples: int = LATENCY_SAMPLES) -> dict:
    """Single-transaction scoring time through the function serving uses
    (app/ML/scorer.py), from FeatureVector to probability - what one request
    pays. Returns milliseconds at p50 and p99."""
    score = build_scorer(model, list(X.columns))
    defaults = {name: 0.0 for name in FEATURE_ORDER}
    vectors = [FeatureVector(**{**defaults, **X.iloc[i % len(X)].to_dict()}) for i in range(samples)]
    for fv in vectors[:20]:
        score(fv)                                            # warm up
    timings = []
    for fv in vectors:
        started = time.perf_counter()
        score(fv)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return {"inference_ms_p50": timings[len(timings) // 2], "inference_ms_p99": timings[int(len(timings) * 0.99)]}


def recall_at_fpr(y_true, y_prob, max_fpr: float = 0.01) -> float:
    """The detection rate achievable while blocking at most `max_fpr` of good
    transactions - the operating metric from §10.5, and the §8 success target."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(tpr[fpr <= max_fpr].max()) if (fpr <= max_fpr).any() else 0.0


def evaluate(model, X_test, y_test, label: str) -> dict:
    y_prob      = model.predict_proba(X_test)[:, 1]
    y_pred      = (y_prob >= 0.5).astype(int)
    curve       = reliability(y_test, y_prob)
    metrics = {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "recall_at_1pct_fpr": recall_at_fpr(y_test, y_prob, 0.01),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "expected_calibration_error": expected_calibration_error(curve),
    }
    metrics["_reliability"] = curve                        # stored on the manifest, not as a metric
    cm          = confusion_matrix(y_test, y_pred, labels=[0, 1])

    log(f"--- {label} ---")
    log(f"AUC-ROC: {metrics['auc']:.4f}   PR-AUC: {metrics['average_precision']:.4f}   "
        f"recall at 1% FPR: {metrics['recall_at_1pct_fpr']:.2%}   "
        f"Brier: {metrics['brier']:.5f}   ECE: {metrics['expected_calibration_error']:.4f}")
    log("Reliability (predicted vs observed fraud rate): " + "; ".join(
        f"[{r['low']:.1f},{r['high']:.1f}) n={r['count']} {r['predicted']:.3f}->{r['observed']:.3f}" for r in curve))
    log(f"Confusion Matrix:\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}")
    log(f"Report:\n{classification_report(y_test, y_pred, labels=[0, 1], target_names=['Legit','Fraud'], zero_division=0)}")
    return metrics


def measure_importance(model, X_test, y_test) -> dict[str, float]:
    """Permutation importance on the held-out window: how much AUC falls when
    each feature is shuffled. Unlike impurity importance it is measured on data
    the model never saw, and it does not favour high-cardinality features."""
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                    n_repeats=IMPORTANCE_REPEATS, random_state=42, n_jobs=-1)
    return {name: float(value) for name, value in zip(X_test.columns, result.importances_mean)}


@dataclass
class RetrainResult:
    promoted: bool
    reason: str
    challenger_metrics: dict = field(default_factory=dict)
    champion_metrics: dict = field(default_factory=dict)
    feature_importance: dict = field(default_factory=dict)
    manifest: ModelManifest | None = None
    challenger: object = None
    train_rows: int = 0
    test_rows: int = 0


def run(db=None, root: Path | None = None, activate_on_win: bool = True) -> RetrainResult:
    """Train a challenger on mature confirmed outcomes and promote it only if it
    beats the active model on the same held-out time window.

    `db` defaults to the application database; `root` to the application's
    model registry.
    """
    log("=" * 60)
    log("Retraining pipeline started")

    own_session = db is None
    db = db or SessionLocal()
    try:
        # ── 1. Load labeled data ──────────────────────────────────
        df = load_labeled_data(db)
        if df.empty:
            log("No confirmed outcomes in transaction_outcomes — nothing to train on. Exiting.")
            log("Labels come from analyst review, chargebacks or exploration holdouts, never from claims.")
            return RetrainResult(False, "no confirmed outcomes")

        log(f"Labeled samples: {len(df)}  (fraud={df['fraud'].sum()}, legit={(df['fraud']==0).sum()})")
        log(f"Label sources: {df['source'].value_counts().to_dict()}")

        if len(df) < MIN_LABELLED_SAMPLES:
            log(f"Only {len(df)} confirmed outcomes, need at least {MIN_LABELLED_SAMPLES}. Exiting.")
            return RetrainResult(False, "too few confirmed outcomes")

        if df["fraud"].nunique() < 2:
            log("Only one class in labeled data — cannot train. Exiting.")
            return RetrainResult(False, "one class only")

        # ── 2. Drop labels that have not ripened yet ──────────────
        before = len(df)
        df = apply_label_maturity(df)
        log(f"Label maturity: dropped {before - len(df)} transaction(s) newer than "
            f"{LABEL_MATURITY_DAYS} days; {len(df)} remain")

        if len(df) < MIN_LABELLED_SAMPLES or df["fraud"].nunique() < 2:
            log("Not enough mature labelled data after the maturity cutoff. Exiting.")
            return RetrainResult(False, "too little mature data")

        # ── 3. Build features ─────────────────────────────────────
        started = datetime.now(timezone.utc)
        df = build_features(db, df)
        log(f"Features built for {len(df)} rows in {(datetime.now(timezone.utc) - started).total_seconds():.0f}s")

        # ── 4. Chronological split ────────────────────────────────
        # A random split on time-ordered data trains on the future to predict
        # the past, which inflates every metric. The cut is on time instead.
        train_df, test_df = chronological_split(df)
        if train_df.empty or test_df.empty:
            log("Chronological split left one side empty. Exiting.")
            return RetrainResult(False, "empty split")

        # DataFrames, not arrays: the model keeps its feature names, which the
        # registry checks against FEATURE_ORDER before it will store or load it.
        X_train, y_train = train_df[MODEL_FEATURES], train_df["fraud"].values
        X_test, y_test = test_df[MODEL_FEATURES], test_df["fraud"].values
        X_all_test = test_df[FEATURE_ORDER]                  # for a champion on other features
        log(f"Train size: {len(X_train)}  ({_date_range(train_df)})  fraud={int(y_train.sum())}")
        log(f"Test  size: {len(X_test)}  ({_date_range(test_df)})  fraud={int(y_test.sum())}")

        if len(set(y_test)) < 2 or len(set(y_train)) < 2:
            log("A window contains a single class — AUC is undefined. Exiting.")
            return RetrainResult(False, "single-class window")

        # ── 5. Train new model ────────────────────────────────────
        new_model, training_info = train_challenger(X_train, y_train)
        log(f"LightGBM fitted on {training_info['fit_rows']} rows (scale_pos_weight "
            f"{training_info['scale_pos_weight']:.1f}); {training_info['calibration_method']} calibration "
            f"on the next {training_info['calibration_rows']} ({training_info['calibration_fraud']} fraud)")
        log(f"Selected on the training window: {training_info['selection']}")
        new_metrics = evaluate(new_model, X_test, y_test, "New Model")
        reliability_curve = new_metrics.pop("_reliability")
        new_metrics.update(measure_latency(new_model, X_test))
        log(f"Inference: p50 {new_metrics['inference_ms_p50']:.2f} ms, p99 {new_metrics['inference_ms_p99']:.2f} ms")
        importance = measure_importance(new_model, X_test, y_test)
        top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
        log("Top features by permutation importance: " + ", ".join(f"{k}={v:.4f}" for k, v in top))

        result = RetrainResult(False, "", challenger_metrics=new_metrics, feature_importance=importance,
                               challenger=new_model, train_rows=len(train_df), test_rows=len(test_df))

        # ── 6. Evaluate the active model on the same test window ──
        try:
            old_model, old_manifest = load_model(root=root)
            log(f"Active model: {old_manifest.version}")
            # The champion may use a subset of the features; give it its own.
            result.champion_metrics = evaluate(old_model, X_all_test[old_manifest.feature_order], y_test,
                                               "Current Model")
            result.champion_metrics.pop("_reliability")
        except RegistryError as exc:
            log(f"No loadable active model ({exc}). The challenger wins by default.")
            result.champion_metrics = {"auc": 0.0}

        # ── 7. Champion / challenger decision ─────────────────────
        old_auc = result.champion_metrics["auc"]
        wins, reason = challenger_wins(new_metrics, result.champion_metrics)
        if wins:
            result.manifest = save_model(
                new_model,
                algorithm=f"LGBMClassifier+{training_info['calibration_method']}",
                training_rows=len(train_df),
                metrics={**new_metrics, "test_rows": len(test_df), "champion_auc": old_auc,
                         "champion_recall_at_1pct_fpr": result.champion_metrics.get("recall_at_1pct_fpr", 0.0),
                         "champion_average_precision": result.champion_metrics.get("average_precision", 0.0),
                         "scale_pos_weight": training_info["scale_pos_weight"],
                         "trees": training_info["selection"]["trees"],
                         "num_leaves": training_info["selection"]["num_leaves"],
                         "validation_average_precision": training_info["selection"]["validation_average_precision"]},
                feature_importance=importance,
                reliability=reliability_curve,
                root=root,
            )
            result.promoted = True
            result.reason = reason
            if activate_on_win:
                activate(result.manifest.version, root=root)
                log(f"New model {result.manifest.version} registered and activated. "
                    f"{result.reason}. Restart the API to serve it.")
            else:
                log(f"New model {result.manifest.version} registered, not activated. {result.reason}.")
        else:
            result.reason = reason
            log(f"Current model retained. {result.reason}")
        return result

    finally:
        if own_session:
            db.close()
        log("Retraining pipeline complete")
        log("=" * 60)


if __name__ == "__main__":
    run()
