# SpecKit — Fraud Detection System

| Field | Value |
|---|---|
| **Project** | Fraud Detection System |
| **Version** | 0.1.0 |
| **Status** | Working prototype — not production ready. Phases A and B complete; Phase C 4 of 7 (T-15, T-16, T-18, T-19). Paused — see §1.5. |
| **Last reviewed** | 2026-09-19 (paused after T-19) |
| **Progress** | 19 / 34 tasks · 16 / 18 defects fixed (A10, A18 open) — Phases A and B complete, C under way — see §1.4 |
| **Stack** | FastAPI · SQLAlchemy · PostgreSQL · scikit-learn · Streamlit · Docker |

---

## 1. Overview

### 1.1 What this system does

A real-time **fraud detection platform for banking transactions**. Every transaction submitted
to the API is scored the moment it arrives, and an automated decision is returned before the
transaction is persisted:

- **ALLOW** — low risk, let it through
- **MANUAL_CHECK** — medium risk, needs a human look
- **REJECT** — high risk, block it

Customers who believe a decision was wrong can file a **dispute claim**, which runs through a
separate verification engine.

### 1.2 Why a hybrid engine

The scoring engine deliberately combines two approaches:

| Approach | Strength | Weakness |
|---|---|---|
| **Rule-based heuristics** | Explainable, instant to change, no training data needed, catches known patterns | Rigid, easy for attackers to probe and evade, cannot find unknown patterns |
| **Machine learning** | Finds non-obvious interactions, adapts to new data | Needs labelled data, opaque, degrades silently as behaviour drifts |

Neither is sufficient alone. Rules provide a hard floor of known-bad detection and give
regulators an auditable reason for every decision; the model adds sensitivity to patterns
nobody wrote a rule for. This is the standard architecture in payments risk.

### 1.3 Intended users

| Role | Needs | Currently supported |
|---|---|---|
| Customer | Submit transactions, see decisions, dispute them | Yes |
| Fraud analyst | Review the manual-check queue, confirm fraud, tune rules | **Yes** — cases via the API (T-12), rules by editing `config/rules.yaml` (T-13); no analyst UI yet |
| Data scientist | Retrain, evaluate, monitor drift | Partially — scripts, plus a versioned model registry (T-13); no drift monitoring |
| Auditor / regulator | Reconstruct why any decision was made | **Yes, via the API** (T-11) — every decision since T-11 replays bit-for-bit; earlier ones were never recorded |

### 1.4 Progress

Phases A and B are complete; Phase C is four tasks in. Each task below is one commit, with its defects struck through in §4.2
and its acceptance boxes ticked in §12.

| Task | Commit | Fixed |
|---|---|---|
| T-01 · Unified feature computation | `dfb0163` | A8, A16 |
| T-02 · SQL aggregates | `d73c065` | A7 |
| T-03 · Score saturation, rules split from policy | `309424e` | A5, A6 |
| T-04 · Policy layer | `dc1bb11` | P1, P4 |
| T-05 · Ground-truth outcomes | `aa35915` | A1, A4 |
| T-06 · Chronological split and label maturity | `38d2968` | A2 |
| T-07 · Monitoring ground truth | `935151b` | A3 |
| T-08 · Correctness fix bundle | `2fedbef` | A11, A12, A14 (A10 documented, not fixed — see below) |
| T-09 · Test infrastructure and rule coverage | `5cff5f0` | A17 |
| T-10 · Role-based access control | `b1e3e38` | prerequisite for T-12 |
| T-12 · Analyst case queue | `f50ea16` | A13 (REVIEW / MANUAL_REVIEW) |
| T-11 · Immutable decision audit trail | `969e0f0` | P6 |
| T-13 · Model registry and config-driven rules | `2d2550c` | — |
| T-14 · Idempotency | `fcbe145` | A15 |
| T-14b · Step-up authentication flow | `25c6a57` | A13 (`STEP_UP`) |
| T-15 · Expand the transaction schema | `2a847d1` | prerequisite for T-16, T-17 |
| T-16 · Feature expansion (retraining moved to T-18) | `05d6902` | two training bugs from T-05 |
| T-18 · Realistic training data, model retrained on 42 features | `0678d1a` | A9 |
| T-19 · LightGBM with calibration (built; not promoted — a tie) | `46ea1df` | — |

**Suite:** 606 tests plus 1 expected failure (strict amount monotonicity, for T-19), about
three minutes: two modules generate and train on real data, and four tests start the API in
a subprocess to prove startup checks.

**Still open:** A10 — the served probability is uncalibrated, because calibrated LightGBM
(T-19) tied the forest and the gate kept the forest; and A18 (`networkx` unused — kept
deliberately, T-17 uses it). **Correction:** A10 was counted as fixed from Phase A until
T-19. T-08 only documented it, and the defect row itself was never struck. The count was one
too high throughout.

**The label loop is closed.** Resolving a case writes an ANALYST `TransactionOutcome`,
which is exactly what `retrain.py` and `monitor.py` read. Retraining still needs at least
20 mature labels (90 days old) before it will train, so the loop turns slowly by design.

**The critical path is complete** (T-01 → T-03 → T-04 → T-05 → T-12 → T-11). Every decision
is now scored on one feature path, decided by a separate policy, recorded immutably, and —
when it needs a human — routed to an analyst whose determination becomes a training label.

**Next:** T-17 (graph features) closes A18 and may break the LightGBM/forest tie; T-20
(anomaly layer) is also unblocked. Phase D is independent. A decision is also open: promote
the calibrated LightGBM despite the tie, which would close A10 and the monotonicity `xfail`.

**The shipped model changed in T-18.** It scores with 42 features and was trained on
synthetic data; its metrics are real only for that data. Decisions made before T-18 used the
baseline and still replay exactly - the audit trail stores the model version per decision.

**Operational to-dos that are not tasks:** schedule `scripts/expire_challenges.py` (every
minute is reasonable), and replace `LogSender` with a real delivery channel before any
production use — it writes one-time codes to the server log.

**Retuning:** edit `config/rules.yaml`. It is validated before use and picked up on the next
request. The first version is `rules-1` / `policy-1`, which reproduces the pre-T-13 numbers
exactly — a test pins that.

**Carried debt, not yet a task:** schema changes land while `create_all()` is still the only
deployment path, so `migrations/` holds hand-written DDL for databases created earlier —
`001` to `010` so far. `007` also settles any transaction left in
STEP_UP from before T-14b as rejected, since those customers were never challenged. `003` also opens a case for every transaction already sitting in
REVIEW, so none stay dead ends. `004` adds a trigger that makes `decision_audits`
append-only in PostgreSQL, and deliberately does *not* backfill audit rows for older
transactions: their feature vectors were never stored, and a reconstructed record would look
authoritative when it is not. T-21 must baseline every file there
when it brings in Alembic.

**Bootstrapping an admin:** the role endpoint needs an admin to call it, so the first one is
created out of band with `python -m scripts.set_role someone@bank.com ADMIN`.

### 1.5 Where we left off (2026-09-19)

Work paused after T-19. Everything is committed and pushed; the working tree is clean.

**A decision is open — promote the calibrated LightGBM or keep the forest?**
T-19's challenger tied the T-18 forest on the synthetic held-out window (PR-AUC 0.8678 vs
0.8712 — about one transaction with 72 positives) and beat it on ROC AUC, calibration and
speed, but the gate decides on PR-AUC and kept the forest.

- *Promote it:* closes A10 (the served probability becomes calibrated), makes the amount
  monotonicity guarantee hold in serving (the `xfail` in `tests/test_calculate_risk.py` then
  passes and must be removed), and scores in ~0.5 ms. It is a manual override of the gate, so
  per convention 20 the reason is written beside the manifest. The challenger was not saved -
  it lost - so promoting means re-running `python -m app.ML.train_model` with an explicit
  override, which does not exist yet and would need adding.
- *Keep the forest:* nothing changes; A10 stays open until a LightGBM model wins on merit.
  T-17's graph features are the most likely thing to break the tie either way.

**Next task:** T-17 (graph features), which also closes A18. T-20 (anomaly layer) and all of
Phase D are unblocked as well.

**Operational to-dos — not tasks, but required before any real deployment:**

1. Run `migrations/001`-`010` in order on any PostgreSQL database created before them.
   `010` uses `CREATE INDEX CONCURRENTLY` and must run outside a transaction block.
2. Create the first admin: `python -m scripts.set_role someone@bank.com ADMIN`.
3. Schedule `python -m scripts.expire_challenges` (every minute is reasonable).
4. Replace the step-up `LogSender` with a real delivery channel; it writes one-time codes
   to the server log. Keep `STEP_UP_DEV_ECHO` off outside local development.
5. Build and run the Docker image once. It now installs `libgomp1` for LightGBM, and has
   never been built in this project's sessions — no Docker was available.
6. Verify on PostgreSQL that sessions run in UTC (`app/database.py` sets it): hour-of-day
   features depend on it, and the test suite only runs on SQLite.

**Things that are not true yet, however the numbers read:** every model metric in this file
is measured on synthetic data with patterns this project planted (T-18). Real performance is
unknown until analyst decisions (T-12) and chargebacks accumulate as labels.

**To reproduce the shipped model:** `python -m app.ML.train_model --regenerate` rebuilds the
50,000-row synthetic database (`data/`, git-ignored) and runs the same gate. The run takes about
five minutes.

---

## 2. Architecture

### 2.1 Request flow

```
Streamlit UI  --HTTP-->  FastAPI  -->  Router  -->  Service layer  -->  SQLAlchemy  -->  PostgreSQL
(frontend/)              (app/main)   (routers/)   (services/,           (models.py)
                                                    fraud_detection.py)
                                                          |
                                                          +--> active model (app/ML/artifacts/,
                                                               scored via app/ML/scorer.py)
```

### 2.2 Scoring pipeline

```
POST /transactions/
        |
        v
  Load user's transaction history
        |
        v
  compute_features()  --  app/features.py   (single feature path, T-01)
        |
        v
  evaluate_rules()  --  app/scoring.py       (T-03)
  |-- R1_HIGH_AMOUNT      : amount > 5000                     0.25
  |-- R2_AMOUNT_DEVIATION : amount > 3x user's average        0.25
  |-- R3_NEW_LOCATION     : location never seen for this user 0.20
  |-- R4_FLAGGED_DEVICE   : device shared by 3+ users         0.30
  +-- R5_VELOCITY         : 5+ transactions in last 120s      0.50
        |
        v
  rule_score = fired weight / 1.5          (normalised to [0, 1])
  risk_score = 0.7 * rule_score + 0.3 * model_probability
        |
  policy.decide()  --  app/policy.py            (T-04)
        |-- < 0.3   ->  ALLOW
        |-- < 0.5   ->  STEP_UP   (additional authentication)
        |-- < 0.7   ->  REVIEW
        +-- >= 0.7  ->  REJECT
  thresholds tighten on high-value amounts, loosen for trusted customers
```

### 2.3 Layer responsibilities

| Layer | Path | Responsibility |
|---|---|---|
| Entry point | `app/main.py` | App construction, router registration |
| Routing | `app/routers/` | HTTP concerns only — validation, status codes, auth dependency |
| Business logic | `app/services/fraud_services.py`, `app/policy.py` | Risk banding, device fraud, claim verification; policy maps score to action |
| Features | `app/features.py` | Single pure feature computation path shared by serving and training |
| Scoring engine | `app/scoring.py`, `app/fraud_detection.py` | Rule evaluation, score aggregation, `ScoreBreakdown` |
| Inference | `app/ML/models.py`, `app/ML/registry.py` | Loads the active registered model, verifies its feature order, takes a `FeatureVector`, returns `(prediction, probability)` |
| Configuration | `app/config.py`, `config/rules.yaml` | Rule weights and thresholds, blend weights, policy bands — validated, hot-reloaded |
| Persistence | `app/models.py`, `app/database.py`, `app/repositories/` | ORM entities, engine, session, bounded scoring reads |
| Auth | `app/auth.py`, `app/dependencies.py` | JWT issue/decode, bcrypt hashing, `get_current_user` |
| Contracts | `app/schemas.py` | Pydantic v2 request/response models |
| Presentation | `frontend/` | Streamlit multipage UI, thin HTTP client |
| MLOps | `scripts/` | Retraining (champion/challenger), performance monitoring |

### 2.4 Domain model

```
User (1) ---< (N) Transaction (1) ---< (N) Claim

User         : id, name, email, hashed_password
Transaction  : id, user_id, location, amount, device_id, predicted_fraud,
               risk_score, risk_level, decision, policy_version, model_version,
               resolved_decision, idempotency_key, created_at,
               merchant_id, merchant_category, currency, channel, ip_address,
               card_token, external_txn_id, latitude, longitude   (payment context, T-15)
TransactionOutcome : id, transaction_id, is_fraud_confirmed, source,
               confirmed_by, confirmed_at, notes   (ground truth, T-05)
Claim        : id, transaction_id, reason, amount, status, created_at
```

---

## 3. Subsystems

### 3.1 Authentication

JWT bearer tokens (`python-jose`, HS256) with bcrypt password hashing via `passlib`.
`get_current_user` resolves the token to a `User` on every protected route. Every data
endpoint is scoped to `current_user.id`, so users cannot read each other's records.

### 3.2 Prediction engine

Five weighted rules blended with a `RandomForestClassifier` probability (§2.2). 42 features
are computed once in `app/features.py` for every decision and stored on its audit row. The
rules read the five baseline features below; the model, trained on synthetic data in T-18,
reads all 42:

| Feature | Meaning |
|---|---|
| `amount` | Transaction amount |
| `amount_deviation` | Ratio of amount to the user's historical average |
| `is_new_location` | 1 if the location has never been seen for this user |
| `is_flagged_device` | 1 if the device has been used by 3+ distinct users |
| `velocity_2m` | Count of this user's transactions in the last 120 seconds |

### 3.3 Claim verification

A three-step engine in `verify_claim`:

```
Step 1 — Serial claimer  : user has > 3 prior claims          -> REJECTED
Step 2 — Staleness       : transaction older than 90 days     -> REJECTED
Step 3 — Pattern match   : non-fraud txn AND first ever claim -> APPROVED
                           everything else                    -> MANUAL_REVIEW
```

### 3.4 MLOps scripts

**`scripts/retrain.py`** — champion/challenger retraining. Loads confirmed outcomes only,
drops labels younger than 90 days, rebuilds all 42 features with the queries serving uses
(as of each row's timestamp), splits chronologically, and trains a calibrated LightGBM
challenger with capacity chosen on the training window. It is registered and activated only
if it beats the active model on PR-AUC without losing more than 0.01 ROC AUC on the same
held-out window. Logs to `logs/retrain.log`.

**`app/ML/train_model.py`** — the same pipeline run on synthetic data from
`scripts/generate_data.py`, which is how the shipped model was produced.

**`scripts/monitor.py`** — performance monitoring over a rolling window. Reports labelled
coverage, confusion matrix, precision, recall, F1, false positive rate, the business metrics
from §10.5, and volume by decision and risk level. Detection metrics are suppressed below 20
confirmed outcomes. Warns if recall drops below 80% or FPR exceeds 10%.
Logs to `logs/monitor_YYYY-MM-DD.log`.

Neither script can do anything useful until something writes `TransactionOutcome` rows,
which is T-12.

### 3.5 Testing

635 tests across 25 modules, run against in-memory SQLite so no PostgreSQL is needed. `conftest.py`
drops and recreates all tables around every test for full isolation, and provides fixtures
for a registered user, auth headers, and a second user for cross-tenant isolation checks.

---

## 4. Current state assessment

### 4.1 What is solid

- Clean layered separation; no business logic leaking into route handlers
- Consistent auth enforcement and cross-user isolation on every endpoint
- Pydantic v2 contracts on every request and response
- Genuine MLOps thinking — champion/challenger gating and performance monitoring are
  well beyond what a project this size usually has
- 227 tests including cross-tenant isolation cases, at 98% coverage of `app/`
- One feature computation path shared by serving and training, pinned by a parity test
- Scoring is bounded: three SQL statements per decision regardless of history size
- Policy separated from scoring, with the score provably in [0, 1] without clamping
- Full Docker Compose stack (API + Postgres + frontend)

### 4.2 What is broken

These are defects in existing code, not missing features.

| # | Defect | Location |
|---|---|---|
| ~~A1~~ | ~~**Label inversion.** APPROVED claim is mapped to fraud=1, but `verify_claim` only approves when `is_fraud is False`. The fraud class therefore contains only non-fraud transactions, so every retrain teaches the inverse of reality.~~ **Resolved by T-05.** | `scripts/retrain.py:62` |
| ~~A2~~ | ~~**Temporal leakage.** Features are built in temporal order, then split with `train_test_split(random_state=42)`. A random split on time-ordered data trains on the future to predict the past. Needs a chronological cutoff.~~ **Resolved by T-06.** | `scripts/retrain.py` |
| ~~A3~~ | ~~**Polluted ground truth.** REJECTED claim is read as "actually legitimate", but claims are rejected for staleness and serial claiming — neither is a fraud judgement.~~ **Resolved by T-07.** | `scripts/monitor.py` |
| ~~A4~~ | ~~**Prediction used as label.** `is_fraud` is derived from `risk_level == "HIGH"`, then consumed downstream as truth. A model's own output must never become its training label.~~ **Resolved by T-05.** | `app/routers/transactions.py:36` |
| ~~A5~~ | ~~**Score saturation.** Rule weights sum to 1.8 against a cap of 1.0. Any transaction firing 3+ rules reaches 1.0 before the ML boost is added, so the model's contribution is silently discarded in exactly the cases that matter most.~~ **Resolved by T-03.** | `app/fraud_detection.py` |
| ~~A6~~ | ~~**Correlated rules double-count.** Rules 1 and 2 (high amount, high deviation) almost always fire together, producing 0.8 and an instant REJECT. A false-positive generator.~~ **Resolved by T-03.** | `app/fraud_detection.py` |
| ~~A7~~ | ~~**Unbounded history load.** Scoring calls `.all()` on the user's entire transaction history and iterates it in Python — O(n) latency and memory per request.~~ **Resolved by T-02.** | `app/routers/transactions.py:22` |
| ~~A8~~ | ~~**Feature logic triplicated.** Computed independently in `fraud_detection.py`, `retrain.py:build_features`, and `train_model.py`. Guaranteed train/serve skew.~~ **Resolved by T-01.** | three files |
| ~~A9~~ | ~~**Model has no real signal.** Trained on 20 hand-written rows that are perfectly separable on amount alone (legit <= 400, fraud >= 5000), fit on 100% of the data with no holdout.~~ **Resolved by T-18.** | `app/ML/train_model.py` |
| A10 | **Uncalibrated probabilities used arithmetically.** RandomForest `predict_proba` is poorly calibrated, yet is multiplied by 0.3 and summed into the score as if it were a true probability. **Still open.** T-08 documented it; T-19 built calibrated LightGBM, but it has not beaten the forest through the promotion gate, so the served probability is still uncalibrated. | `app/fraud_detection.py` |
| ~~A11~~ | ~~**Feature-name mismatch.** A bare numpy array is passed to a model fitted on a DataFrame — emits a warning and relies silently on positional order.~~ **Resolved by T-08.** | `app/ML/models.py:15` |
| ~~A12~~ | ~~**Naive datetime columns.** `Column(DateTime)` without `timezone=True`, forcing scattered `.replace(tzinfo=utc)` patches at every use site.~~ **Resolved by T-08.** | `app/models.py` |
| ~~A13~~ | ~~**Dead decision states.** `REVIEW`, `MANUAL_REVIEW` and `STEP_UP` are terminal — no code path can resolve them.~~ **Resolved by T-12** (`REVIEW`, `MANUAL_REVIEW`) **and T-14b** (`STEP_UP`). | system-wide |
| ~~A14~~ | ~~**Claim amount unvalidated server-side.** The backend accepts any amount regardless of the transaction's value; only the frontend enforces a maximum.~~ **Resolved by T-08.** | `app/routers/claims.py` |
| ~~A15~~ | ~~**No idempotency.** A retried POST creates a duplicate transaction and falsely inflates the velocity rule.~~ **Resolved by T-14.** | `app/routers/transactions.py` |
| ~~A16~~ | ~~**No cold-start handling.** A user's first transaction always has empty known locations, so `is_new_location` fires for every new customer.~~ **Resolved by T-01.** | `app/fraud_detection.py` |
| ~~A17~~ | ~~**Test DB is file-based, not in-memory** as the docstring and README both claim. File-based SQLite can leak state between runs.~~ **Resolved by T-09.** | `tests/conftest.py` |
| A18 | **`networkx` declared but never imported.** Dead dependency. **Kept deliberately: T-17 uses it for graph features.** | `pyproject.toml` |

---

## 5. What to add for best performance

### 5.1 Prediction accuracy

The engine's ceiling is set by its features, not its model. Five features is roughly 2% of
what a production fraud engine uses.

**Prerequisite — expand the transaction schema.** ✅ Done in T-15. Most high-value features
could not be built because the columns did not exist. Add: `merchant_id`, `merchant_category`, `currency`,
`channel` (web/mobile/pos/atm), `ip_address`, `card_token`, `external_txn_id`.

**Then build these feature families:**

| Family | Features | Why it matters |
|---|---|---|
| Multi-window velocity | counts and sums over 1m / 5m / 1h / 24h / 7d | You have only a 120s window. Card-testing attacks surface at 5m; account takeover at 24h. |
| Geo-velocity | distance from previous location, **implied travel speed**, distinct locations per 24h | Impossible travel is one of the single strongest fraud signals in existence. |
| Temporal | hour-of-day, day-of-week, is_night, time since last txn, account age | Fraud has a strong diurnal signature, and account age is a top-5 feature in most payment models. |
| Amount shape | z-score vs user, percentile vs population, round-number flag, ratio to lifetime max | Raw amount alone is weak; contextualised amount is strong. |
| Device | device age, devices-per-user, users-per-device, first-use flag | The current check is a single 1-hop degree threshold. |
| Behavioural deviation | distance from the user's typical hour / location / amount band / merchant category | Converts a generic model into a per-customer one. |
| Graph | shared-device ring size, connected-component size, 2-3 hop degree, community detection | Fraud is a network phenomenon. `networkx` is already a declared dependency and completely unused — the largest untapped lever in the project. |

### 5.2 Model architecture

1. **Replace additive scoring with a learned combination.** The current
   `rule_score + 0.3 * probability` is arbitrary and saturating (defect A5). Options, in
   increasing order of sophistication:
   - feed rule outputs in as *features* to the model rather than summing their weights onto its output
   - stack rules + model into a meta-learner
   - keep rules only as **hard overrides** (blocklist to instant reject, trusted customer to bypass) and let the model own the continuous score
2. **Swap RandomForest for gradient boosting** — LightGBM, XGBoost or CatBoost. Consistently
   better on tabular fraud, native categorical handling, stronger under class imbalance,
   faster inference. **Built in T-19; tied the forest and was not promoted (§1.5).**
3. **Calibrate the probability** with `CalibratedClassifierCV`, since the score is consumed
   numerically. **Built in T-19; serves only once a LightGBM model is promoted.**
4. **Add an unsupervised layer** — IsolationForest or an autoencoder, ensembled with the
   supervised model. Essential because labelled fraud is scarce and *novel* attack patterns
   have no labels at all.
5. **Two-stage scoring** — cheap rules resolve obvious cases; the expensive model runs only on
   the grey zone. Keeps p99 latency flat as the feature count grows.
6. **Cost-based thresholds.** `0.3` and `0.7` are round numbers. Optimise them against a real
   cost matrix: a missed fraud costs the transaction value, a false positive costs customer friction.

### 5.3 Training data

- **Realistic class imbalance** — real fraud is 0.1-1%, not the current 50/50
- **Overlapping distributions** — the current data is perfectly separable, which teaches nothing
- **Temporal structure**, so time-based splits are meaningful
- Or adopt a public dataset: **IEEE-CIS Fraud Detection** or **PaySim**
- **Imbalance handling** — `class_weight`, SMOTE, or focal loss
- **Chronological splits**, never random (defect A2)

### 5.4 Runtime performance

| Change | Impact |
|---|---|
| ~~Replace the unbounded `.all()` with SQL aggregates (`AVG`, `COUNT`, `EXISTS`)~~ **Done, T-02** | Removes the O(n) scoring path — the system's hardest scalability limit |
| Redis counters for velocity windows | Sub-millisecond velocity lookups instead of scanning history |
| ~~Composite index on `(user_id, created_at)`~~ **Done, T-02** | Every history query filters on exactly this pair |
| Async endpoints | Handlers are sync `def`, so each blocks a threadpool worker during DB I/O |
| Connection pool tuning (`pool_size`, `max_overflow`, `pool_pre_ping`) | The engine currently uses defaults |
| Pagination on list endpoints | `GET /transactions/` and `GET /claims/` are unbounded |
| Two-stage scoring | Avoids paying full model cost on obvious traffic |
| Caching of device flags and user aggregates | Removes repeated identical queries |

### 5.5 Fraud-domain capability

| Capability | Why it is required |
|---|---|
| **RBAC** (`customer` / `analyst` / `admin`) | Nothing can currently act on a manual-review decision |
| **`TransactionOutcome` table** | Investigator-confirmed labels, decoupled from the customer dispute flow. Fixes A1 and A4 at the root. |
| **Case management queue** | Assignment, status transitions, SLA timers, analyst notes — also the source of real training labels |
| **Decision audit trail** | Immutable record of every rule that fired, its contribution, the full feature vector, model version and final score. A regulatory requirement, not a nicety. |
| **Reason codes** | Human-readable adverse-action explanations on every REJECT |
| **SHAP explainability** | Per-decision attributions, stored with the audit record |
| ~~**Model registry**~~ **Done, T-13** | `model.pkl` has no version, training date, feature schema or metrics. Stamp `model_version` on every decision. |
| ~~**Config-driven thresholds**~~ **Done, T-13** | `5000`, `0.4`, `0.3/0.7` are hardcoded. Risk teams retune weekly; that cannot require a deploy. |
| **Blocklists / allowlists** | Known-bad devices and locations; trusted-customer bypass |
| **Chargeback ingestion** | The authoritative fraud label source in payments |
| ~~**Idempotency keys**~~ **Done, T-14** | Payment systems retry (defect A15) |
| **Customer notification** | Nothing currently informs a user that a transaction was blocked |
| **Shadow mode** | Run the challenger against live traffic without acting, compare, then promote. Current single-split AUC promotion is fragile. |
| **Drift detection** | PSI/KS tests on input feature distributions. `monitor.py` only watches output performance, which degrades weeks after the inputs do. |
| **Backtesting harness** | Replay historical traffic against a candidate model or ruleset |
| **Experiment tracking** | MLflow or equivalent; currently only a text log |

### 5.6 Production engineering

| Area | Required change |
|---|---|
| Migrations | Alembic. `Base.metadata.create_all()` cannot alter columns and runs on every boot. |
| CI/CD | No `.github/` exists. Add ruff + mypy + pytest with a coverage gate + docker build. |
| Logging | Zero `logging` calls in `app/`. Structured JSON logs with request and correlation IDs. |
| Health | `/health` (liveness) and `/ready` (DB reachable, model loaded) |
| Metrics | Prometheus: scoring latency p50/p99, decisions/sec by outcome, score distribution, rule-hit rates |
| CORS | No middleware configured at all |
| Rate limiting | `/auth/login` with no limit is a credential-stuffing target |
| Security headers | HSTS, CSP, X-Content-Type-Options |
| Exception handling | No global handler — unhandled errors leak stack traces |
| Docker | Single-stage, runs as root, `COPY . .` ships `.venv`, tests and `test.db` into the image. Needs multi-stage, a non-root user, and a `.dockerignore`. |
| Config | `pydantic-settings` to replace scattered `os.getenv` plus manual `raise ValueError` |
| API versioning | No `/v1` prefix |
| Token lifecycle | 30-minute access token with no refresh path and no revocation on logout |
| Secrets | `.env` only; no vault/KMS path for production |
| Dependencies | `requirements.txt` uses ranges and the Docker build ignores the existing `uv.lock` |

### 5.7 Testing

- Direct unit tests for `calculate_risk` — rule weights, rule interaction, and the score cap
- ML behaviour tests — monotonicity (a higher amount must not lower the score), feature schema stability, inference latency budget
- Contract tests between `frontend/api.py` and the backend OpenAPI schema
- Load tests (Locust/k6) against `POST /transactions/` — the endpoint carrying the O(n) defect
- Coverage gate enforced in CI
- Switch to true in-memory SQLite (`sqlite:///:memory:` with `StaticPool`), fixing A17

### 5.8 Frontend

- Token persistence — `st.session_state` alone means a browser refresh logs the user out
- Analyst UI — case queue, review actions, label submission
- Pagination and filtering on transaction and claim tables
- Error boundaries so backend failures surface cleanly
- Risk visualisation — per-rule score breakdown, trend charts, SHAP display

---

## 6. Target architecture

```
                      +-----------------------------------------+
   Customer UI ---->  |            FastAPI  /v1                 |
   Analyst UI  ---->  |  auth · RBAC · rate limit · idempotency  |
                      +--------------------+--------------------+
                                           |
                             +-------------v-------------+
                             |   features.py (single     |
                             |   computation path)       |<---- Redis (velocity counters)
                             +-------------+-------------+
                                           |
                    +----------------------+----------------------+
                    v                      v                      v
            Hard overrides          Supervised model         Anomaly model
            (blocklist,             (LightGBM,               (IsolationForest)
             allowlist)              calibrated)
                    +----------------------+----------------------+
                                           v
                                 Decision + reason codes
                                           |
                    +----------------------+----------------------+
                    v                      v                      v
              PostgreSQL             Audit trail             Case queue
                                     (immutable)             (analyst review)
                                                                  |
                                                                  v
                                                         TransactionOutcome
                                                         (ground truth labels)
                                                                  |
                                           +----------------------+
                                           v
                                 Retraining · drift detection · shadow mode
```

The critical loop this closes: **analyst review produces real labels**, which feed retraining,
which improves the model, which shrinks the review queue. The current system has no such loop —
it trains on its own predictions.

---

## 7. Execution plan

### Phase 1 — Correctness (blocks everything else) — **COMPLETE**

1. ~~Extract `app/features.py` as the single feature computation path — fixes A8~~ T-01
2. ~~Add the `TransactionOutcome` table and decouple labels from claims — fixes A1 and A4~~ T-05
3. ~~Chronological train/test split in `retrain.py` — fixes A2~~ T-06
4. ~~Correct the ground-truth mapping in `monitor.py` — fixes A3~~ T-07
5. ~~Fix score saturation: normalise weights or drop additive scoring — fixes A5 and A6~~ T-03
6. ~~Replace the unbounded `.all()` with SQL aggregates — fixes A7~~ T-02
7. ~~Pass a DataFrame to `predict_fraud` — fixes A11~~ T-08
8. ~~Timezone-aware datetime columns — fixes A12~~ T-08
9. ~~Server-side claim amount validation — fixes A14~~ T-08
10. ~~True in-memory test DB — fixes A17~~ T-09

### Phase 2 — Make it a fraud *system* — **COMPLETE**

11. ~~RBAC with `customer` / `analyst` / `admin`~~ T-10
12. ~~Case management queue for manual review — fixes A13~~ T-12, with T-14b for `STEP_UP`
13. ~~Immutable decision audit trail~~ T-11
14. ~~Model registry and `model_version` stamped on every decision~~ T-13
15. ~~Config-driven rule thresholds~~ T-13
16. ~~Idempotency keys — fixes A15~~ T-14

### Phase 3 — Prediction power

17. ~~Expand the transaction schema (merchant, currency, channel, IP, card token)~~ T-15
18. ~~Multi-window velocity and geo-velocity features~~ T-16
19. ~~Temporal, amount-shape, device and behavioural-deviation features~~ T-16
20. Graph features with `networkx` — fixes A18 — **next (T-17)**
21. ~~Realistic imbalanced training data — fixes A9~~ T-18
22. LightGBM with calibration — fixes A10 — **built in T-19, not promoted; A10 open (§1.5)**
23. Unsupervised anomaly layer (T-20)
24. Cost-based threshold optimisation

### Phase 4 — Production readiness

25. Alembic, CI, structured logging, health and metrics endpoints
26. CORS, rate limiting, security headers, global exception handler
27. Docker hardening, `pydantic-settings`, API versioning, refresh tokens
28. Load tests, contract tests, ML behaviour tests, coverage gate

### Phase 5 — Operational maturity

29. Drift detection (PSI/KS on feature distributions)
30. SHAP explainability with stored attributions
31. Shadow mode for challenger models
32. Backtesting harness and experiment tracking
33. Redis velocity counters, async scoring, read replicas

---

## 8. Success criteria

| Dimension | Current | Target |
|---|---|---|
| Features in the model | 42 (T-18) | 60+ |
| Training samples | ✅ 50k generated, realistic and imbalanced (T-18) | 50k+ realistic, imbalanced |
| Class balance | ✅ 0.70% fraud (T-18) | ~0.5% fraud |
| Scoring latency p99 | unmeasured, but O(1) in history since T-02 | < 100 ms, O(1) |
| Recall at fixed 1% FPR | 88.9% on synthetic held-out data; unmeasured on real data | > 70% |
| Label source | analyst confirmations via T-12; chargeback ingestion not yet built | analyst confirmations + chargebacks |
| Decision auditability | ✅ full feature vector, rule trace and parameters per decision, replayable (T-11) | full feature vector and rule trace per decision |
| Manual review resolution | ✅ analyst queue with SLA (T-12), API only | analyst queue with SLA |
| Deployment | `create_all()` on boot | Alembic migrations via CI |

---

## 9. Highest-leverage items

If only a handful of changes are made, these carry the most signal:

1. ~~**`features.py`** — a single computation path, eliminating train/serve skew~~ **Done, T-01**
2. ~~**`TransactionOutcome`** — breaks the circular labelling that makes the ML loop meaningless~~ **Done, T-05**
3. ~~**Fix score saturation** — the ML contribution is discarded when it matters most~~ **Done, T-03**
4. ~~**SQL aggregates instead of `.all()`** — removes the hardest scalability limit~~ **Done, T-02**
5. ~~**Decision audit trail** — the clearest signal of regulated-fintech experience~~ **Done, T-11**
6. **Graph features with `networkx`** (T-17) — the most technically impressive addition available

Graph features (T-17) are the one that remains, and they need T-15's schema expansion first.

---

# Part II — Industry Reference and Build Guide

Part I describes what this project *is* and what is wrong with it.
Part II describes what it should *become* and gives an executable task list to get there.

---

## 10. Industry reference architecture

This is the target the roadmap is aimed at. Every task in §12 exists to close a specific gap
against this reference.

### 10.1 Algorithms used in production fraud systems

| Rank | Approach | Where it is used | Why |
|---|---|---|---|
| 1 | **Gradient Boosted Trees** (LightGBM / XGBoost / CatBoost) | The primary scorer in the large majority of deployed payment-fraud engines | Best-in-class on tabular data, handles mixed types and nulls, strong under extreme imbalance via `scale_pos_weight`, scales to 300+ features at 1-10 ms inference |
| 2 | **Logistic regression** | Regulated lending and credit decisions; often a final calibration or policy layer on top of GBDT | Coefficients map directly to adverse-action reason codes, which regulation requires. Chosen for defensibility, not accuracy. |
| 3 | **Graph methods** | Ring, collusion and mule-network detection | Fraud is organised. Either graph *features* fed into GBDT (cheap, captures most of the value) or full GNNs such as GraphSAGE / relational GCN (expensive to operate). |
| 4 | **Sequence models** (LSTM / GRU / Transformer) | Behavioural drift over a customer's transaction sequence | Real but less common; high engineering cost and often modest lift over well-engineered aggregate features. |
| 5 | **Unsupervised anomaly detection** (IsolationForest, autoencoder) | Runs *alongside* the supervised model | A supervised model is structurally blind to attack patterns that have never been labelled. This layer is what catches novel fraud. |
| 6 | **Rules engines** | Every mature system, permanently | A new attack can be blocked by a rule in 20 minutes. A model cannot be retrained, validated and deployed that fast. Rules also carry hard regulatory constraints (sanctions screening) that must never be probabilistic. |

A mature stack runs **all of these simultaneously**, not one of them.

### 10.2 Decision pipeline

```
Transaction
     |
     v
[0] Hard blocks          sanctions, blocklist, regulatory        ~1 ms
     |                   deterministic, never probabilistic
     v
[1] Feature fetch        online store (Redis / DynamoDB)         ~20 ms
     |                   pre-computed streaming aggregates
     v
[2] Cheap rules          resolves the obvious majority            ~2 ms
     |
     v  (grey zone only)
[3] Model ensemble       GBDT + anomaly score + graph features   ~10 ms
     |
     v
[4] Calibration          raw score -> true probability
     |
     v
[5] POLICY LAYER         probability + amount + customer tier
     |                   + merchant risk appetite -> action
     v
ALLOW / STEP-UP / REVIEW / DECLINE
     |
     +--> audit log (feature vector, rules fired, model version, SHAP)
```

### 10.3 The six design principles

**P1 — The policy layer is separate from the model.**
The model emits exactly one calibrated probability. A separate policy layer converts that into an
action using business context: a 0.6 score on a 5-unit transaction and a 50,000-unit transaction
deserve different responses. Separation means risk teams retune policy daily without revalidating
or redeploying the model. *This project currently fuses the two inside `get_decision()`.*

**P2 — Features are computed once, in a feature store.**
Offline store for point-in-time-correct training data, online store for millisecond serving reads,
and one shared computation path between them. Train/serve skew is the leading silent killer of
production fraud models, and this architecture exists solely to prevent it.

**P3 — Velocity comes from stream processing, not queries.**
Sliding-window counters (1m / 5m / 1h / 24h / 30d) maintained in Flink or Kafka Streams and written
to the online store. History is never scanned at decision time.

**P4 — Step-up authentication is a first-class action.**
Industry does not think in allow/deny binaries. The middle action is *friction*: 3-D Secure, OTP,
biometric re-auth. This converts would-be false declines into completed transactions and is where
much of the commercial value lives.

**P5 — Latency is a hard budget.**
Card authorisation typically allows 100-300 ms end to end. Feature fetch dominates; model inference
is nearly free. The tiered design exists so full model cost is not paid on traffic that cheap rules
already resolved.

**P6 — Every decision is reconstructable.**
Full feature vector, every rule fired, model version, policy version, final score and SHAP
attributions, stored immutably per decision. Bank model-governance regimes (SR 11-7 and equivalents)
require independent validation and auditability.

### 10.4 The two genuinely hard problems

**Label delay.** Chargebacks arrive 30-120 days after the transaction. The labels available today
describe last quarter's fraud, and the data about the attack happening right now is unlabelled.
Mitigations: analyst decisions as fast proxy labels, customer fraud reports, and explicitly modelling
label maturity so models are never evaluated on a window that has not yet ripened.

**Selection bias / reject inference.** Outcomes are observed only for transactions that were
*allowed*. Everything declined has no ground truth, so naive training progressively reinforces the
model's own past decisions. Mitigations: a small random exploration holdout allowed through
deliberately, inverse propensity weighting, or explicit reject-inference modelling.
*This project has the extreme form: it trains on labels derived from its own output.*

### 10.5 Metrics the business actually watches

Not F1. The operating metrics are:

| Metric | Definition |
|---|---|
| Fraud loss (bps) | Fraud value as basis points of total processed volume |
| Approval rate | Share of transactions allowed — the metric executives watch |
| False positive ratio | Good transactions blocked per fraud caught, e.g. "8:1" |
| Review queue depth | Manual cases outstanding, and SLA breach rate |
| Recall at fixed FPR | Detection rate held at a fixed operational false-positive budget |

### 10.6 Gap table — reference vs this project

| Industry practice | This project today | Task |
|---|---|---|
| GBDT primary scorer | GBDT built and gated (T-19); it tied the forest, which still serves | T-19 |
| Feature store, one code path | ~~Features computed in three places~~ one path, `app/features.py` | ~~T-01~~ ✅ |
| Streaming velocity counters | ~~In-Python scan of full history~~ bounded SQL; Redis still T-33 | ~~T-02~~ ✅ |
| Policy layer separate from model | ~~Thresholds fused into the scorer~~ `app/policy.py` | ~~T-04~~ ✅ |
| Calibrated probability | Calibration built (T-19); the serving forest is still uncalibrated | T-19 |
| Step-up auth as a third action | ~~Allow / review / reject only~~ STEP_UP decided (T-04) and completed by an OTP challenge (T-14b) | ~~T-04, T-14b~~ ✅ |
| Chargeback and analyst labels | ~~Model's own output used as label~~ `TransactionOutcome`, unwritten until T-12 | ~~T-05~~ ✅ |
| Full decision audit log | ~~Nothing logged~~ append-only, replayable | ~~T-11~~ ✅ |
| Graph and ring features | 1-hop device degree only | T-17 |
| Multi-window velocity | ~~Single 120-second window~~ six windows, 1m to 30d | ~~T-16~~ ✅ |

---

## 11. How to work this document

Conventions for anyone — human or agent — executing the build guide in §12.

### 11.1 Task format

Every task states its dependencies, the defect it fixes, the files it touches, a concrete code
contract, and a checklist of acceptance criteria. A task is done only when every box is ticked.

### 11.2 Working rules

1. **One task per commit.** Never combine two task IDs in one change.
2. **Respect `Depends on`.** Tasks are ordered by dependency; starting out of order will fail.
3. **Run `pytest` before marking a task done.** The suite must stay green — currently 635 tests.
4. **Add tests in the same commit as the code.** A task with no new test is not complete.
5. **Do not change behaviour not named in the task.** Refactors that touch scoring must keep
   existing test expectations passing, or must update them explicitly and say why.
6. **Update this file.** Tick the acceptance boxes and mark the defect resolved in §4.2.
7. **Never delete a defect row** in §4.2 — strike it through and note the fixing task.

### 11.2b Conventions settled during Phase A

Decisions made while executing Phase A that later tasks must not silently undo.

1. **`compute_features` stays pure, and aggregates have one implementation.** No database, no
   clock, no I/O: the instant is `TransactionInput.at`. Aggregates come only from
   `app/repositories/transaction_repo.py`, through `fraud_detection.features_at`, which
   serving calls at request time and retraining calls at each row's own timestamp. Every
   query filters `created_at < at`. Never add a second, Python-side aggregate path: the
   parity test in `tests/test_training_parity.py` will catch any drift.
2. **Rule weights are normalised, never clamped.** `final_score` is in [0, 1] because
   `w_rules + w_model == 1.0` — which `config/rules.yaml` validation enforces — not because
   of a `min()`. Changing any rule weight changes the total and therefore rescales every
   score: retune the policy bands in the same edit, and bump both `version` fields.
3. **Rules are named.** Every rule carries a stable `id` such as `R4_FLAGGED_DEVICE`, because
   the audit trail stores those ids and `config/rules.yaml` is keyed by them. Renaming one is
   a schema change to both. A new rule needs its logic in `app/scoring.py`, its id in
   `app/config.py`'s `KNOWN_RULE_IDS`, and its weight in the YAML — the loader refuses any
   mismatch.
4. **Scoring never names an action.** `app/scoring.py` must not contain ALLOW, REVIEW,
   STEP_UP, REJECT or a threshold; `app/policy.py` must not compute a feature. A test in
   `tests/test_policy.py` enforces both directions.
5. **A prediction is never a label.** `Transaction.predicted_fraud` is model output.
   Training and monitoring read `TransactionOutcome` and nothing else.
6. **One timezone helper.** `features.as_utc` is the single place that normalises a stored
   timestamp, because SQLite returns naive datetimes whatever the column type. Do not
   reintroduce `.replace(tzinfo=utc)` at call sites.
7. **The test suite is in-memory.** `StaticPool` is required; without it each connection gets
   its own empty database. No test may write a file to the repository. Tests hash passwords
   at bcrypt's minimum cost; production keeps passlib's default.
8. **Schema changes need a hand-written migration** in `migrations/` until T-21, because
   `create_all()` cannot alter an existing table.
9. **Staff never use customer routes.** Transaction and claim endpoints are customer-only via
   `require_customer`; analyst and admin access goes through `/analyst` and `/admin` routers.
   New analyst endpoints use `require_analyst`, which admins also pass. Roles are read from
   the database per request, never from the token.
10. **The engine's decision is immutable.** `Transaction.decision` records what the engine
    decided and is never rewritten; a human or challenge outcome goes in
    `resolved_decision`. T-11's audit trail replays `decision`, so rewriting it would break
    replay.
11. **Labels come from lifecycle endpoints, never direct writes.** An ANALYST outcome is
    written only by resolving a case, which refuses to overwrite an existing outcome of any
    source. Future label sources (chargeback ingestion) follow the same rule. Step-up
    results are deliberately *not* a label source: an OTP proves possession of a phone, not
    the legitimacy of a payment.
12. **Every decision is audited in its own database transaction.** `record_decision` runs
    before the commit that saves the transaction, never after. Anything that changes how a
    score is computed — a new rule, a new weight, a new policy input — must also land in
    the audit row, or `replay()` stops reproducing it; `tests/test_audit.py` will fail if
    it does. `replay()` proves consistency, not authenticity: tamper-evidence such as a
    hash chain over rows is not yet built.
13. **Scoring reads one config snapshot per decision.** `score()` takes the config once and
    threads it through every rule and the blend, so a reload mid-request cannot mix two
    versions of the file. The snapshot's parameters go into `ScoreBreakdown.scoring_params`
    and from there into the audit row.
14. **No model is served without its manifest.** Train on a DataFrame with named feature
    columns - all of `FEATURE_ORDER` or a subset - save through `app/ML/registry.py`, and
    activate explicitly. Serving hands each model exactly the columns its manifest lists.
    Never write a `model.pkl` by hand; the registry will refuse to load it. Renaming or
    removing a feature breaks every model that uses it - add, never rename.
15. **Anything that moves money is idempotent.** `POST /transactions/` honours
    `Idempotency-Key`, and a replay must never rescore, re-audit or reopen a case. Any future
    endpoint that creates a payment-like record — refunds, chargeback ingestion, T-14b's
    challenge completion — follows the same pattern: check the key, then rely on a unique
    constraint to settle races, with the whole write inside one database transaction.
16. **Secrets and side effects wait for the commit.** Anything that leaves the system — an
    OTP, and later notifications or webhooks — is sent only after the database commit
    succeeds, never inside the transaction. One-time codes are stored only as keyed hashes
    and compared in constant time.
17. **Card numbers never enter the system.** Only vault tokens are stored, and the API
    rejects any `card_token` that passes the Luhn check as a 13-19 digit number. The same
    rule applies to any future field, log line or audit row that could carry a card number.
18. **Missing means `None`, normalised at the boundary.** Everything entering feature
    computation passes through `TransactionInput.from_request`, which turns `NaN`, `pd.NA`
    and `NaT` into `None`. Never test a DataFrame value with `is not None` anywhere else.
19. **Synthetic data stays quarantined.** `scripts/generate_data.py` writes only to a database
    holding nothing but `@synthetic.invalid` accounts, and labels only with `SYNTHETIC`
    outcomes. Never point it at the application database, and never report a metric
    measured on synthetic data without saying so.
20. **A model is promoted by the gate, not by hand.** `app/ML/train_model.py` and
    `scripts/retrain.py` both go through `retrain.run`: chronological split, held-out
    metrics, and promotion only when the challenger beats the active model on **PR-AUC**
    without losing more than 0.01 ROC AUC, on the same window. Hyper-parameters are chosen on
    the training window only, and the test window is looked at once per decision - never
    re-run until a model wins. A person may override the gate, but the reason goes in
    writing beside the manifest. Keep the previous model registered.
21. **Serving scores through `app/ML/scorer.py`,** held to within 1e-12 of `predict_proba`
    by test. Any new model type gets its fast path only with its own parity test; until
    then it falls back to `predict_proba`.

### 11.3 Definition of done

- [ ] Code implements the stated contract
- [ ] New unit tests cover the contract, including edge cases
- [ ] `pytest` passes with no new warnings
- [ ] Type hints on every new public function
- [ ] Docstrings on every new module and public function
- [ ] Acceptance boxes in this document ticked, and §1.4 updated with the commit
- [ ] Behaviour changes that force an existing test to be rewritten are called out in the
      commit message, with the reason stated in the test itself
- [ ] Unrelated working-tree changes stay out of the commit
- [ ] Committed with the task ID in the message, e.g. `T-01: unified feature computation`

### 11.4 Commit message format

```
T-<id>: <short imperative summary>

<what changed and why>
Fixes: A<n>[, A<n>]
```

---

## 12. Build guide

### Phase A — Correctness

These make the existing system behave the way it already claims to. Nothing in later phases is
worth doing first, because later phases amplify whatever is wrong here.

---

#### T-01 · Unified feature computation — ✅ DONE (`dfb0163`)

**Depends on:** none
**Fixes:** A8, A16
**Create:** `app/features.py`
**Modify:** `app/fraud_detection.py`, `scripts/retrain.py`, `app/ML/train_model.py`

The single most important structural change. One pure function computes features; every caller —
serving, retraining, initial training — routes through it. Purity is what makes it testable and
what guarantees train/serve consistency.

**Contract**

```python
"""Single source of truth for feature computation.

Every consumer of features - live scoring, retraining, initial training -
MUST compute them through this module. Never recompute features elsewhere.
"""
from dataclasses import dataclass, asdict, field
import pandas as pd

FEATURE_ORDER: list[str] = [
    "amount",
    "amount_deviation",
    "is_new_location",
    "is_flagged_device",
    "velocity_2m",
]

@dataclass(frozen=True)
class UserAggregates:
    """Pre-computed history for one user, sourced either from SQL (serving)
    or from a DataFrame slice (training). Never computed inside this module."""
    txn_count: int
    avg_amount: float
    known_locations: frozenset[str]
    count_last_2m: int
    is_cold_start: bool = field(default=False)

@dataclass(frozen=True)
class FeatureVector:
    amount: float
    amount_deviation: float
    is_new_location: int
    is_flagged_device: int
    velocity_2m: int

    def to_frame(self) -> pd.DataFrame:
        """Column order MUST match the model's feature_names_in_."""
        return pd.DataFrame([asdict(self)], columns=FEATURE_ORDER)

def compute_features(
    amount: float,
    location: str,
    aggregates: UserAggregates,
    device_user_count: int,
) -> FeatureVector:
    """Pure. No database access, no clock access, no I/O.

    Cold start: when aggregates.is_cold_start is True the user has no history,
    so amount_deviation defaults to 1.0 and is_new_location is 0 rather than 1
    (a first-ever transaction must not be penalised for a location that could
    not possibly have been seen before). Fixes A16.
    """
```

**Acceptance**

- [x] `compute_features` performs no I/O and takes no clock — fully deterministic
- [x] `app/fraud_detection.py` calls it instead of computing inline
- [x] `scripts/retrain.py:build_features` calls it per row instead of its own loop
- [x] `app/ML/train_model.py` builds its training matrix through `FEATURE_ORDER`
- [x] Cold-start users get `is_new_location == 0`, not 1
- [x] Unit tests: cold start, zero average amount, unknown location, exact-threshold values
- [x] A property test asserting serving and training paths produce identical vectors for identical input

---

#### T-02 · Replace history scan with SQL aggregates — ✅ DONE (`d73c065`)

**Depends on:** T-01
**Fixes:** A7
**Create:** `app/repositories/transaction_repo.py`
**Modify:** `app/routers/transactions.py`, `app/services/fraud_services.py`

Removes the system's hardest scalability limit. The current path loads every transaction a user
has ever made into Python on every single score.

**Contract**

```python
from datetime import datetime
from sqlalchemy.orm import Session
from app.features import UserAggregates

def get_user_aggregates(db: Session, user_id: int, now: datetime) -> UserAggregates:
    """One round trip. Uses COUNT, AVG and a bounded window filter.
    Must NOT materialise the user's transaction rows."""

def get_device_user_count(db: Session, device_id: str) -> int:
    """COUNT(DISTINCT user_id) for this device. Returns the count, not a bool,
    so callers can threshold it themselves."""
```

**Acceptance**

- [x] No `.all()` on a user's transaction history anywhere in the scoring path
- [x] `get_user_aggregates` issues at most two queries
- [x] Composite index added: `Index("ix_txn_user_created", user_id, created_at)`
- [x] Known-locations lookup uses `EXISTS`, not a full set materialisation
- [x] A test asserting scoring issues a bounded number of queries regardless of history size
- [x] Existing 76 tests still pass

---

#### T-03 · Fix score saturation and separate rules from policy — ✅ DONE (`309424e`)

**Depends on:** T-01
**Fixes:** A5, A6
**Create:** `app/scoring.py`
**Modify:** `app/fraud_detection.py`

Rule weights currently sum to 1.8 against a cap of 1.0, so the ML contribution is discarded on
any transaction firing three or more rules. Rules 1 and 2 are also near-duplicates that fire
together and force an instant reject.

**Contract**

```python
from dataclasses import dataclass
from collections.abc import Callable
from app.features import FeatureVector

@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    weight: float
    predicate: Callable[[FeatureVector], bool]

@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    weight: float

@dataclass(frozen=True)
class ScoreBreakdown:
    """Everything needed to reconstruct the decision. Feeds the audit log in T-11."""
    rule_hits: list[RuleHit]
    rule_score: float          # normalised to [0, 1]
    model_probability: float   # calibrated, [0, 1]
    final_score: float         # [0, 1]
    model_version: str

RULES: list[Rule] = [...]   # loaded from config in T-13, hardcoded for now

def evaluate_rules(fv: FeatureVector) -> list[RuleHit]: ...

def combine(rule_hits: list[RuleHit], model_probability: float) -> float:
    """Normalise rule_score by the sum of ALL rule weights so it cannot exceed 1.0,
    then blend:  final = W_RULES * rule_score + W_MODEL * model_probability
    with W_RULES + W_MODEL == 1.0. No min() clipping - the arithmetic
    cannot exceed 1.0 by construction. Fixes A5."""
```

**Acceptance**

- [x] `final_score` is provably in `[0, 1]` without a `min()` clamp
- [x] The model contribution is never discarded, at any rule-hit count
- [x] Rules 1 and 2 either merged or explicitly re-weighted to stop double-counting — decision documented in a comment
- [x] `ScoreBreakdown` returned from the scoring path, not just a float
- [x] Unit tests: all rules firing, no rules firing, each rule individually, exact threshold boundaries
- [x] A test asserting the score stays in range for 10,000 randomised feature vectors

---

#### T-04 · Policy layer — ✅ DONE (`dc1bb11`)

**Depends on:** T-03
**Fixes:** design principle P1, P4
**Create:** `app/policy.py`
**Modify:** `app/services/fraud_services.py`, `app/routers/transactions.py`

Separates "how risky is this" from "what should we do about it". Adds STEP_UP as a real action.

**Contract**

```python
from enum import StrEnum
from dataclasses import dataclass

class Decision(StrEnum):
    ALLOW   = "ALLOW"
    STEP_UP = "STEP_UP"     # new: request additional authentication
    REVIEW  = "REVIEW"
    REJECT  = "REJECT"

@dataclass(frozen=True)
class PolicyContext:
    amount: float
    customer_tier: str = "standard"
    account_age_days: int = 0

@dataclass(frozen=True)
class PolicyConfig:
    """Loaded from config, NOT hardcoded. Risk teams retune this without a deploy."""
    version: str
    allow_below: float
    step_up_below: float
    review_below: float
    high_value_amount: float   # above this, thresholds tighten

def decide(score: float, ctx: PolicyContext, cfg: PolicyConfig) -> Decision:
    """Maps a calibrated score plus business context to an action.
    Contains NO risk computation - it only reads the score."""
```

**Acceptance**

- [x] `app/scoring.py` contains no thresholds and no decision vocabulary
- [x] `app/policy.py` contains no feature or risk computation
- [x] Thresholds read from config, with the existing 0.3 / 0.7 as defaults
- [x] `STEP_UP` added to the `Decision` enum and handled by the frontend
- [x] `policy_version` recorded alongside every decision
- [x] High-value transactions demonstrably use tighter thresholds
- [x] Unit tests for every threshold boundary and both tiers

---

#### T-05 · Ground-truth outcomes, decoupled from claims — ✅ DONE (`aa35915`)

**Depends on:** none
**Fixes:** A1, A4
**Create:** `app/models.py::TransactionOutcome`, migration
**Modify:** `scripts/retrain.py`, `scripts/monitor.py`

Breaks the circular labelling that makes the entire ML loop meaningless. A customer dispute is
*not* a fraud determination; only an investigator or a chargeback is.

**Contract**

```python
class OutcomeSource(StrEnum):
    ANALYST    = "ANALYST"       # investigator determination
    CHARGEBACK = "CHARGEBACK"    # authoritative, arrives 30-120 days later
    CUSTOMER   = "CUSTOMER"      # self-reported, weaker
    EXPLORATION = "EXPLORATION"  # random holdout allowed through for unbiased labels

class TransactionOutcome(Base):
    __tablename__ = "transaction_outcomes"
    id                 = Column(Integer, primary_key=True)
    transaction_id     = Column(Integer, ForeignKey("transactions.id"), unique=True, nullable=False)
    is_fraud_confirmed = Column(Boolean, nullable=False)
    source             = Column(String, nullable=False)
    confirmed_by       = Column(Integer, ForeignKey("users.id"), nullable=True)
    confirmed_at       = Column(DateTime(timezone=True), nullable=False)
    notes              = Column(String, nullable=True)
```

**Acceptance**

- [x] `retrain.py` reads labels **only** from `TransactionOutcome`, never from `Claim` or `is_fraud`
- [x] `monitor.py` reads ground truth **only** from `TransactionOutcome`
- [x] `Transaction.is_fraud` renamed to `predicted_fraud` to stop the conflation
- [x] `retrain.py` exits cleanly with a clear log line when too few confirmed outcomes exist
- [x] Unique constraint on `transaction_id` — one outcome per transaction
- [x] Tests covering each `OutcomeSource`

---

#### T-06 · Chronological split and label maturity — ✅ DONE (`38d2968`)

**Depends on:** T-05
**Fixes:** A2
**Modify:** `scripts/retrain.py`

`build_features` carefully preserves temporal order, and then `train_test_split(random_state=42)`
throws that away by training on the future to predict the past.

**Contract**

```python
def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by created_at, cut at the (1 - test_fraction) quantile.
    All training rows strictly precede all test rows."""

def apply_label_maturity(df: pd.DataFrame, maturity_days: int = 90) -> pd.DataFrame:
    """Drop transactions newer than maturity_days. Their labels have not ripened -
    a chargeback may still arrive. Including them understates the fraud rate."""
```

**Acceptance**

- [x] No `train_test_split` anywhere in `retrain.py`
- [x] Every training timestamp is strictly earlier than every test timestamp
- [x] Immature labels excluded, with the count logged
- [x] The log states the train and test date ranges explicitly
- [x] Test asserting zero temporal overlap between the splits

---

#### T-07 · Correct the monitoring ground truth — ✅ DONE (`935151b`)

**Depends on:** T-05
**Fixes:** A3
**Modify:** `scripts/monitor.py`

A REJECTED claim currently reads as "transaction was legitimate", but claims are rejected for
staleness and serial claiming — neither is a statement about fraud.

**Acceptance**

- [x] `get_ground_truth` sources only from `TransactionOutcome`
- [x] Claim status no longer influences any metric
- [x] Report states the labelled coverage, e.g. "412 of 10,340 transactions have confirmed outcomes"
- [x] Business metrics added from §10.5: approval rate, false-positive ratio, fraud bps
- [x] Metrics suppressed with a clear warning when labelled volume is too small to be meaningful

---

#### T-08 · Correctness fix bundle — ✅ DONE (`2fedbef`)

**Depends on:** T-01
**Fixes:** A10, A11, A12, A14, A18
**Modify:** `app/ML/models.py`, `app/models.py`, `app/routers/claims.py`, `pyproject.toml`

Small independent fixes, grouped because each is a few lines.

**Acceptance**

- [x] `predict_fraud` accepts a `FeatureVector` and calls `.to_frame()` — no bare numpy array, warning gone (A11)
- [x] All `DateTime` columns become `DateTime(timezone=True)`; every ad-hoc `.replace(tzinfo=utc)` removed (A12)
- [x] `ClaimCreate` validates `amount > 0` and `amount <= transaction.amount`, server-side (A14)
- [x] `networkx` removed from dependencies, or T-17 scheduled to use it (A18)
- [x] Model probability documented as uncalibrated until T-19 lands (A10)
- [x] Migration written for the datetime column change

---

#### T-09 · Test infrastructure and rule coverage — ✅ DONE (`5cff5f0`)

**Depends on:** T-03
**Fixes:** A17
**Modify:** `tests/conftest.py`
**Create:** `tests/test_features.py`, `tests/test_scoring.py`, `tests/test_policy.py`

**Contract**

```python
# conftest.py - genuinely in-memory, matching what the docstring already claims
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,      # required, or each connection gets a fresh empty DB
)
```

**Acceptance**

- [x] No `test.db` file created by the suite
- [x] `StaticPool` used — without it, in-memory SQLite gives every connection its own database
- [x] README and the conftest docstring now match reality
- [x] `calculate_risk` directly unit-tested: each rule alone, all rules together, boundary values
- [x] Monotonicity test: raising `amount` with all else fixed must never lower the score
- [x] Coverage measured and reported

---

### Phase B — Make it a fraud system

---

#### T-10 · Role-based access control — ✅ DONE (`b1e3e38`)

**Depends on:** none
**Fixes:** prerequisite for T-12
**Modify:** `app/models.py`, `app/dependencies.py`, all routers

**Contract**

```python
class Role(StrEnum):
    CUSTOMER = "CUSTOMER"
    ANALYST  = "ANALYST"
    ADMIN    = "ADMIN"

def require_role(*allowed: Role):
    """Dependency factory. Raises 403 when current_user.role is not in allowed."""
```

**Acceptance**

- [x] `User.role` column added, defaulting to `CUSTOMER`, with a migration
- [x] Analyst endpoints reject customers with 403
- [x] Customer endpoints remain scoped to `current_user.id` — an analyst must not silently gain access to customer-scoped routes
- [x] Tests for every role against every endpoint

---

#### T-11 · Immutable decision audit trail — ✅ DONE (`969e0f0`)

**Depends on:** T-03, T-04
**Fixes:** design principle P6
**Create:** `app/models.py::DecisionAudit`

The clearest signal of regulated-fintech experience in the whole roadmap.

**Contract**

```python
class DecisionAudit(Base):
    __tablename__ = "decision_audits"
    id             = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False, index=True)
    feature_vector = Column(JSON, nullable=False)   # full input, for replay
    rule_hits      = Column(JSON, nullable=False)   # [{rule_id, weight}, ...]
    rule_score     = Column(Float, nullable=False)
    model_prob     = Column(Float, nullable=False)
    final_score    = Column(Float, nullable=False)
    decision       = Column(String, nullable=False)
    model_version  = Column(String, nullable=False)
    policy_version = Column(String, nullable=False)
    created_at     = Column(DateTime(timezone=True), nullable=False)
```

**As built:** three more JSON columns — `scoring_params` (rule-weight total and the blend
weights), `policy_config` (thresholds) and `policy_context` (amount, tier, account age).
Without them a row replays against *today's* settings, which T-13 makes configurable, so
"sufficient to replay exactly" would silently stop being true. The endpoint is served at
`/analyst/...` until T-27 adds `/v1`.

**Acceptance**

- [x] One audit row written for every scored transaction, in the same database transaction
- [x] Append-only — no update or delete path exists anywhere in the codebase
- [x] A stored audit row is sufficient to replay the decision exactly
- [x] `GET /v1/analyst/transactions/{id}/audit` exposes it to analysts only
- [x] Test asserting a replayed audit row reproduces the original score bit-for-bit

---

#### T-12 · Analyst case queue — ✅ DONE (`f50ea16`)

**Depends on:** T-10, T-05
**Fixes:** A13
**Create:** `app/routers/analyst.py`, `app/models.py::Case`

Closes the feedback loop: review produces labels, labels feed retraining, retraining shrinks
the queue. Without this the system can never learn.

**Contract**

```
GET   /v1/analyst/cases                 list queue, filter by status/priority, paginated
POST  /v1/analyst/cases/{id}/assign     claim a case
POST  /v1/analyst/cases/{id}/resolve    body: {is_fraud_confirmed: bool, notes: str}
                                        -> writes a TransactionOutcome with source=ANALYST
                                        -> transitions the case to RESOLVED
```

**Acceptance**

- [x] Every `REVIEW` decision automatically creates a `Case`
- [x] Resolving a case writes exactly one `TransactionOutcome`
- [x] `MANUAL_REVIEW` and `REVIEW` are no longer terminal states (A13)
- [x] Queue sorted by risk score and age; SLA breach flagged
- [x] Analyst-only access enforced via `require_role`
- [x] Tests for the full lifecycle: create, assign, resolve, label written

---

#### T-13 · Model registry and config-driven rules — ✅ DONE (`2d2550c`)

**Depends on:** T-03
**Create:** `app/ML/registry.py`, `config/rules.yaml`

**Contract**

```python
@dataclass(frozen=True)
class ModelManifest:
    version: str            # semver or timestamp
    trained_at: datetime
    feature_order: list[str]
    metrics: dict[str, float]
    training_rows: int
    algorithm: str

def load_model(version: str | None = None) -> tuple[Any, ModelManifest]:
    """Loads a specific version, or the active one. Raises if the manifest's
    feature_order does not match features.FEATURE_ORDER - this is the
    train/serve skew tripwire."""
```

**As built:**

- Models live in `app/ML/artifacts/<version>/` with `model.pkl` and `manifest.json`, and
  `artifacts/ACTIVE` names the version serving traffic. Versions are immutable; a model
  that fails the feature-order check can be neither saved nor activated.
- The tripwire checks both the manifest *and* the fitted model's `feature_names_in_`, so a
  hand-swapped `model.pkl` is caught too. A model fitted on a bare array is refused.
- `config/rules.yaml` holds rule weights and thresholds, the blend weights, and the policy
  bands and multipliers. Rule *logic* stays in `app/scoring.py`. The file is reloaded when
  it changes; an edit that fails validation is rejected and the last good version stays in
  force, so a typo during a retune cannot stop payments.
- The environment-variable thresholds from T-04 are gone: one source of truth.
- Activating a new model takes effect on the next API restart; config changes do not need one.

**Acceptance**

- [x] Every saved model has a manifest written beside it
- [x] Loading refuses to proceed on a feature-order mismatch
- [x] `model_version` stamped on every decision and audit row
- [x] Rule weights and policy thresholds load from `config/rules.yaml`
- [x] Changing a threshold requires no code change
- [x] Config schema validated on startup — malformed config fails fast, not at first request

---

#### T-14 · Idempotency — ✅ DONE (`fcbe145`)

**Depends on:** none
**Fixes:** A15
**Modify:** `app/routers/transactions.py`, `app/schemas.py`

**As built:**

- Follows the IETF Idempotency-Key draft: the same key sent with a *different* payment is
  refused with 422 rather than answered with the old one, which would silently drop a real
  payment from a buggy client. A replay carries `Idempotent-Replayed: true`.
- Concurrent duplicates are settled by the unique constraint: the loser's transaction,
  audit row and case roll back together and it returns the winner.
- A replay returns the transaction's *current* state, so `resolved_decision` may have been
  filled in since the first response. Every field the engine wrote is unchanged.
- Keys do not expire. Stripe-style 24-hour expiry would need a cleanup job and is not built.
- The Streamlit client sends one key per payment and retries network failures with it.

**Acceptance**

- [x] `Idempotency-Key` header accepted on `POST /transactions/`
- [x] A repeat key returns the original response without rescoring
- [x] Unique constraint on `(user_id, idempotency_key)`
- [x] Retries no longer inflate the velocity feature
- [x] Test: the same key twice yields one transaction and identical responses

---

#### T-14b · Step-up authentication flow — ✅ DONE (`25c6a57`)

**Depends on:** T-04
**Fixes:** A13 (the `STEP_UP` half)
**Added:** during T-12, when it became clear no task completes a `STEP_UP` decision

T-04 made `STEP_UP` a real policy action, but nothing acts on it: the customer is never
challenged, and the transaction sits in `STEP_UP` forever. Industry treats step-up as the
action that turns would-be false declines into completed payments (§10.3, P4), so it needs a
lifecycle of its own rather than being folded into the analyst queue.

**As built:**

- The acceptance line "failing sets `REJECT`" conflicted with "repeated failures open a
  `Case` rather than silently rejecting". Resolved in favour of the case: a wrong code spends
  an attempt, and exhausting them marks the challenge FAILED and holds the payment
  (`resolved_decision` stays empty) until an analyst settles it through T-12. Only expiry
  rejects automatically.
- Codes are six digits, stored only as an HMAC keyed with the server secret, compared in
  constant time, and delivered *after* the commit so a rolled-back payment sends nothing.
- Delivery is a pluggable `ChallengeSender`. The default `LogSender` writes the code to the
  server log and is for development only. `STEP_UP_DEV_ECHO=true` returns the code in the
  creating response so the local frontend is usable; it is off by default.
- Expiry is applied when the customer answers and by `scripts/expire_challenges.py`, which
  must be scheduled for abandoned challenges to settle.
- No step-up outcome writes a `TransactionOutcome`: passing an OTP is not proof of
  legitimacy (SIM swap, phishing), and failing one is not proof of fraud.
- The Streamlit prompt was syntax-checked and its API call tested end to end; the page
  itself was not run, because Streamlit is not installed in the API's test environment.

**Acceptance**

- [x] A `STEP_UP` decision issues a challenge (OTP in development; pluggable for 3-D Secure)
- [x] Passing the challenge sets `resolved_decision = ALLOW`; failing or expiring sets `REJECT`
- [x] Challenge expiry is configurable, and expired challenges are resolved, not left pending
- [x] Repeated failures open a `Case` for an analyst rather than silently rejecting
- [x] The engine's original `decision` is never rewritten — same rule as T-12
- [x] Frontend prompts the customer for the challenge
- [x] Tests for pass, fail, expiry and the escalation path

---

### Phase C — Prediction power

---

#### T-15 · Expand the transaction schema — ✅ DONE (`2a847d1`)

**Depends on:** T-08
**Blocks:** T-16, T-17
**Modify:** `app/models.py`, `app/schemas.py`, migration

Most high-value features cannot be built because the columns do not exist. This is a hard
prerequisite for the rest of Phase C.

**Add:** `merchant_id`, `merchant_category`, `currency`, `channel` (web/mobile/pos/atm),
`ip_address`, `card_token`, `external_txn_id`, `latitude`, `longitude`

**As built:**

- Validated at the API: `merchant_category` is a 4-digit ISO 18245 MCC, `currency` an
  ISO 4217 code, `channel` one of the four values, `ip_address` a real IPv4/IPv6 address,
  and coordinates are range-checked and accepted only as a pair.
- `card_token` refuses anything that looks like a raw card number (13-19 digits passing
  Luhn), so a primary account number cannot reach the database (PCI DSS).
- The T-14 idempotency fingerprint now hashes only the fields a client sends, so adding
  optional fields never changes the fingerprint of an old-shape request; a test pins the
  pre-T-15 hash.
- `populate_db.py` gives each generated customer consistent cards, favourite merchants,
  home network and local currency; fraud skews to high-risk MCCs, card-not-present channels
  and a few shared proxy ranges (useful signal for T-17). A test runs the seeder and
  requires every generated row to pass the API's own validation.
- The engine does not read any of these fields yet. T-16 does.
- Not done: the Streamlit form does not collect the new fields. `populate_db.py` also still
  scores with the pre-T-03 arithmetic and writes the old `MANUAL_CHECK` label, and writes no
  audit rows; T-18's `scripts/generate_data.py` supersedes it rather than patching it further.

**Acceptance**

- [x] Columns added with a migration, all nullable for backward compatibility
- [x] `populate_db.py` generates realistic values for every new field
- [x] `TransactionCreate` accepts them as optional
- [x] Existing tests still pass unchanged

---

#### T-16 · Feature expansion — ✅ DONE (`05d6902`), retraining moved to T-18

**Depends on:** T-15
**Modify:** `app/features.py`, `app/repositories/transaction_repo.py`

**As built:**

- **42 features** in eight families. `BASELINE_FEATURES` (the original five, which the rules
  read) lead `FEATURE_ORDER` unchanged, so every existing score is unchanged.
- **Split by decision (option 3):** the features are built and tested here; retraining on
  them and measuring importance moved to T-18, because the only training data - 20
  hand-written rows - describes five features. Importances measured on it would have been
  recorded and then not trusted.
- **One aggregate implementation.** Training no longer rebuilds history in Python: it calls
  the same repository queries as serving, evaluated as of each transaction's own timestamp.
  Scoring and storage now share one instant, and a test requires retraining to reproduce
  every audited feature vector exactly, field for field.
- **Two latent training bugs fixed by that move** (both since T-05): training built history
  from *labelled* rows only while serving counted all rows, and it counted device sharing
  across the whole dataset, future users included. Regression tests pin both.
- **A third caught by the parity test:** pandas 3 reports a missing string as `NaN`, which
  passes `is not None`, so a transaction with no merchant trained as a first visit to one.
  `TransactionInput.from_request` now normalises every missing marker at the boundary.
- **Registry rule changed:** a model may use a *subset* of `FEATURE_ORDER`; serving hands it
  exactly its own columns. It is refused only if it needs a feature serving does not compute,
  or its manifest misdescribes its fitted columns. Without this the 5-feature model would
  have stopped the API from starting.
- **Four statements per decision**, whatever the history size (was three).
- **Missing data:** cold-start customers are never penalised (shares read 1.0, first-time
  flags 0); unknowable values - no previous transaction, no coordinates - are `-1`, never 0.
  Impossible travel = over 900 km/h across more than 100 km. `is_night` is UTC, since
  customer time zones are unknown; `typical_hour_share` is the per-customer timing signal.
- Added `users.created_at` for account age (migration `009` backfills from each account's
  first transaction). PostgreSQL sessions are pinned to UTC so hour-of-day matches SQLite -
  set, but untested here, as no PostgreSQL is available to the test suite.
- `amount_population_percentile` counts over every prior transaction: bounded in statements,
  but its cost grows with total volume. A precomputed quantile sketch is the fix at scale.

Target: 5 features to 40+. The engine's ceiling is set by features, not by the model.

| Family | Features |
|---|---|
| Multi-window velocity | count and sum over 1m / 5m / 1h / 24h / 7d / 30d |
| Geo-velocity | distance from previous location, **implied travel speed km/h**, distinct locations per 24h |
| Temporal | hour of day, day of week, is_night, seconds since previous transaction, account age days |
| Amount shape | z-score vs user, percentile vs population, round-number flag, ratio to lifetime max |
| Device | device age days, devices per user, users per device, first-use flag |
| Merchant | merchant risk rate, first time at merchant, category deviation from user norm |
| Behavioural | deviation from the user's typical hour, location, amount band and category |

**Acceptance**

- [x] All features flow through `compute_features` — no exceptions (T-01 invariant preserved)
- [x] Implied travel speed implemented; a documented threshold flags physically impossible journeys
- [x] `FEATURE_ORDER` updated — 42 features
- [ ] ~~The model retrained against it~~ **→ moved to T-18** (no training data carries these features yet)
- [x] Aggregates still fetched in a bounded number of queries
- [x] Every feature unit-tested, including null and cold-start handling
- [ ] ~~Feature importance re-measured and recorded in the manifest~~ **→ moved to T-18**

---

#### T-17 · Graph features

**Depends on:** T-15
**Fixes:** A18
**Create:** `app/graph/builder.py`, `app/graph/features.py`

The most technically impressive addition available, and `networkx` is already a declared,
completely unused dependency.

**Contract**

```python
def build_entity_graph(db: Session, window_days: int = 30) -> nx.Graph:
    """Bipartite-ish graph over users, devices, IPs, cards and merchants.
    Rebuilt on a schedule, cached - NOT constructed per request."""

@dataclass(frozen=True)
class GraphFeatures:
    component_size: int          # size of the connected component this user sits in
    device_degree: int           # distinct users on this device (1-hop)
    two_hop_user_count: int      # users reachable in two hops
    shared_card_count: int
    component_fraud_rate: float  # confirmed fraud rate within the component
    is_in_dense_cluster: bool    # community detection flag
```

**Acceptance**

- [ ] Graph built as a scheduled batch job, never inside the request path
- [ ] Graph features served from a cache with a documented staleness bound
- [ ] Features merged into `FeatureVector` via `compute_features`
- [ ] Lift over the non-graph baseline measured and recorded
- [ ] Tests on a synthetic fraud ring: a planted ring must be detected

---

#### T-18 · Realistic training data — ✅ DONE (`0678d1a`)

**Depends on:** T-16
**Fixes:** A9
**Rewrite:** `app/ML/train_model.py`
**Create:** `scripts/generate_data.py`

The current 20 hand-written rows are perfectly separable on amount alone, which is why the
forest collapsed to depth-1 stumps.

**As built:**

- `scripts/generate_data.py`: 50,000 transactions for 1,000 customers over a year, 0.70%
  fraud, deterministic under a seed, in about 10 seconds. It writes only to a database that
  holds nothing but synthetic data (`data/synthetic.db` by default, git-ignored), marks every
  account `@synthetic.invalid`, and labels every row with a `SYNTHETIC` outcome, so
  synthetic labels cannot reach a real database.
- **The first version was too easy: AUC 1.000**, every permutation importance zero. Amounts
  overlapped, but every attack left a fingerprint no customer ever produced. Hard cases fixed
  it, in both directions: legitimate phone upgrades, borrowed devices, VPN purchases that
  geolocate abroad (so impossible travel fires on real customers), 0.99 app purchases,
  remittance wires and trips to attacker cities; and fraud on the victim's own device, from
  local attackers, in the daytime. Some takeovers follow a genuine purchase at home by
  minutes, which is what makes impossible travel appear at all.
- Training runs through `scripts/retrain.py` — now callable on any database — so the shipped
  model was produced by the same point-in-time feature path, chronological split and
  champion/challenger gate as any future retrain. `python -m app.ML.train_model` does the
  whole run: **under 5 minutes** for 50,000 rows, 259 s of it feature building.
- Getting there took two optimisations: statements are built once and reused with bound
  parameters (Python was rebuilding a 35-expression query per row, 4x the cost of running
  it), and the population amount percentile became a **daily quantile snapshot** — ranked
  against everything before the start of the UTC day, computed once per day and cached —
  instead of an exact count over every prior row, which made retraining quadratic. First
  estimate before either: 39 minutes.
- The challenger is a class-balanced `RandomForestClassifier` (150 trees, depth <= 14),
  a 2.3 MB artifact. The 20-row baseline stays registered, inactive, for comparison.
- **Result on the held-out window** (the last 10,000 rows by time, 72 fraud):

  | | Baseline, 5 features / 20 rows | Shipped, 42 features / 40k rows |
  |---|---|---|
  | AUC | 0.633 | **0.990** |
  | PR-AUC | 0.043 | **0.871** |
  | Recall at 1% FPR | 13.9% | **88.9%** |

  Top permutation importances: `is_card_not_present`, `is_first_time_merchant`,
  `seconds_since_previous`, `km_from_previous`, `location_share`, `device_age_days`.
- **These numbers describe synthetic data with patterns this project planted.** They show the
  pipeline and features work; they say nothing yet about real fraud. Real performance is
  unknown until analyst and chargeback labels accumulate (T-12, chargeback ingestion).
- **Found by the new model: amount monotonicity no longer holds strictly.** The forest's
  probability wobbles between trees, so raising an amount from 301 to 1,000 lowered one score
  by 0.0001. The rule score is still exactly monotone and the full score stays within 0.01;
  the strict test is kept as `xfail(strict=True)` and moved to T-19, whose LightGBM supports
  monotone constraints.
- One API test relied on the old model rating any large amount as fraud. It now fires all
  five rules, so it tests the policy rather than a model's opinion.

**Acceptance**

- [x] At least 50,000 generated transactions
- [x] Fraud rate between 0.5% and 1%, not 50%
- [x] Fraud and legitimate amount distributions **overlap** — separability must not be trivial
- [x] Realistic temporal structure: diurnal pattern, weekday and weekend variation
- [x] Planted attack patterns: card testing, account takeover, a device-sharing ring
- [x] Trained trees reach a meaningful depth, not 1 — assert this in a test
- [x] ~~Or, alternatively, document ingestion of IEEE-CIS / PaySim instead~~ not needed: generated data chosen
- [x] *(from T-16)* The model is retrained on the full `FEATURE_ORDER`, through
      `scripts/retrain.py`'s point-in-time feature path
- [x] *(from T-16)* Feature importance measured and recorded in the manifest
- [x] Generated data populates every T-15 field, so all 42 features carry signal
- [x] Feature building is fast enough for 50k rows: `build_features` issues four queries per
      row, which is correct but slow at that size - batch it or document the runtime

---

#### T-19 · LightGBM with calibration — ✅ DONE (`46ea1df`), challenger not promoted

**Depends on:** T-18
**Fixes:** A9, A10
**Modify:** `app/ML/train_model.py`, `scripts/retrain.py`, `app/ML/models.py`

**As built:**

- Both training paths now train a LightGBM booster, `scale_pos_weight` from the class ratio,
  constrained to be non-decreasing in the five amount features. `is_round_amount` and
  `amount_band_share` are left out of the model (40 of 42 features): they do not rise with
  the amount, so no model using them can be monotone in it. Serving still computes both.
- **Calibration** fits on the latest 20% of the training window, never on training or test
  rows. `cv="prefit"` is gone from scikit-learn 1.8, so it wraps a `FrozenEstimator`.
  **Isotonic only with 1,000+ fraud cases in that slice, Platt scaling below**, decided by
  rule before evaluation. With 15 fraud cases isotonic collapsed every probability onto three
  values, looked perfectly calibrated, and threw away the ranking; the spec's contract said
  isotonic, and this is the reason it is not always used.
- **The promotion gate now decides on PR-AUC**, and refuses a model that loses more than
  0.01 ROC AUC. The AUC-only gate promoted a challenger in trial that was +0.0016 AUC but
  -0.031 PR-AUC and -5.9 points of recall at 1% FPR - worse where rare fraud is concerned.
- **Model selection on the training window only:** early stopping plus three capacities
  (the spec's `num_leaves=31`, and two more regularised), chosen by validation PR-AUC. The
  spec's configuration alone lost clearly (PR-AUC 0.850 vs 0.871): 500 trees on 185
  positives weighted ~145x memorised them. The test window was then used once more.
- **`app/ML/scorer.py`** scores straight from `FeatureVector` to probability. Through
  scikit-learn's API one transaction cost 6.2 ms p50 / 20.8 ms p99, almost all DataFrame
  validation; the model itself takes 0.27 ms. The fast path is held to within 1e-12 of
  `predict_proba` by test - which caught a real bug: `CalibratedClassifierCV` prefers
  LightGBM's `decision_function`, so its calibrator was fitted on log-odds, and feeding it
  probabilities returned nonsense (0.308 where the model said 0.001).
- **Result, 50k synthetic, held-out window of 10,000 rows (72 fraud):**

  | | LightGBM, selected (7 leaves, 57 trees, Platt) | Forest (T-18, serving) |
  |---|---|---|
  | PR-AUC — decides promotion | 0.8678 | **0.8712** |
  | ROC AUC | **0.9929** | 0.9895 |
  | Recall at 1% FPR | 88.89% | 88.89% |
  | Expected calibration error | **0.0010** | 0.0036 |
  | Brier score | 0.00193 | 0.00190 |
  | Inference p50 / p99 | **0.36 / 0.54 ms** | — |

  **The gate kept the forest.** 0.0034 PR-AUC with 72 positives is about one transaction
  ranked differently: a tie, not a loss. LightGBM wins on everything else, including the
  monotonicity guarantee. Changing the gate, or re-running until it wins, after seeing this
  would be tuning to the test window, so neither was done. Whether to promote it anyway is
  a decision for a person, and convention 20 says how: record the reason with the manifest.
- LightGBM needs the OpenMP runtime; the Docker image now installs `libgomp1`, without which
  the API would build and then fail at import. Not verified here - no Docker available.

**Contract**

```python
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=<neg/pos ratio>,   # imbalance handling
    random_state=42,
)
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
```

**Acceptance**

- [x] LightGBM replaces RandomForest in both training paths
- [x] `scale_pos_weight` set from the actual class ratio
- [x] Probabilities calibrated; a reliability curve is produced and stored
- [x] Brier score recorded in the manifest alongside AUC
- [x] Inference latency measured and under 10 ms
- [x] AUC and recall-at-1%-FPR compared against the RandomForest baseline, both logged
- [ ] Calibration verified: among transactions scored ~0.7, roughly 70% are actually fraud —
      **not verifiable on this data:** only 5 held-out transactions scored 0.6-0.8 (observed
      50% and 67%). Overall calibration error is 0.0010; the ~0.7 claim needs more fraud
- [x] *(from T-18)* Monotone constraints on the amount features — built and tested on every
      LightGBM challenger
- [ ] Strict amount monotonicity **in serving**: the `xfail` stays, because the forest still
      serves. It clears the day a LightGBM model is promoted
- [ ] Beat the shipped forest (AUC 0.990, PR-AUC 0.871 on the synthetic held-out window)
      through the same champion/challenger gate — **not met.** Effectively a tie; see below

---

#### T-20 · Unsupervised anomaly layer

**Depends on:** T-16
**Create:** `app/ML/anomaly.py`

Catches novel attacks that have no labels — the class of fraud a supervised model is
structurally blind to.

**Acceptance**

- [ ] `IsolationForest` trained on legitimate traffic only
- [ ] Anomaly score exposed as a feature to the supervised model **and** as a standalone signal
- [ ] A high anomaly score with a low supervised score routes to REVIEW, not ALLOW
- [ ] Test: a synthetic novel attack pattern absent from training is flagged

---

### Phase D — Production readiness

Lower risk, largely mechanical. Conspicuous by their absence in review.

| Task | Scope | Fixes |
|---|---|---|
| T-21 | Alembic — replace `create_all()` at `app/main.py:7`, baseline the existing schema | §5.6 |
| T-22 | GitHub Actions — ruff, mypy, pytest with a coverage gate, docker build | §5.6 |
| T-23 | Structured JSON logging with request and correlation IDs | §5.6 |
| T-24 | `/health` and `/ready` endpoints; Prometheus metrics per §10.5 | §5.6 |
| T-25 | CORS, rate limiting on `/auth/login`, security headers, global exception handler | §5.6 |
| T-26 | Docker: multi-stage build, non-root user, `.dockerignore`, install from `uv.lock` | §5.6 |
| T-27 | `pydantic-settings`, `/v1` API prefix, refresh tokens with revocation | §5.6 |
| T-28 | Load tests against `POST /transactions/`; contract tests against the OpenAPI schema | §5.7 |

### Phase E — Operational maturity

| Task | Scope |
|---|---|
| T-29 | Drift detection — PSI and KS on feature distributions, alerting on threshold breach |
| T-30 | SHAP attributions computed per decision and stored on the audit row |
| T-31 | Shadow mode — challenger scores live traffic without acting; compare, then ramp by canary percentage |
| T-32 | Backtesting harness and MLflow experiment tracking |
| T-33 | Redis velocity counters, async scoring path, read replica for analytics |

---

## 13. Dependency graph

Start anywhere with no unmet dependency. T-01 and T-05 are the two roots.

```
✅ T-01 features.py ──┬── ✅ T-02 SQL aggregates
                      ├── ✅ T-03 scoring ──┬── ✅ T-04 policy ──┐
                      │                     ├── ✅ T-09 tests    │
                      │                     └── ✅ T-13 registry │
                      └── ✅ T-08 fix bundle ── ✅ T-15 schema ────┼── ✅ T-16 features+
                                                                  │   └── ✅ T-18 data ── ✅ T-19 LightGBM
                                                                  │   └── T-20 anomaly
                                                                  └── T-17 graph
✅ T-05 outcomes ──┬── ✅ T-06 chronological split
                   ├── ✅ T-07 monitor fix
                   └── ✅ T-12 case queue
✅ T-10 RBAC ─────────┘
✅ T-03 + ✅ T-04 ── ✅ T-11 audit trail
✅ T-14 idempotency   (independent)
✅ T-04 ── ✅ T-14b step-up auth   (added during T-12)
   Phase D            (independent, can run in parallel throughout)
```

**Unblocked right now:** T-17, T-20, and all of Phase D. **Phase B is complete.**

**Phase B order, as executed:** T-10 → T-12 → T-11 → T-13 → T-14 → T-14b.

**Suggested order for Phase C:** ~~T-15~~ → ~~T-16~~ → ~~T-18~~ → ~~T-19~~ → T-17 → T-20. T-18 (realistic data) is what makes every later model comparison meaningful. That finishes the critical
path (T-12 and T-11 are its last two links) before the supporting work.

**Critical path to a credible system:** T-01 → T-03 → T-04 → T-05 → T-12 → T-11. ✅ **Complete.**
That sequence alone converts a scoring function into an auditable fraud platform with a
working label feedback loop.
