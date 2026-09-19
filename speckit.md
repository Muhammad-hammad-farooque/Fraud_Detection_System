# SpecKit — Fraud Detection System

| Field | Value |
|---|---|
| **Project** | Fraud Detection System |
| **Version** | 0.1.0 |
| **Status** | Working prototype — not production ready. Phase A complete; Phase B in progress. |
| **Last reviewed** | 2026-09-19 |
| **Progress** | 11 / 34 tasks · 14 / 18 defects fixed, A13 half-fixed — see §1.4 |
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
| Fraud analyst | Review the manual-check queue, confirm fraud, tune rules | **Yes, via the API** (T-12) — no analyst UI yet, and rule tuning waits on T-13 |
| Data scientist | Retrain, evaluate, monitor drift | Partially (scripts only) |
| Auditor / regulator | Reconstruct why any decision was made | **No — not implemented** |

### 1.4 Progress

Phase A is complete and Phase B is under way. Each task below is one commit, with its defects struck through in §4.2
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
| T-08 · Correctness fix bundle | `2fedbef` | A10, A11, A12, A14 |
| T-09 · Test infrastructure and rule coverage | `5cff5f0` | A17 |
| T-10 · Role-based access control | `b1e3e38` | prerequisite for T-12 |
| T-12 · Analyst case queue | `pending` | A13 (REVIEW / MANUAL_REVIEW) |

**Suite:** 324 tests, 99% coverage of `app/`, under 30 seconds.

**Still open:** A9 (weak training data, T-18), A13's `STEP_UP` half (T-14b, added during
T-12), A15 (no idempotency, T-14), A18 (`networkx` unused — kept deliberately, T-17 uses it).

**The label loop is closed.** Resolving a case writes an ANALYST `TransactionOutcome`,
which is exactly what `retrain.py` and `monitor.py` read. Retraining still needs at least
20 mature labels (90 days old) before it will train, so the loop turns slowly by design.

**Next:** T-11 (audit trail) completes the critical path. T-13, T-14 and T-14b are also
unblocked.

**Carried debt, not yet a task:** schema changes land while `create_all()` is still the only
deployment path, so `migrations/` holds hand-written DDL for databases created earlier —
`001_phase_a_schema.sql`, `002_user_roles.sql` and `003_cases.sql` so far. `003` also opens a
case for every transaction already sitting in REVIEW, so none stay dead ends. T-21 must baseline every file there
when it brings in Alembic.

**Bootstrapping an admin:** the role endpoint needs an admin to call it, so the first one is
created out of band with `python -m scripts.set_role someone@bank.com ADMIN`.

---

## 2. Architecture

### 2.1 Request flow

```
Streamlit UI  --HTTP-->  FastAPI  -->  Router  -->  Service layer  -->  SQLAlchemy  -->  PostgreSQL
(frontend/)              (app/main)   (routers/)   (services/,           (models.py)
                                                    fraud_detection.py)
                                                          |
                                                          +--> ML model (app/ML/model.pkl)
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
| Inference | `app/ML/models.py` | Loads `model.pkl`, takes a `FeatureVector`, returns `(prediction, probability)` |
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
               risk_score, risk_level, decision, policy_version, created_at
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

Five weighted rules blended with a `RandomForestClassifier` probability (§2.2). Features are
computed once in `app/features.py` and passed to the model as a named DataFrame, so the
training and serving paths cannot drift apart:

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
drops labels younger than 90 days, rebuilds features through `app/features.py`, splits
chronologically, trains a challenger, compares AUC-ROC against the deployed model, and
replaces `model.pkl` only if the challenger wins. Logs to `logs/retrain.log`.

**`scripts/monitor.py`** — performance monitoring over a rolling window. Reports labelled
coverage, confusion matrix, precision, recall, F1, false positive rate, the business metrics
from §10.5, and volume by decision and risk level. Detection metrics are suppressed below 20
confirmed outcomes. Warns if recall drops below 80% or FPR exceeds 10%.
Logs to `logs/monitor_YYYY-MM-DD.log`.

Neither script can do anything useful until something writes `TransactionOutcome` rows,
which is T-12.

### 3.5 Testing

324 tests across 14 modules, run against in-memory SQLite so no PostgreSQL is needed. `conftest.py`
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
| A9 | **Model has no real signal.** Trained on 20 hand-written rows that are perfectly separable on amount alone (legit <= 400, fraud >= 5000), fit on 100% of the data with no holdout. | `app/ML/train_model.py` |
| A10 | **Uncalibrated probabilities used arithmetically.** RandomForest `predict_proba` is poorly calibrated, yet is multiplied by 0.3 and summed into the score as if it were a true probability. | `app/fraud_detection.py` |
| ~~A11~~ | ~~**Feature-name mismatch.** A bare numpy array is passed to a model fitted on a DataFrame — emits a warning and relies silently on positional order.~~ **Resolved by T-08.** | `app/ML/models.py:15` |
| ~~A12~~ | ~~**Naive datetime columns.** `Column(DateTime)` without `timezone=True`, forcing scattered `.replace(tzinfo=utc)` patches at every use site.~~ **Resolved by T-08.** | `app/models.py` |
| A13 | ~~**Dead decision states.** `REVIEW` and `MANUAL_REVIEW` are terminal — no code path can resolve them.~~ **Resolved by T-12** via the case queue. **Still open for `STEP_UP`** (since T-04): no task builds the step-up authentication flow that would complete or abandon it — see T-14b. | system-wide |
| ~~A14~~ | ~~**Claim amount unvalidated server-side.** The backend accepts any amount regardless of the transaction's value; only the frontend enforces a maximum.~~ **Resolved by T-08.** | `app/routers/claims.py` |
| A15 | **No idempotency.** A retried POST creates a duplicate transaction and falsely inflates the velocity rule. | `app/routers/transactions.py` |
| ~~A16~~ | ~~**No cold-start handling.** A user's first transaction always has empty known locations, so `is_new_location` fires for every new customer.~~ **Resolved by T-01.** | `app/fraud_detection.py` |
| ~~A17~~ | ~~**Test DB is file-based, not in-memory** as the docstring and README both claim. File-based SQLite can leak state between runs.~~ **Resolved by T-09.** | `tests/conftest.py` |
| A18 | **`networkx` declared but never imported.** Dead dependency. **Kept deliberately: T-17 uses it for graph features.** | `pyproject.toml` |

---

## 5. What to add for best performance

### 5.1 Prediction accuracy

The engine's ceiling is set by its features, not its model. Five features is roughly 2% of
what a production fraud engine uses.

**Prerequisite — expand the transaction schema.** Most high-value features cannot be built
because the columns do not exist. Add: `merchant_id`, `merchant_category`, `currency`,
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
   faster inference.
3. **Calibrate the probability** with `CalibratedClassifierCV`, since the score is consumed numerically.
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
| **Model registry** | `model.pkl` has no version, training date, feature schema or metrics. Stamp `model_version` on every decision. |
| **Config-driven thresholds** | `5000`, `0.4`, `0.3/0.7` are hardcoded. Risk teams retune weekly; that cannot require a deploy. |
| **Blocklists / allowlists** | Known-bad devices and locations; trusted-customer bypass |
| **Chargeback ingestion** | The authoritative fraud label source in payments |
| **Idempotency keys** | Payment systems retry (defect A15) |
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

### Phase 2 — Make it a fraud *system*

11. RBAC with `customer` / `analyst` / `admin`
12. Case management queue for manual review — fixes A13
13. Immutable decision audit trail
14. Model registry and `model_version` stamped on every decision
15. Config-driven rule thresholds
16. Idempotency keys — fixes A15

### Phase 3 — Prediction power

17. Expand the transaction schema (merchant, currency, channel, IP, card token)
18. Multi-window velocity and geo-velocity features
19. Temporal, amount-shape, device and behavioural-deviation features
20. Graph features with `networkx` — fixes A18
21. Realistic imbalanced training data — fixes A9
22. LightGBM with calibration — fixes A10
23. Unsupervised anomaly layer
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
| Features in the model | 5 | 60+ |
| Training samples | 20 synthetic | 50k+ realistic, imbalanced |
| Class balance | 50 / 50 | ~0.5% fraud |
| Scoring latency p99 | unmeasured, but O(1) in history since T-02 | < 100 ms, O(1) |
| Recall at fixed 1% FPR | unmeasured | > 70% |
| Label source | `TransactionOutcome` only, but nothing writes to it until T-12 | analyst confirmations + chargebacks |
| Decision auditability | `ScoreBreakdown` carries the trace, nothing persists it until T-11 | full feature vector and rule trace per decision |
| Manual review resolution | impossible | analyst queue with SLA |
| Deployment | `create_all()` on boot | Alembic migrations via CI |

---

## 9. Highest-leverage items

If only a handful of changes are made, these carry the most signal:

1. ~~**`features.py`** — a single computation path, eliminating train/serve skew~~ **Done, T-01**
2. ~~**`TransactionOutcome`** — breaks the circular labelling that makes the ML loop meaningless~~ **Done, T-05**
3. ~~**Fix score saturation** — the ML contribution is discarded when it matters most~~ **Done, T-03**
4. ~~**SQL aggregates instead of `.all()`** — removes the hardest scalability limit~~ **Done, T-02**
5. **Decision audit trail** (T-11) — the clearest signal of regulated-fintech experience
6. **Graph features with `networkx`** (T-17) — the most technically impressive addition available

The two that remain are the highest-leverage work left, together with **T-12**, which turns
the table T-05 created into an actual supply of labels.

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
| GBDT primary scorer | Degenerate RandomForest depth-1 stumps | T-19 |
| Feature store, one code path | ~~Features computed in three places~~ one path, `app/features.py` | ~~T-01~~ ✅ |
| Streaming velocity counters | ~~In-Python scan of full history~~ bounded SQL; Redis still T-33 | ~~T-02~~ ✅ |
| Policy layer separate from model | ~~Thresholds fused into the scorer~~ `app/policy.py` | ~~T-04~~ ✅ |
| Calibrated probability | Raw uncalibrated `predict_proba` | T-19 |
| Step-up auth as a third action | ~~Allow / review / reject only~~ STEP_UP shipped | ~~T-04~~ ✅ |
| Chargeback and analyst labels | ~~Model's own output used as label~~ `TransactionOutcome`, unwritten until T-12 | ~~T-05~~ ✅ |
| Full decision audit log | Nothing logged | T-11 |
| Graph and ring features | 1-hop device degree only | T-17 |
| Multi-window velocity | Single 120-second window | T-16 |

---

## 11. How to work this document

Conventions for anyone — human or agent — executing the build guide in §12.

### 11.1 Task format

Every task states its dependencies, the defect it fixes, the files it touches, a concrete code
contract, and a checklist of acceptance criteria. A task is done only when every box is ticked.

### 11.2 Working rules

1. **One task per commit.** Never combine two task IDs in one change.
2. **Respect `Depends on`.** Tasks are ordered by dependency; starting out of order will fail.
3. **Run `pytest` before marking a task done.** The suite must stay green — currently 324 tests.
4. **Add tests in the same commit as the code.** A task with no new test is not complete.
5. **Do not change behaviour not named in the task.** Refactors that touch scoring must keep
   existing test expectations passing, or must update them explicitly and say why.
6. **Update this file.** Tick the acceptance boxes and mark the defect resolved in §4.2.
7. **Never delete a defect row** in §4.2 — strike it through and note the fixing task.

### 11.2b Conventions settled during Phase A

Decisions made while executing Phase A that later tasks must not silently undo.

1. **`compute_features` stays pure.** No database, no clock, no I/O. Aggregates are built by
   the caller: `transaction_repo.get_user_aggregates` when serving, `aggregates_from_history`
   when training. T-16 adds features to the same function, never around it.
2. **Rule weights are normalised, never clamped.** `final_score` is in [0, 1] because
   `W_RULES + W_MODEL == 1.0`, not because of a `min()`. Adding a rule changes
   `TOTAL_RULE_WEIGHT` and therefore rescales every existing score — retune the policy
   thresholds in the same task, and say so.
3. **Rules are named.** Every rule carries a stable `id` such as `R4_FLAGGED_DEVICE`, because
   the audit trail in T-11 stores those ids. Renaming one is a schema change to the audit log.
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
    source. Future label sources (chargeback ingestion, T-14b) follow the same rule.

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

#### T-11 · Immutable decision audit trail

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

**Acceptance**

- [ ] One audit row written for every scored transaction, in the same database transaction
- [ ] Append-only — no update or delete path exists anywhere in the codebase
- [ ] A stored audit row is sufficient to replay the decision exactly
- [ ] `GET /v1/analyst/transactions/{id}/audit` exposes it to analysts only
- [ ] Test asserting a replayed audit row reproduces the original score bit-for-bit

---

#### T-12 · Analyst case queue — ✅ DONE (`pending`)

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

#### T-13 · Model registry and config-driven rules

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

**Acceptance**

- [ ] Every saved model has a manifest written beside it
- [ ] Loading refuses to proceed on a feature-order mismatch
- [ ] `model_version` stamped on every decision and audit row
- [ ] Rule weights and policy thresholds load from `config/rules.yaml`
- [ ] Changing a threshold requires no code change
- [ ] Config schema validated on startup — malformed config fails fast, not at first request

---

#### T-14 · Idempotency

**Depends on:** none
**Fixes:** A15
**Modify:** `app/routers/transactions.py`, `app/schemas.py`

**Acceptance**

- [ ] `Idempotency-Key` header accepted on `POST /transactions/`
- [ ] A repeat key returns the original response without rescoring
- [ ] Unique constraint on `(user_id, idempotency_key)`
- [ ] Retries no longer inflate the velocity feature
- [ ] Test: the same key twice yields one transaction and identical responses

---

#### T-14b · Step-up authentication flow

**Depends on:** T-04
**Fixes:** A13 (the `STEP_UP` half)
**Added:** during T-12, when it became clear no task completes a `STEP_UP` decision

T-04 made `STEP_UP` a real policy action, but nothing acts on it: the customer is never
challenged, and the transaction sits in `STEP_UP` forever. Industry treats step-up as the
action that turns would-be false declines into completed payments (§10.3, P4), so it needs a
lifecycle of its own rather than being folded into the analyst queue.

**Acceptance**

- [ ] A `STEP_UP` decision issues a challenge (OTP in development; pluggable for 3-D Secure)
- [ ] Passing the challenge sets `resolved_decision = ALLOW`; failing or expiring sets `REJECT`
- [ ] Challenge expiry is configurable, and expired challenges are resolved, not left pending
- [ ] Repeated failures open a `Case` for an analyst rather than silently rejecting
- [ ] The engine's original `decision` is never rewritten — same rule as T-12
- [ ] Frontend prompts the customer for the challenge
- [ ] Tests for pass, fail, expiry and the escalation path

---

### Phase C — Prediction power

---

#### T-15 · Expand the transaction schema

**Depends on:** T-08
**Blocks:** T-16, T-17
**Modify:** `app/models.py`, `app/schemas.py`, migration

Most high-value features cannot be built because the columns do not exist. This is a hard
prerequisite for the rest of Phase C.

**Add:** `merchant_id`, `merchant_category`, `currency`, `channel` (web/mobile/pos/atm),
`ip_address`, `card_token`, `external_txn_id`, `latitude`, `longitude`

**Acceptance**

- [ ] Columns added with a migration, all nullable for backward compatibility
- [ ] `populate_db.py` generates realistic values for every new field
- [ ] `TransactionCreate` accepts them as optional
- [ ] Existing tests still pass unchanged

---

#### T-16 · Feature expansion

**Depends on:** T-15
**Modify:** `app/features.py`, `app/repositories/transaction_repo.py`

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

- [ ] All features flow through `compute_features` — no exceptions (T-01 invariant preserved)
- [ ] Implied travel speed implemented; a documented threshold flags physically impossible journeys
- [ ] `FEATURE_ORDER` updated and the model retrained against it
- [ ] Aggregates still fetched in a bounded number of queries
- [ ] Every feature unit-tested, including null and cold-start handling
- [ ] Feature importance re-measured and recorded in the manifest

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

#### T-18 · Realistic training data

**Depends on:** T-16
**Fixes:** A9
**Rewrite:** `app/ML/train_model.py`
**Create:** `scripts/generate_data.py`

The current 20 hand-written rows are perfectly separable on amount alone, which is why the
forest collapsed to depth-1 stumps.

**Acceptance**

- [ ] At least 50,000 generated transactions
- [ ] Fraud rate between 0.5% and 1%, not 50%
- [ ] Fraud and legitimate amount distributions **overlap** — separability must not be trivial
- [ ] Realistic temporal structure: diurnal pattern, weekday and weekend variation
- [ ] Planted attack patterns: card testing, account takeover, a device-sharing ring
- [ ] Trained trees reach a meaningful depth, not 1 — assert this in a test
- [ ] Or, alternatively, document ingestion of IEEE-CIS / PaySim instead

---

#### T-19 · LightGBM with calibration

**Depends on:** T-18
**Fixes:** A9, A10
**Modify:** `app/ML/train_model.py`, `scripts/retrain.py`, `app/ML/models.py`

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

- [ ] LightGBM replaces RandomForest in both training paths
- [ ] `scale_pos_weight` set from the actual class ratio
- [ ] Probabilities calibrated; a reliability curve is produced and stored
- [ ] Brier score recorded in the manifest alongside AUC
- [ ] Inference latency measured and under 10 ms
- [ ] AUC and recall-at-1%-FPR compared against the RandomForest baseline, both logged
- [ ] Calibration verified: among transactions scored ~0.7, roughly 70% are actually fraud

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
                      │                     └──    T-13 registry │
                      └── ✅ T-08 fix bundle ──   T-15 schema ────┼── T-16 features+
                                                                  │   └── T-18 data ── T-19 LightGBM
                                                                  │   └── T-20 anomaly
                                                                  └── T-17 graph
✅ T-05 outcomes ──┬── ✅ T-06 chronological split
                   ├── ✅ T-07 monitor fix
                   └── ✅ T-12 case queue
✅ T-10 RBAC ─────────┘
✅ T-03 + ✅ T-04 ── T-11 audit trail
   T-14 idempotency   (independent)
✅ T-04 ── T-14b step-up auth   (added during T-12)
   Phase D            (independent, can run in parallel throughout)
```

**Unblocked right now:** T-11, T-13, T-14, T-14b, T-15.

**Suggested order for Phase B:** ~~T-10~~ → ~~T-12~~ → T-11 → T-13 → T-14 → T-14b. That finishes the critical
path (T-12 and T-11 are its last two links) before the supporting work.

**Critical path to a credible system:** T-01 → T-03 → T-04 → T-05 → T-12 → T-11.
That sequence alone converts a scoring function into an auditable fraud platform with a
working label feedback loop.
