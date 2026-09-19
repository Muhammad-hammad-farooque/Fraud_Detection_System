# Fraud Detection System

A production-ready fraud detection platform built with **FastAPI**, **SQLAlchemy**, **scikit-learn**, and a **Streamlit** frontend. Every transaction is scored in real time by combining rule-based heuristics with a machine learning model.

---

## Architecture

```
POST /transactions/
        │
        ▼
  Rule Engine (fraud_detection.py)
  ├── High amount threshold       (+0.4)
  ├── Amount deviation from avg   (+0.4)
  ├── New location for user       (+0.2)
  ├── Shared/flagged device       (+0.3)
  ├── Rapid transaction velocity  (+0.5)
  └── ML model probability boost  (+0.3 max)
        │
        ▼
  risk_score (0.0 → 1.0)
        │
        ├── < 0.3  →  LOW     →  ALLOW
        ├── < 0.7  →  MEDIUM  →  MANUAL_CHECK
        └── ≥ 0.7  →  HIGH    →  REJECT
```

---

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| API         | FastAPI                             |
| Frontend    | Streamlit                           |
| Database    | PostgreSQL + SQLAlchemy ORM         |
| ML Model    | scikit-learn RandomForestClassifier |
| Auth        | JWT (python-jose) + bcrypt          |
| Validation  | Pydantic v2                         |
| Containers  | Docker + Docker Compose             |
| Testing     | pytest + httpx                      |
| Packaging   | uv                                  |

---

## Project Structure

```
fraud-detection-system/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── models.py                # SQLAlchemy ORM models (User, Transaction, Claim)
│   ├── schemas.py               # Pydantic v2 request/response schemas
│   ├── database.py              # DB engine and session setup
│   ├── auth.py                  # JWT creation/decoding, bcrypt password hashing
│   ├── dependencies.py          # get_db and get_current_user dependencies
│   ├── fraud_detection.py       # Risk scoring engine (rule-based + ML)
│   ├── services/
│   │   └── fraud_services.py    # risk level, decision, device fraud, claim verification
│   ├── ML/
│   │   ├── train_model.py       # Model training script
│   │   ├── models.py            # predict_fraud() inference function
│   │   ├── registry.py          # Versioned models, each with a manifest
│   │   └── artifacts/           # <version>/model.pkl + manifest.json, and ACTIVE
│   └── routers/
│       ├── auth.py              # /auth endpoints (register, login, me)
│       ├── user_route.py        # /users endpoints
│       ├── transactions.py      # /transactions endpoints
│       └── claims.py            # /claims endpoints
├── frontend/
│   ├── app.py                   # Streamlit app entry point
│   ├── api.py                   # HTTP client wrapper for the FastAPI backend
│   └── pages/
│       ├── login.py             # Login / register page
│       ├── dashboard.py         # Overview dashboard
│       ├── transactions.py      # Submit and view transactions
│       └── claims.py            # File and track claims
├── tests/
│   ├── conftest.py              # Shared fixtures (in-memory SQLite + StaticPool, test client)
│   ├── test_auth.py             # Auth endpoint tests
│   ├── test_transactions.py     # Transaction endpoint + fraud rule tests
│   ├── test_claims.py           # Claims endpoint + verification logic tests
│   ├── test_fraud_services.py   # Unit tests for services and auth helpers
│   └── test_users.py            # User endpoint tests
├── scripts/
│   ├── retrain.py               # Retraining pipeline (champion/challenger)
│   └── monitor.py               # Daily performance monitor
├── logs/                        # Auto-generated log files
├── populate_db.py               # Seed database with dummy data
├── Dockerfile                   # Container image for the API
├── docker-compose.yml           # API + PostgreSQL services
├── pytest.ini                   # Test configuration
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Pip-compatible dependency list
└── .env.example                 # Environment variable template
```

---

## Setup

### Option A — Docker (recommended)

**1. Clone the repository**
```bash
git clone <repo-url>
cd fraud-detection-system
```

**2. Configure environment**
```bash
cp .env.example .env
# Edit .env and fill in SECRET_KEY and POSTGRES_PASSWORD
```

**3. Build and start all services**
```bash
docker compose up --build
```

API available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

---

### Option B — Local development

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd fraud-detection-system
uv sync
```

**2. Configure environment**
```bash
cp .env.example .env
# Edit .env with your PostgreSQL connection string and a strong SECRET_KEY
```

**3. Train the ML model**
```bash
python app/ML/train_model.py
```

**4. Start the API**
```bash
uvicorn app.main:app --reload
```

**5. Start the Streamlit frontend** (separate terminal)
```bash
streamlit run frontend/app.py
```

**6. (Optional) Seed the database**
```bash
python populate_db.py
```

API docs at: `http://localhost:8000/docs`
Frontend at: `http://localhost:8501`

---

## Environment Variables

Copy `.env.example` to `.env` and set all values:

| Variable                    | Description                              | Default  |
|-----------------------------|------------------------------------------|----------|
| `DATABASE_URL`              | PostgreSQL connection string             | required |
| `SECRET_KEY`                | Secret key for JWT signing               | required |
| `POSTGRES_PASSWORD`         | Password used by docker-compose          | required |
| `ALGORITHM`                 | JWT signing algorithm                    | `HS256`  |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token lifetime in minutes          | `30`     |

---

## API Endpoints

All endpoints except `/auth/register` and `/auth/login` require a Bearer token.

### Auth
| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| POST   | `/auth/register`  | Register a new user                |
| POST   | `/auth/login`     | Login and receive a JWT token      |
| GET    | `/auth/me`        | Get the currently logged-in user   |

### Users
| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/users/me`     | Get own profile                    |
| GET    | `/users/{id}`   | Get user by ID (own account only)  |

### Transactions
| Method | Endpoint                | Description                         |
|--------|-------------------------|-------------------------------------|
| POST   | `/transactions/`        | Submit a transaction for fraud check|
| GET    | `/transactions/`        | List all transactions for current user |
| GET    | `/transactions/{id}`    | Get a transaction by ID             |

### Claims
| Method | Endpoint        | Description                          |
|--------|-----------------|--------------------------------------|
| POST   | `/claims/`      | File a dispute claim on a transaction|
| GET    | `/claims/`      | List all claims for current user     |
| GET    | `/claims/{id}`  | Get a claim by ID                    |

---

## ML Model

**Algorithm:** RandomForestClassifier (100 estimators)

**Features used at inference time:**

| Feature              | Description                                     |
|----------------------|-------------------------------------------------|
| `amount`             | Transaction amount                              |
| `amount_deviation`   | Ratio of amount to user's historical average    |
| `is_new_location`    | 1 if location never seen for this user before   |
| `is_flagged_device`  | 1 if device used by 3+ distinct users           |
| `velocity`           | Transactions by this user in the last 2 minutes |

The model outputs a **fraud probability (0.0–1.0)** that contributes up to +0.3 to the overall risk score.

---

## Claim Verification Logic

Claims go through a 3-step verification engine:

```
Step 1 — Serial claimer check
  User has filed > 3 previous claims       →  REJECTED

Step 2 — Age rule
  Transaction is older than 90 days        →  REJECTED

Step 3 — Pattern matching
  Non-fraud transaction + first ever claim →  APPROVED
  Everything else                          →  MANUAL_REVIEW
```

---

## Retraining Pipeline

```bash
python -m scripts.retrain
```

1. Loads all transactions with confirmed outcomes from approved/rejected claims
2. Recomputes features preserving temporal order (no data leakage)
3. Trains a new RandomForestClassifier on an 80/20 train/test split
4. Compares AUC-ROC of new model vs current deployed model
5. Registers and activates the new model only if it wins (champion/challenger)

All decisions are logged to `logs/retrain.log`.

---

## Performance Monitoring

```bash
# Monitor last 24 hours
python -m scripts.monitor

# Monitor last 7 days
python -m scripts.monitor --days 7
```

Reports are logged to `logs/monitor_YYYY-MM-DD.log` and include:
- Confusion matrix (TP / FP / TN / FN)
- Precision, Recall, F1 score
- False positive rate
- Volume stats by decision and risk level
- Health warnings if recall drops below 80% or FPR exceeds 10%

---

## Testing

```bash
# Install test dependencies
pip install pytest httpx pytest-cov

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific module
pytest tests/test_auth.py -v
```

Tests use a genuinely in-memory SQLite database (StaticPool, no file on disk) — no PostgreSQL
required to run the suite, and no `test.db` left behind.

Coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

**76 tests across 5 modules — all passing.**

| Module                      | Coverage                                      |
|-----------------------------|-----------------------------------------------|
| `test_auth.py`              | Register, login, JWT, /me                     |
| `test_transactions.py`      | CRUD, fraud rules, cross-user isolation       |
| `test_claims.py`            | CRUD, claim verification, cross-user isolation|
| `test_fraud_services.py`    | Unit tests for all services and auth helpers  |
| `test_users.py`             | Profile endpoints, access control             |
