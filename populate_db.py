import random
import sys
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, Base
from app import models
from app.auth import hash_password

Base.metadata.create_all(bind=engine)

# ── Configuration ────────────────────────────────────────────────
TOTAL_USERS        = 200
TOTAL_TRANSACTIONS = 10_000
FRAUD_RATE         = 0.04   # 4% fraud — realistic for banking
CLAIM_RATE         = 0.60   # 60% of fraud transactions get a claim filed

random.seed(42)

# ── Reference Data ───────────────────────────────────────────────
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fatima", "George",
    "Hannah", "Ibrahim", "Julia", "Kevin", "Layla", "Muhammad", "Nina",
    "Omar", "Priya", "Quinn", "Rachel", "Samuel", "Tara", "Usman",
    "Victoria", "William", "Xena", "Yasir", "Zara", "Ahmed", "Bella",
    "Carlos", "Dina", "Elena", "Faisal", "Grace", "Hassan", "Iris",
    "James", "Kiran", "Liam", "Maria", "Nathan"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"
]

LOCATIONS = [
    "New York", "London", "Paris", "Dubai", "Tokyo", "Sydney",
    "Toronto", "Berlin", "Mumbai", "Singapore", "Los Angeles",
    "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio",
    "San Diego", "Dallas", "San Jose", "Austin", "Karachi", "Lahore",
    "Istanbul", "Moscow", "Beijing", "Shanghai", "Seoul", "Bangkok",
    "Jakarta", "Cairo"
]

CLAIM_REASONS = [
    "I did not authorize this transaction.",
    "This transaction was not made by me.",
    "My card was stolen and used without my knowledge.",
    "I never made this purchase.",
    "Unauthorized transaction on my account.",
    "I was abroad and this transaction happened locally without me.",
    "My account was hacked and this charge is fraudulent.",
    "I never received the goods/services for this transaction.",
    "Duplicate charge — I only authorized one payment.",
    "This transaction occurred after I reported my card lost.",
]

# ── Helpers ──────────────────────────────────────────────────────
def random_email(first: str, last: str, idx: int) -> str:
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "bank.com"]
    return f"{first.lower()}.{last.lower()}{idx}@{random.choice(domains)}"

def random_datetime(days_back: int = 365) -> datetime:
    now = datetime.now(timezone.utc)
    delta = timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    return now - delta

def get_risk_level(score: float) -> str:
    if score < 0.3:
        return "LOW"
    elif score < 0.7:
        return "MEDIUM"
    return "HIGH"

def get_decision(level: str) -> str:
    return {"LOW": "ALLOW", "MEDIUM": "MANUAL_CHECK", "HIGH": "REJECT"}[level]

def compute_risk_score(amount: float, avg_amount: float, is_new_location: int,
                       is_flagged_device: int, velocity: int) -> float:
    score = 0.0
    if amount > 5000:
        score += 0.4
    deviation = (amount / avg_amount) if avg_amount > 0 else 1.0
    if deviation > 3:
        score += 0.4
    if is_new_location:
        score += 0.2
    if is_flagged_device:
        score += 0.3
    if velocity >= 5:
        score += 0.5
    # small ML-like noise
    score += random.uniform(0, 0.1)
    return min(round(score, 4), 1.0)

# ── Main Seeder ──────────────────────────────────────────────────
def populate():
    db: Session = SessionLocal()

    try:
        # ── 1. Clear existing data ────────────────────────────────
        print("Clearing existing data...")
        db.query(models.Claim).delete()
        db.query(models.Transaction).delete()
        db.query(models.User).delete()
        db.commit()

        # ── 2. Create Users ───────────────────────────────────────
        print(f"Creating {TOTAL_USERS} users...")
        users = []
        used_emails = set()

        for i in range(TOTAL_USERS):
            first = random.choice(FIRST_NAMES)
            last  = random.choice(LAST_NAMES)
            email = random_email(first, last, i)
            while email in used_emails:
                email = random_email(first, last, i + random.randint(100, 999))
            used_emails.add(email)

            user = models.User(
                name=f"{first} {last}",
                email=email,
                hashed_password=hash_password("password123")
            )
            db.add(user)

        db.commit()
        users = db.query(models.User).all()
        print(f"  Created {len(users)} users")

        # ── 3. Build user profiles ────────────────────────────────
        # Each user has a home location, usual device, and avg spend
        user_profiles = {}
        for user in users:
            user_profiles[user.id] = {
                "home_location": random.choice(LOCATIONS),
                "home_device":   f"DEV_{user.id:04d}",
                "avg_spend":     round(random.uniform(50, 800), 2),
            }

        # Shared devices — used by fraud rings (3+ users share one device)
        shared_devices = [f"SHARED_{i:03d}" for i in range(20)]

        # ── 4. Create Transactions ────────────────────────────────
        print(f"Creating {TOTAL_TRANSACTIONS} transactions...")
        fraud_count = 0
        legit_count = 0
        transactions = []

        for _ in range(TOTAL_TRANSACTIONS):
            user          = random.choice(users)
            profile       = user_profiles[user.id]
            is_fraud_tx   = random.random() < FRAUD_RATE
            created_at    = random_datetime(365)

            if is_fraud_tx:
                # Fraud pattern — high amount, new location, shared device, high velocity
                amount          = round(random.uniform(3000, 15000), 2)
                location        = random.choice([l for l in LOCATIONS if l != profile["home_location"]])
                device_id       = random.choice(shared_devices)
                is_new_location = 1
                is_flagged_device = 1
                velocity        = random.randint(5, 15)
                fraud_count    += 1
            else:
                # Legitimate pattern — normal amount, home location, own device
                amount          = round(random.gauss(profile["avg_spend"], profile["avg_spend"] * 0.3), 2)
                amount          = max(5.0, amount)   # no negative amounts
                location        = profile["home_location"] if random.random() < 0.8 else random.choice(LOCATIONS)
                device_id       = profile["home_device"] if random.random() < 0.9 else f"DEV_{random.randint(1000, 9999)}"
                is_new_location = 0 if location == profile["home_location"] else 1
                is_flagged_device = 0
                velocity        = random.randint(1, 4)
                legit_count    += 1

            risk_score  = compute_risk_score(
                amount, profile["avg_spend"], is_new_location, is_flagged_device, velocity
            )
            risk_level  = get_risk_level(risk_score)
            decision    = get_decision(risk_level)

            tx = models.Transaction(
                user_id    = user.id,
                amount     = amount,
                location   = location,
                device_id  = device_id,
                is_fraud   = is_fraud_tx,
                risk_score = risk_score,
                risk_level = risk_level,
                decision   = decision,
                created_at = created_at,
            )
            db.add(tx)
            transactions.append(tx)

            # Batch commit every 1000 rows for performance
            if len(transactions) % 1000 == 0:
                db.commit()
                sys.stdout.write(f"\r  Progress: {len(transactions)}/{TOTAL_TRANSACTIONS}")
                sys.stdout.flush()

        db.commit()
        print(f"\n  Legitimate: {legit_count} | Fraud: {fraud_count} ({fraud_count/TOTAL_TRANSACTIONS*100:.1f}%)")

        # ── 5. Create Claims for fraud transactions ───────────────
        print("Creating claims for fraud transactions...")
        all_transactions = db.query(models.Transaction).all()
        fraud_transactions = [t for t in all_transactions if t.is_fraud]

        claim_count = 0
        for tx in fraud_transactions:
            if random.random() > CLAIM_RATE:
                continue

            # Determine claim age relative to transaction
            tx_time   = tx.created_at
            claim_lag = timedelta(days=random.randint(1, 30))
            claim_time = tx_time + claim_lag

            # Status based on fraud patterns
            if random.random() < 0.5:
                status = "APPROVED"
            elif random.random() < 0.3:
                status = "MANUAL_REVIEW"
            else:
                status = "REJECTED"

            claim = models.Claim(
                transaction_id = tx.id,
                reason         = random.choice(CLAIM_REASONS),
                amount         = tx.amount,
                status         = status,
                created_at     = claim_time,
            )
            db.add(claim)
            claim_count += 1

        # Also add a few false claims on legit transactions (serial claimers)
        legit_transactions = [t for t in all_transactions if not t.is_fraud]
        serial_claimer_txs = random.sample(legit_transactions, min(50, len(legit_transactions)))

        for tx in serial_claimer_txs:
            claim = models.Claim(
                transaction_id = tx.id,
                reason         = random.choice(CLAIM_REASONS),
                amount         = tx.amount,
                status         = "REJECTED",
                created_at     = tx.created_at + timedelta(days=random.randint(1, 10)),
            )
            db.add(claim)
            claim_count += 1

        db.commit()
        print(f"  Created {claim_count} claims")

        # ── 6. Summary ────────────────────────────────────────────
        print("\n=== Database populated successfully! ===")
        print(f"  Users:        {len(users)}")
        print(f"  Transactions: {TOTAL_TRANSACTIONS}")
        print(f"  Fraud txns:   {fraud_count} ({fraud_count/TOTAL_TRANSACTIONS*100:.1f}%)")
        print(f"  Legit txns:   {legit_count}")
        print(f"  Claims:       {claim_count}")

    except Exception as e:
        print(f"\nError: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    populate()
