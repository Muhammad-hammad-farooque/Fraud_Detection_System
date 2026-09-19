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

# City -> (latitude, longitude, ISO 4217 currency). Covers every entry in LOCATIONS.
CITY_INFO = {
    "New York": (40.7128, -74.0060, "USD"),     "London": (51.5074, -0.1278, "GBP"),
    "Paris": (48.8566, 2.3522, "EUR"),          "Dubai": (25.2048, 55.2708, "AED"),
    "Tokyo": (35.6762, 139.6503, "JPY"),        "Sydney": (-33.8688, 151.2093, "AUD"),
    "Toronto": (43.6532, -79.3832, "CAD"),      "Berlin": (52.5200, 13.4050, "EUR"),
    "Mumbai": (19.0760, 72.8777, "INR"),        "Singapore": (1.3521, 103.8198, "SGD"),
    "Los Angeles": (34.0522, -118.2437, "USD"), "Chicago": (41.8781, -87.6298, "USD"),
    "Houston": (29.7604, -95.3698, "USD"),      "Phoenix": (33.4484, -112.0740, "USD"),
    "Philadelphia": (39.9526, -75.1652, "USD"), "San Antonio": (29.4241, -98.4936, "USD"),
    "San Diego": (32.7157, -117.1611, "USD"),   "Dallas": (32.7767, -96.7970, "USD"),
    "San Jose": (37.3382, -121.8863, "USD"),    "Austin": (30.2672, -97.7431, "USD"),
    "Karachi": (24.8607, 67.0011, "PKR"),       "Lahore": (31.5204, 74.3587, "PKR"),
    "Istanbul": (41.0082, 28.9784, "TRY"),      "Moscow": (55.7558, 37.6173, "RUB"),
    "Beijing": (39.9042, 116.4074, "CNY"),      "Shanghai": (31.2304, 121.4737, "CNY"),
    "Seoul": (37.5665, 126.9780, "KRW"),        "Bangkok": (13.7563, 100.5018, "THB"),
    "Jakarta": (-6.2088, 106.8456, "IDR"),      "Cairo": (30.0444, 31.2357, "EGP"),
}

# ISO 18245 merchant category codes. Everyday spend for legitimate customers;
# categories fraudsters favour because the goods resell or the money moves.
EVERYDAY_MCCS = ["5411", "5812", "5541", "4111", "5311", "5912", "5814", "4121", "5999", "5651"]
HIGH_RISK_MCCS = ["4829", "6051", "7995", "5732", "5944"]  # wires, quasi-cash, gambling, electronics, jewellery

MERCHANTS_PER_MCC = 12
LEGIT_CHANNEL_WEIGHTS = {"pos": 0.55, "mobile": 0.25, "web": 0.15, "atm": 0.05}
FRAUD_CHANNEL_WEIGHTS = {"web": 0.70, "mobile": 0.25, "pos": 0.03, "atm": 0.02}  # card-not-present

# A handful of anonymising proxy ranges shared across fraud rings - the kind
# of repeated infrastructure the T-17 graph features are meant to surface.
PROXY_PREFIXES = ["185.220", "45.153", "91.219", "103.251"]

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

def merchant_id_for(mcc: str, index: int) -> str:
    return f"M{mcc}-{index:03d}"


def city_ip_prefix(city: str) -> str:
    """A stable, public-looking /16 per city, so a customer's home IP is consistent."""
    idx = LOCATIONS.index(city)
    return f"{41 + idx}.{(idx * 37) % 256}"


def build_payment_profile(rng: random.Random, home_location: str) -> dict:
    """What a customer's normal payments look like: cards, merchants, network."""
    return {
        "cards": [f"tok_{rng.getrandbits(64):016x}" for _ in range(rng.choice([1, 1, 2]))],
        "favourite_merchants": [
            (mcc, merchant_id_for(mcc, rng.randrange(MERCHANTS_PER_MCC)))
            for mcc in rng.sample(EVERYDAY_MCCS, 5)
        ],
        "ip_prefix": city_ip_prefix(home_location),
        "home_host": f"{rng.randrange(256)}.{rng.randrange(1, 255)}",
    }


def _jitter(rng: random.Random, coordinate: float, spread: float = 0.08) -> float:
    return round(coordinate + rng.uniform(-spread, spread), 6)


def payment_context(rng: random.Random, profile: dict, location: str, is_fraud: bool) -> dict:
    """The T-15 payment-context fields for one generated transaction.

    Legitimate: the customer's own cards and favourite merchants, mostly in
    person, from home network space, priced in the local currency. Fraud:
    high-risk merchant categories over card-not-present channels, through
    shared proxy ranges, often priced in a foreign currency.
    """
    lat, lon, currency = CITY_INFO[location]
    if is_fraud:
        mcc = rng.choice(HIGH_RISK_MCCS)
        merchant = merchant_id_for(mcc, rng.randrange(MERCHANTS_PER_MCC))
        channel_weights = FRAUD_CHANNEL_WEIGHTS
        ip = f"{rng.choice(PROXY_PREFIXES)}.{rng.randrange(256)}.{rng.randrange(1, 255)}"
        if rng.random() < 0.4:
            currency = rng.choice(["USD", "EUR"])  # cross-border cash-out
    else:
        if rng.random() < 0.85:
            mcc, merchant = rng.choice(profile["favourite_merchants"])
        else:
            mcc = rng.choice(EVERYDAY_MCCS)
            merchant = merchant_id_for(mcc, rng.randrange(MERCHANTS_PER_MCC))
        channel_weights = LEGIT_CHANNEL_WEIGHTS
        ip = f"{profile['ip_prefix']}.{profile['home_host']}"

    channel = rng.choices(list(channel_weights), weights=list(channel_weights.values()))[0]
    return {
        "merchant_id": merchant,
        "merchant_category": mcc,
        "currency": currency,
        "channel": channel,
        "ip_address": ip,
        # Fraud is usually a real customer's stolen card, not a new one.
        "card_token": rng.choice(profile["cards"]),
        "external_txn_id": f"psp_{rng.getrandbits(96):024x}",
        "latitude": _jitter(rng, lat),
        "longitude": _jitter(rng, lon),
    }


# ── Main Seeder ──────────────────────────────────────────────────
def populate():
    db: Session = SessionLocal()

    try:
        # ── 1. Clear existing data ────────────────────────────────
        # decision_audits is append-only (T-11): its rows cannot be deleted, so
        # a database that already holds real decisions cannot be reseeded.
        if db.query(models.DecisionAudit).count():
            raise SystemExit(
                "Refusing to seed: decision_audits already has rows and is append-only. "
                "Seed a fresh database instead."
            )
        print("Clearing existing data...")
        for model in (models.StepUpChallenge, models.Case, models.TransactionOutcome,
                      models.Claim, models.Transaction, models.User):
            db.query(model).delete()
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
                hashed_password=hash_password("password123"),
                # Signed up before any generated transaction (those span 365 days),
                # so account_age_days is never negative.
                created_at=datetime.now(timezone.utc) - timedelta(days=random.randint(400, 2000)),
            )
            db.add(user)

        db.commit()
        users = db.query(models.User).all()
        print(f"  Created {len(users)} users")

        # ── 3. Build user profiles ────────────────────────────────
        # Each user has a home location, usual device, and avg spend
        user_profiles = {}
        for user in users:
            home_location = random.choice(LOCATIONS)
            user_profiles[user.id] = {
                "home_location": home_location,
                "home_device":   f"DEV_{user.id:04d}",
                "avg_spend":     round(random.uniform(50, 800), 2),
                "payments":      build_payment_profile(random, home_location),
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
                **payment_context(random, profile["payments"], location, is_fraud_tx),
                predicted_fraud = is_fraud_tx,
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
        fraud_transactions = [t for t in all_transactions if t.predicted_fraud]

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
        legit_transactions = [t for t in all_transactions if not t.predicted_fraud]
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
