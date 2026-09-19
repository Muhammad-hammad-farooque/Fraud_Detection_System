"""
Realistic synthetic transaction history for training (T-18).

Replaces the 20 hand-written rows that were perfectly separable on amount
alone (A9). The data is built to be hard in the ways real fraud data is hard:

  * Fraud is rare: about 0.7% of transactions, not 50%.
  * Amounts overlap. Legitimate customers make occasional large purchases, and
    much fraud stays inside the victim's normal spending range.
  * Time has structure: a daily rhythm in each customer's local time zone, and
    a different mix at weekends.
  * Three attacks are planted, each with the fingerprint it leaves in practice:
    card testing, account takeover, and a device-sharing ring - plus a share of
    subtle fraud that looks like ordinary spending.
  * No fingerprint is exclusive to fraud. Legitimate customers upgrade phones,
    borrow laptops, shop through VPNs that geolocate abroad, buy 0.99 apps,
    wire money home and travel to the cities attackers operate from; and some
    fraud comes from the victim's own device, city and daytime. Without these
    hard cases the classes separate perfectly on device or origin alone, and
    the model learns nothing that would survive real data.

Every transaction is labelled with its ground truth as a SYNTHETIC outcome.
The generator writes only to a database that holds nothing but synthetic data,
so its labels can never mix with real ones.

Usage:
    python -m scripts.generate_data                      # 50,000 rows into data/synthetic.db
    python -m scripts.generate_data --transactions 5000 --reset
"""

import argparse
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DATABASE = "sqlite:///data/synthetic.db"
os.environ.setdefault("DATABASE_URL", DEFAULT_DATABASE)
os.environ.setdefault("SECRET_KEY", "synthetic-data-only")

from sqlalchemy import create_engine, insert                     # noqa: E402
from sqlalchemy.orm import Session, sessionmaker                 # noqa: E402

from app import models                                           # noqa: E402
from app.auth import hash_password                               # noqa: E402
from app.database import Base                                    # noqa: E402
from app.models import OutcomeSource                             # noqa: E402

SYNTHETIC_DOMAIN = "synthetic.invalid"

# City -> (latitude, longitude, ISO 4217 currency)
CITIES = {
    "New York": (40.7128, -74.0060, "USD"), "London": (51.5074, -0.1278, "GBP"),
    "Paris": (48.8566, 2.3522, "EUR"), "Dubai": (25.2048, 55.2708, "AED"),
    "Tokyo": (35.6762, 139.6503, "JPY"), "Sydney": (-33.8688, 151.2093, "AUD"),
    "Toronto": (43.6532, -79.3832, "CAD"), "Berlin": (52.5200, 13.4050, "EUR"),
    "Mumbai": (19.0760, 72.8777, "INR"), "Singapore": (1.3521, 103.8198, "SGD"),
    "Chicago": (41.8781, -87.6298, "USD"), "Karachi": (24.8607, 67.0011, "PKR"),
    "Lahore": (31.5204, 74.3587, "PKR"), "Istanbul": (41.0082, 28.9784, "TRY"),
    "Seoul": (37.5665, 126.9780, "KRW"), "Cairo": (30.0444, 31.2357, "EGP"),
}
FRAUD_ORIGINS = ["Moscow", "Lagos", "Bucharest", "Jakarta", "Sao Paulo"]
CITIES.update({
    "Moscow": (55.7558, 37.6173, "RUB"), "Lagos": (6.5244, 3.3792, "NGN"),
    "Bucharest": (44.4268, 26.1025, "RON"), "Jakarta": (-6.2088, 106.8456, "IDR"),
    "Sao Paulo": (-23.5505, -46.6333, "BRL"),
})
HOME_CITIES = [c for c in CITIES if c not in FRAUD_ORIGINS]

# ISO 18245 merchant category codes
EVERYDAY_MCCS = {"5411": 0.30, "5812": 0.18, "5541": 0.12, "4111": 0.08, "5311": 0.07,
                 "5912": 0.07, "5814": 0.08, "5651": 0.05, "5999": 0.05}
WEEKEND_MCCS = {"5812": 0.30, "5813": 0.15, "5651": 0.15, "7832": 0.10, "5411": 0.20, "5541": 0.10}
BIG_TICKET_MCCS = ["5732", "5944", "4722", "5712"]       # electronics, jewellery, travel, furniture
CASH_OUT_MCCS = ["4829", "6051", "5732", "5944"]          # wires, quasi-cash, resellable goods
TESTING_MCCS = ["5815", "5816", "5999"]                   # digital goods: cheap, instant
PROXY_PREFIXES = ["185.220", "45.153", "91.219", "103.251"]
VPN_EXITS = ["London", "Singapore", "New York", "Bucharest", "Moscow"]   # commercial VPNs exit here
REMITTANCE_HOMES = {"Karachi", "Lahore", "Mumbai", "Cairo"}

# Relative activity by local hour: quiet overnight, lunch and evening peaks.
DIURNAL = [0.15, 0.08, 0.05, 0.04, 0.05, 0.12, 0.35, 0.7, 0.9, 1.0, 1.0, 1.15,
           1.35, 1.2, 1.0, 1.0, 1.05, 1.2, 1.4, 1.45, 1.3, 1.0, 0.65, 0.35]
NIGHT_OWL = DIURNAL[18:] + DIURNAL[:18]                   # the same shape, six hours later


@dataclass(frozen=True)
class GeneratorConfig:
    transactions: int = 50_000
    users: int = 1_000
    days: int = 365
    fraud_rate: float = 0.007
    seed: int = 42
    # Labels need 90 days to mature (T-06), so the window ends before that.
    end: datetime = field(default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=91))


@dataclass
class GenerationReport:
    transactions: int = 0
    fraud: int = 0
    users: int = 0
    patterns: Counter = field(default_factory=Counter)

    @property
    def fraud_rate(self) -> float:
        return self.fraud / self.transactions if self.transactions else 0.0


class NotASyntheticDatabase(RuntimeError):
    """The target database holds data that the generator did not create."""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _weighted(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _utc_offset_hours(city: str) -> int:
    return round(CITIES[city][1] / 15)          # longitude is a fine proxy for a time zone here


def _jitter(rng: random.Random, value: float, spread: float = 0.05) -> float:
    return round(value + rng.uniform(-spread, spread), 6)


def _ip(rng: random.Random, prefix: str) -> str:
    return f"{prefix}.{rng.randrange(256)}.{rng.randrange(1, 255)}"


def _city_prefix(city: str) -> str:
    idx = sorted(CITIES).index(city)
    return f"{60 + idx}.{(idx * 41) % 256}"


def _merchant(rng: random.Random, mcc: str, pool: int = 25) -> str:
    return f"M{mcc}-{rng.randrange(pool):03d}"


def _money(amount: float) -> float:
    return round(max(amount, 0.5), 2)


# ── Customers ───────────────────────────────────────────────────────────────

def _make_customer(rng: random.Random, user_id: int, window_start: datetime) -> dict:
    home = rng.choice(HOME_CITIES)
    lat, lon, currency = CITIES[home]
    new_account = rng.random() < 0.06
    created = (window_start + timedelta(days=rng.uniform(0, 200)) if new_account
               else window_start - timedelta(days=rng.uniform(30, 2000)))
    return {
        "id": user_id,
        "home": home,
        "coords": (lat, lon),
        "currency": currency,
        "tz": _utc_offset_hours(home),
        "created_at": created,
        "activity": rng.lognormvariate(0, 0.8),
        "spend_mu": math.log(rng.uniform(15, 250)),
        "spend_sigma": rng.uniform(0.45, 0.9),
        "diurnal": NIGHT_OWL if rng.random() < 0.08 else DIURNAL,
        "weekend_boost": rng.uniform(0.6, 1.8),
        "cards": [f"tok_{rng.getrandbits(64):016x}" for _ in range(rng.choice([1, 1, 1, 2]))],
        "devices": [f"dev-{user_id:05d}-{i}" for i in range(rng.choice([1, 1, 2]))],
        "favourites": [(mcc, _merchant(rng, mcc)) for mcc in rng.sample(list(EVERYDAY_MCCS), 6)],
        "online_share": rng.uniform(0.05, 0.6),
        "host": f"{rng.randrange(256)}.{rng.randrange(1, 255)}",
        "traveller": rng.random() < 0.15,
        # Hard negatives: behaviour that looks like fraud but is not.
        "vpn": rng.random() < 0.05,                       # online purchases geolocate to a VPN exit
        "remits": home in REMITTANCE_HOMES and rng.random() < 0.35 or rng.random() < 0.03,
        "upgrade_day": rng.randrange(365) if rng.random() < 0.3 else None,   # a new phone mid-year
    }


def _legit_amount(rng: random.Random, customer: dict) -> tuple[float, str | None]:
    """Everyday spend, with the occasional big purchase that overlaps fraud amounts."""
    if rng.random() < 0.025:
        return _money(rng.lognormvariate(customer["spend_mu"], customer["spend_sigma"]) * rng.uniform(5, 25)), \
            rng.choice(BIG_TICKET_MCCS)
    return _money(rng.lognormvariate(customer["spend_mu"], customer["spend_sigma"])), None


def _local_time(rng: random.Random, day: datetime, customer: dict) -> datetime:
    local_hour = rng.choices(range(24), weights=customer["diurnal"])[0]
    moment = day + timedelta(hours=local_hour - customer["tz"],
                             minutes=rng.randrange(60), seconds=rng.randrange(60))
    return moment


def _legit_transaction(rng: random.Random, customer: dict, at: datetime, city: str,
                       day_index: int = 0) -> dict:
    amount, big_ticket_mcc = _legit_amount(rng, customer)
    weekend = at.weekday() >= 5
    roll = rng.random()
    if roll < 0.04:
        # App stores and streaming: tiny, online, digital - the same shape as card testing.
        big_ticket_mcc = rng.choice(["5815", "5816"])
        amount = rng.choice([0.99, 1.99, 2.99, 4.99, 9.99])
    elif roll < 0.07 and customer["remits"]:
        # Money sent home: a wire, often large, often at night in the recipient's time zone.
        big_ticket_mcc = "4829"
        amount = _money(rng.lognormvariate(customer["spend_mu"], customer["spend_sigma"]) * rng.uniform(3, 12))
    if big_ticket_mcc:
        mcc = big_ticket_mcc
        merchant = _merchant(rng, mcc)
    elif rng.random() < 0.8:
        mcc, merchant = rng.choice(customer["favourites"])
        if weekend and rng.random() < 0.35:
            mcc = _weighted(rng, WEEKEND_MCCS)
            merchant = _merchant(rng, mcc)
    else:
        mcc = _weighted(rng, WEEKEND_MCCS if weekend else EVERYDAY_MCCS)
        merchant = _merchant(rng, mcc)

    online = rng.random() < customer["online_share"] or mcc in ("5815", "5816", "4829")
    channel = rng.choice(["web", "mobile"]) if online else ("atm" if rng.random() < 0.04 else "pos")
    at_home = city == customer["home"]
    ip_prefix = _city_prefix(city)
    if online and customer["vpn"]:
        # A VPN places online purchases wherever its exit is - often "impossible" travel.
        city = rng.choice(VPN_EXITS)
        ip_prefix = rng.choice(PROXY_PREFIXES)
        at_home = False
    lat, lon, currency = CITIES[city]
    currency = customer["currency"] if at_home or channel in ("web", "mobile") else currency

    devices = list(customer["devices"])
    if customer["upgrade_day"] is not None and day_index >= customer["upgrade_day"]:
        devices = [f"dev-{customer['id']:05d}-new"]      # the old phone is retired
    device = rng.choice(devices)
    if rng.random() < 0.015:
        device = f"dev-borrowed-{rng.getrandbits(32):08x}"   # a work laptop, a friend's phone
    return {
        "user_id": customer["id"],
        "amount": amount,
        "location": city,
        "device_id": device,
        "merchant_id": merchant,
        "merchant_category": mcc,
        "currency": currency,
        "channel": channel,
        "ip_address": (_ip(rng, ip_prefix) if not at_home
                       else f"{ip_prefix}.{customer['host']}"),
        "card_token": rng.choice(customer["cards"]),
        "external_txn_id": f"psp_{rng.getrandbits(96):024x}",
        "latitude": _jitter(rng, lat),
        "longitude": _jitter(rng, lon),
        "created_at": at,
        "fraud": False,
        "pattern": "legit",
    }


def _legit_history(rng: random.Random, customers: list[dict], count: int,
                   window_start: datetime, days: int) -> list[dict]:
    rows = []
    weights = [c["activity"] for c in customers]
    trips = {}                       # customer id -> (start day, end day, city)
    for customer in customers:
        if customer["traveller"]:
            start = rng.randrange(days - 10)
            # Some trips go to the same cities attackers operate from.
            destinations = [c for c in CITIES if c != customer["home"]]
            trips[customer["id"]] = (start, start + rng.randint(3, 10), rng.choice(destinations))

    for customer in rng.choices(customers, weights=weights, k=count):
        # Weekends get the customer's own weight; weekdays weight 1.
        for _ in range(20):
            day_index = rng.randrange(days)
            day = window_start + timedelta(days=day_index)
            factor = customer["weekend_boost"] if day.weekday() >= 5 else 1.0
            if rng.random() < factor / max(customer["weekend_boost"], 1.0):
                break
        at = _local_time(rng, day, customer)
        if at < customer["created_at"]:
            at = customer["created_at"] + timedelta(hours=rng.uniform(1, 72))
        city = customer["home"]
        trip = trips.get(customer["id"])
        if trip and trip[0] <= day_index <= trip[1]:
            city = trip[2]
        rows.append(_legit_transaction(rng, customer, at, city, day_index))
    return rows


# ── Attacks ─────────────────────────────────────────────────────────────────

def _fraud_row(rng, customer, at, *, amount, mcc, channel, city, device, ip, card, pattern):
    lat, lon, currency = CITIES[city]
    return {
        "user_id": customer["id"], "amount": _money(amount), "location": city, "device_id": device,
        "merchant_id": _merchant(rng, mcc, pool=8), "merchant_category": mcc,
        "currency": currency if rng.random() < 0.6 else rng.choice(["USD", "EUR"]),
        "channel": channel, "ip_address": ip, "card_token": card,
        "external_txn_id": f"psp_{rng.getrandbits(96):024x}",
        "latitude": _jitter(rng, lat), "longitude": _jitter(rng, lon),
        "created_at": at, "fraud": True, "pattern": pattern,
    }


def _card_testing(rng, customer, start) -> list[dict]:
    """Many tiny charges in minutes on a stolen card, then a cash-out. Some
    testers work from residential connections in the victim's own country."""
    local = rng.random() < 0.3
    city = customer["home"] if local else rng.choice(FRAUD_ORIGINS)
    device = f"atk-{rng.getrandbits(40):010x}"
    ip = _ip(rng, _city_prefix(city) if local else rng.choice(PROXY_PREFIXES))
    card = rng.choice(customer["cards"])
    rows, at = [], start
    for _ in range(rng.randint(4, 10)):
        at += timedelta(seconds=rng.randint(15, 90))
        rows.append(_fraud_row(rng, customer, at, amount=rng.uniform(0.5, 4.99), mcc=rng.choice(TESTING_MCCS),
                               channel="web", city=city, device=device, ip=ip, card=card, pattern="card_testing"))
    for _ in range(rng.randint(1, 2)):
        at += timedelta(minutes=rng.randint(10, 240))
        cash_out = rng.lognormvariate(customer["spend_mu"], customer["spend_sigma"]) * rng.uniform(1, 6)
        rows.append(_fraud_row(rng, customer, at, amount=cash_out, mcc=rng.choice(CASH_OUT_MCCS),
                               channel="web", city=city, device=device, ip=ip, card=card, pattern="card_testing"))
    return rows


def _account_takeover(rng, customer, start) -> list[dict]:
    """A new device moving money out, often inside normal amounts. Usually far
    away at night; sometimes a local attacker in the daytime."""
    local = rng.random() < 0.3
    city = customer["home"] if local else rng.choice(FRAUD_ORIGINS)
    device = f"atk-{rng.getrandbits(40):010x}"
    ip = _ip(rng, _city_prefix(city) if local else rng.choice(PROXY_PREFIXES + [_city_prefix(city)]))
    hour = rng.choice([1, 2, 3, 4]) if not local or rng.random() < 0.4 else rng.choice(range(9, 20))
    rows, at = [], start.replace(hour=(hour - customer["tz"]) % 24)
    if not local and rng.random() < 0.4:
        # The customer paid at home a little earlier, so the attacker's first
        # charge abroad implies an impossible journey - the signal that makes
        # geo-velocity one of the strongest in fraud detection.
        anchor = at - timedelta(minutes=rng.randint(20, 180))
        rows.append(_legit_transaction(rng, customer, anchor, customer["home"]))
    for _ in range(rng.randint(1, 4)):
        at += timedelta(minutes=rng.randint(5, 120))
        normal = rng.lognormvariate(customer["spend_mu"], customer["spend_sigma"])
        amount = normal * (rng.uniform(0.8, 2.0) if rng.random() < 0.4 else rng.uniform(4, 30))
        rows.append(_fraud_row(rng, customer, at, amount=amount, mcc=rng.choice(["4829", "6051", "4829"]),
                               channel=rng.choice(["web", "mobile"]), city=city, device=device, ip=ip,
                               card=rng.choice(customer["cards"]), pattern="account_takeover"))
    return rows


def _device_ring(rng, victims, start) -> list[dict]:
    """One device and one network block cashing out several compromised accounts."""
    device = f"ring-{rng.getrandbits(40):010x}"
    prefix = rng.choice(PROXY_PREFIXES)
    city = rng.choice(FRAUD_ORIGINS)
    rows = []
    for victim in victims:
        at = start + timedelta(hours=rng.uniform(0, 96))
        for _ in range(rng.randint(1, 3)):
            at += timedelta(minutes=rng.randint(3, 90))
            rows.append(_fraud_row(rng, victim, at, amount=rng.uniform(150, 1800), mcc=rng.choice(["5732", "5944"]),
                                   channel="web", city=city, device=device, ip=_ip(rng, prefix),
                                   card=rng.choice(victim["cards"]), pattern="device_ring"))
    return rows


def _subtle(rng, customer, start) -> list[dict]:
    """A stolen card used like the owner would: home city, normal amount, everyday
    shop - and half the time from the owner's own device (malware, a household
    member), so there is no new-device signal either."""
    mcc, _ = rng.choice(customer["favourites"])
    amount = rng.lognormvariate(customer["spend_mu"], customer["spend_sigma"]) * rng.uniform(0.8, 1.6)
    device = rng.choice(customer["devices"]) if rng.random() < 0.5 else f"atk-{rng.getrandbits(40):010x}"
    at = start.replace(hour=(rng.choice(range(9, 22)) - customer["tz"]) % 24)
    return [_fraud_row(rng, customer, at, amount=amount, mcc=mcc, channel=rng.choice(["web", "mobile"]),
                       city=customer["home"], device=device,
                       ip=f"{_city_prefix(customer['home'])}.{customer['host']}",
                       card=rng.choice(customer["cards"]), pattern="subtle")]


def _attacks(rng, customers, budget, window_start, days) -> tuple[list[dict], list[dict]]:
    """Fraud rows up to `budget`, plus the genuine purchases some attacks are
    anchored to, which are legitimate and returned separately."""
    rows = []
    while sum(r["fraud"] for r in rows) < budget:
        start = window_start + timedelta(days=rng.uniform(days * 0.05, days - 1))
        eligible = [c for c in customers if c["created_at"] < start]
        # Episodes differ in size (card testing ~8 rows, a ring ~11, subtle 1),
        # so these weights aim at rows split roughly 45 / 25 / 15 / 15.
        kind = rng.choices(["card_testing", "account_takeover", "device_ring", "subtle"],
                           weights=[0.167, 0.316, 0.043, 0.474])[0]
        if kind == "device_ring":
            rows += _device_ring(rng, rng.sample(eligible, rng.randint(4, 7)), start)
        else:
            maker = {"card_testing": _card_testing, "account_takeover": _account_takeover, "subtle": _subtle}[kind]
            rows += maker(rng, rng.choice(eligible), start)
    fraud = [r for r in rows if r["fraud"]][:budget]
    anchors = [r for r in rows if not r["fraud"]]
    return fraud, anchors


# ── Writing ─────────────────────────────────────────────────────────────────

def _check_target(db: Session, reset: bool) -> None:
    users = db.query(models.User.email).all()
    real = [email for (email,) in users if not email.endswith("@" + SYNTHETIC_DOMAIN)]
    if real:
        raise NotASyntheticDatabase(
            f"Refusing to write: the target database holds {len(real)} real account(s). "
            "Synthetic labels must never mix with real ones - point --database somewhere else."
        )
    if users and not reset:
        raise NotASyntheticDatabase("The target already holds synthetic data; pass reset=True to replace it.")
    if users:
        for model in (models.TransactionOutcome, models.Transaction, models.User):
            db.query(model).delete()
        db.commit()


def generate(db: Session, cfg: GeneratorConfig = GeneratorConfig(), reset: bool = False) -> GenerationReport:
    """Write cfg.transactions labelled transactions for cfg.users customers into `db`."""
    _check_target(db, reset)
    rng = random.Random(cfg.seed)
    window_start = (cfg.end - timedelta(days=cfg.days)).replace(hour=0, minute=0, second=0, microsecond=0)

    customers = [_make_customer(rng, i + 1, window_start) for i in range(cfg.users)]
    fraud_budget = max(1, round(cfg.transactions * cfg.fraud_rate))
    fraud_rows, anchor_rows = _attacks(rng, customers, fraud_budget, window_start, cfg.days)
    legit_rows = anchor_rows + _legit_history(
        rng, customers, cfg.transactions - len(fraud_rows) - len(anchor_rows), window_start, cfg.days)

    password = hash_password("synthetic-accounts-cannot-log-in-" + str(rng.getrandbits(64)))
    db.execute(insert(models.User), [
        {"id": c["id"], "name": f"Synthetic {c['id']}", "email": f"user{c['id']}@{SYNTHETIC_DOMAIN}",
         "hashed_password": password, "role": "CUSTOMER", "created_at": c["created_at"]}
        for c in customers
    ])

    rows = sorted(fraud_rows + legit_rows, key=lambda r: r["created_at"])
    report = GenerationReport(transactions=len(rows), users=len(customers))
    txn_rows, outcome_rows = [], []
    for txn_id, row in enumerate(rows, start=1):
        report.patterns[row["pattern"]] += 1
        report.fraud += row["fraud"]
        txn = {k: v for k, v in row.items() if k not in ("fraud", "pattern")}
        txn_rows.append({"id": txn_id, **txn})
        # Fraud is confirmed like a chargeback, weeks later; legitimate spend
        # is confirmed once the chargeback window has passed without one.
        delay = timedelta(days=rng.uniform(15, 75)) if row["fraud"] else timedelta(days=90)
        outcome_rows.append({"transaction_id": txn_id, "is_fraud_confirmed": row["fraud"],
                             "source": OutcomeSource.SYNTHETIC.value, "confirmed_at": row["created_at"] + delay,
                             "notes": row["pattern"]})

    for start in range(0, len(txn_rows), 5000):
        db.execute(insert(models.Transaction), txn_rows[start:start + 5000])
        db.execute(insert(models.TransactionOutcome), outcome_rows[start:start + 5000])
    db.commit()
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate labelled synthetic transactions.")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--transactions", type=int, default=GeneratorConfig.transactions)
    parser.add_argument("--users", type=int, default=GeneratorConfig.users)
    parser.add_argument("--days", type=int, default=GeneratorConfig.days)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--reset", action="store_true", help="replace existing synthetic data")
    args = parser.parse_args(argv)

    if args.database.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(args.database[len("sqlite:///"):]) or ".", exist_ok=True)
    engine = create_engine(args.database)
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as db:
        report = generate(db, GeneratorConfig(transactions=args.transactions, users=args.users,
                                              days=args.days, seed=args.seed), reset=args.reset)
    print(f"Generated {report.transactions} transactions for {report.users} customers")
    print(f"Fraud: {report.fraud} ({report.fraud_rate:.2%})  patterns: {dict(report.patterns)}")


if __name__ == "__main__":
    main()
