"""Single source of truth for feature computation.

Every consumer of features - live scoring, retraining, initial training -
MUST compute them through this module. Never recompute features elsewhere.

compute_features is pure: it reads a candidate transaction and pre-computed
aggregates and nothing else. The aggregates come from exactly one place,
app/repositories/transaction_repo.py, evaluated "as of" the candidate's
timestamp - at request time when serving, at each historical transaction's
own timestamp when training. One implementation of every aggregate, so
training and serving cannot drift apart (T-16).

Conventions for missing information:
  * Cold start: a customer with no history is never penalised. Behavioural
    shares read as fully typical (1.0) and first-time flags read as 0.
  * Unknowable values - no previous transaction, no coordinates - are UNKNOWN
    (-1.0), never 0, which would claim "same place" or "no time elapsed".
"""
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

# ── Feature schema ──────────────────────────────────────────────────────────

# The five features the engine shipped with. Rules read these, and the
# baseline model was trained on exactly these.
BASELINE_FEATURES: list[str] = [
    "amount",
    "amount_deviation",
    "is_new_location",
    "is_flagged_device",
    "velocity_2m",
]

VELOCITY_WINDOWS: tuple[tuple[str, timedelta], ...] = (
    ("1m", timedelta(minutes=1)),
    ("5m", timedelta(minutes=5)),
    ("1h", timedelta(hours=1)),
    ("24h", timedelta(hours=24)),
    ("7d", timedelta(days=7)),
    ("30d", timedelta(days=30)),
)

FEATURE_ORDER: list[str] = BASELINE_FEATURES + [
    # Multi-window velocity
    *[f"txn_count_{name}" for name, _ in VELOCITY_WINDOWS],
    *[f"amount_sum_{name}" for name, _ in VELOCITY_WINDOWS],
    # Geo-velocity
    "km_from_previous",
    "travel_speed_kmh",
    "is_impossible_travel",
    "distinct_locations_24h",
    # Temporal
    "hour_of_day",
    "day_of_week",
    "is_night",
    "seconds_since_previous",
    "account_age_days",
    # Amount shape
    "amount_zscore",
    "amount_population_percentile",
    "is_round_amount",
    "ratio_to_lifetime_max",
    # Device
    "device_age_days",
    "devices_per_user",
    "users_per_device",
    "is_first_use_of_device",
    # Merchant
    "merchant_fraud_rate",
    "is_first_time_merchant",
    "is_new_category",
    "is_card_not_present",
    # Behavioural
    "typical_hour_share",
    "location_share",
    "amount_band_share",
    "category_share",
]

# ── Documented thresholds ───────────────────────────────────────────────────

VELOCITY_WINDOW = timedelta(seconds=120)      # velocity_2m, the rule-facing window
FLAGGED_DEVICE_MIN_USERS = 3

# Impossible travel: faster than a commercial airliner cruises (~900 km/h).
# Journeys under 100 km are ignored, so GPS jitter and neighbouring cells
# seconds apart are not flagged as teleportation.
IMPOSSIBLE_SPEED_KMH = 900.0
IMPOSSIBLE_MIN_KM = 100.0
MIN_ELAPSED_SECONDS = 1.0                     # avoid dividing by zero-second gaps

NIGHT_HOURS = frozenset(range(0, 6))          # UTC; see is_night below
TYPICAL_HOUR_RADIUS = 2                       # an hour within +-2 of the usual counts as typical

# merchant_fraud_rate is smoothed toward a prior, so a merchant with one
# confirmed fraud out of one label does not read as 100% fraudulent.
MERCHANT_PRIOR_RATE = 0.01
MERCHANT_PRIOR_WEIGHT = 10.0

CARD_NOT_PRESENT_CHANNELS = frozenset({"web", "mobile"})
EARTH_RADIUS_KM = 6371.0088
UNKNOWN = -1.0


# ── Inputs ──────────────────────────────────────────────────────────────────

def _is_missing(value: Any) -> bool:
    """None, NaN, pd.NA or NaT - anything a DataFrame might use for 'absent'."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):          # containers: not a scalar "missing"
        return False


@dataclass(frozen=True)
class TransactionInput:
    """The transaction being scored, and the instant it is scored at."""
    amount: float
    location: str
    at: datetime
    device_id: str | None = None
    merchant_id: str | None = None
    merchant_category: str | None = None
    channel: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    @classmethod
    def from_request(cls, request: Any, at: datetime) -> "TransactionInput":
        """Build from a TransactionCreate, an ORM row, or a DataFrame row.

        Every candidate - served or retrained - enters through here, so this is
        where "missing" is normalised. A DataFrame reports a missing value as
        NaN (or pd.NA), not None, and NaN passes an `is not None` check: left
        alone, a transaction with no merchant would train as a first visit to
        one, and missing coordinates would reach the distance maths.
        """
        def field(name: str):
            value = getattr(request, name, None)
            value = getattr(value, "value", value)          # enums, e.g. Channel
            return None if _is_missing(value) else value

        return cls(
            amount=float(request.amount),
            location=request.location,
            at=at,
            device_id=field("device_id"),
            merchant_id=field("merchant_id"),
            merchant_category=field("merchant_category"),
            channel=field("channel"),
            latitude=field("latitude"),
            longitude=field("longitude"),
        )


@dataclass(frozen=True)
class UserAggregates:
    """One customer's history before the candidate's timestamp.

    Several fields are relative to the candidate (location_count counts prior
    transactions at *this* location, and so on), which is why the repository
    takes the candidate as input. Never computed inside compute_features.
    """
    txn_count: int
    avg_amount: float
    known_locations: frozenset[str]
    count_last_2m: int
    is_cold_start: bool = False
    window_counts: tuple[int, ...] = (0,) * len(VELOCITY_WINDOWS)
    window_sums: tuple[float, ...] = (0.0,) * len(VELOCITY_WINDOWS)
    amount_std: float = 0.0
    max_amount: float = 0.0
    distinct_locations_24h: int = 0
    distinct_devices: int = 0
    location_count: int = 0
    device_count: int = 0
    merchant_count: int = 0
    category_count: int = 0
    typical_hour_count: int = 0
    amount_band_count: int = 0
    previous_at: datetime | None = None
    previous_latitude: float | None = None
    previous_longitude: float | None = None
    account_created_at: datetime | None = None


@dataclass(frozen=True)
class DeviceAggregates:
    """The candidate's device across every customer, before the candidate's timestamp."""
    user_count: int = 0
    first_seen_at: datetime | None = None


@dataclass(frozen=True)
class PopulationAggregates:
    """Everyone's history before the candidate's timestamp."""
    amount_rank: int = 0            # prior transactions with a smaller amount
    total_count: int = 0
    merchant_labelled: int = 0      # confirmed outcomes at the candidate's merchant
    merchant_fraud: int = 0


# ── Output ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureVector:
    # Baseline features: no defaults, always computed.
    amount: float
    amount_deviation: float
    is_new_location: int
    is_flagged_device: int
    velocity_2m: int
    # T-16 features. The defaults exist only so audit rows written before T-16,
    # which stored the five baseline features, can still be reconstructed for
    # replay. compute_features always sets every one of them.
    txn_count_1m: int = 0
    txn_count_5m: int = 0
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    txn_count_7d: int = 0
    txn_count_30d: int = 0
    amount_sum_1m: float = 0.0
    amount_sum_5m: float = 0.0
    amount_sum_1h: float = 0.0
    amount_sum_24h: float = 0.0
    amount_sum_7d: float = 0.0
    amount_sum_30d: float = 0.0
    km_from_previous: float = UNKNOWN
    travel_speed_kmh: float = UNKNOWN
    is_impossible_travel: int = 0
    distinct_locations_24h: int = 0
    hour_of_day: int = 0
    day_of_week: int = 0
    is_night: int = 0
    seconds_since_previous: float = UNKNOWN
    account_age_days: float = UNKNOWN
    amount_zscore: float = 0.0
    amount_population_percentile: float = 0.5
    is_round_amount: int = 0
    ratio_to_lifetime_max: float = 1.0
    device_age_days: float = 0.0
    devices_per_user: int = 0
    users_per_device: int = 0
    is_first_use_of_device: int = 0
    merchant_fraud_rate: float = MERCHANT_PRIOR_RATE
    is_first_time_merchant: int = 0
    is_new_category: int = 0
    is_card_not_present: int = 0
    typical_hour_share: float = 1.0
    location_share: float = 1.0
    amount_band_share: float = 1.0
    category_share: float = 1.0

    def to_frame(self, columns: list[str] | None = None) -> pd.DataFrame:
        """One-row frame. Pass the model's own feature list (its manifest's
        feature_order) so it receives exactly the columns it was trained on,
        in that order."""
        return pd.DataFrame([asdict(self)], columns=columns or FEATURE_ORDER)


# ── Pure helpers, shared with the repository ────────────────────────────────

def as_utc(ts: datetime) -> datetime:
    """Normalise a stored timestamp to UTC.

    Columns are DateTime(timezone=True) since T-08, so PostgreSQL returns aware
    datetimes. SQLite has no timezone type and still hands back naive ones, so
    this single helper exists instead of the .replace(tzinfo=utc) patches that
    were previously scattered across the scoring, claim and script paths (A12).
    """
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def typical_hours(hour: int) -> list[int]:
    """UTC hours within TYPICAL_HOUR_RADIUS of `hour`, wrapping round midnight."""
    return sorted({(hour + offset) % 24 for offset in range(-TYPICAL_HOUR_RADIUS, TYPICAL_HOUR_RADIUS + 1)})


def amount_band(amount: float) -> tuple[float, float]:
    """The order-of-magnitude band containing `amount`: [10^k, 10^(k+1))."""
    if amount < 1:
        return 0.0, 1.0
    k = math.floor(math.log10(amount))
    return float(10 ** k), float(10 ** (k + 1))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def _share(count: int, total: int, cold_start: bool) -> float:
    return 1.0 if cold_start or total == 0 else count / total


def _is_round(amount: float) -> int:
    cents = round(amount * 100)
    return 1 if cents > 0 and cents % 10_000 == 0 else 0     # a whole multiple of 100


# ── The one computation ─────────────────────────────────────────────────────

def compute_features(
    txn: TransactionInput,
    user: UserAggregates,
    device: DeviceAggregates,
    population: PopulationAggregates,
) -> FeatureVector:
    """Pure. No database access, no clock access, no I/O - the instant being
    scored is txn.at, supplied by the caller.

    Cold start: a first-ever transaction must not be penalised for a location,
    merchant or device that could not possibly have been seen before (A16).
    """
    at = as_utc(txn.at)
    cold = user.is_cold_start

    # Baseline
    if cold:
        amount_deviation = 1.0
        is_new_location = 0
    else:
        amount_deviation = txn.amount / user.avg_amount if user.avg_amount > 0 else 1.0
        is_new_location = 0 if txn.location in user.known_locations else 1

    # Geo-velocity
    km = speed = seconds_since = UNKNOWN
    if user.previous_at is not None:
        seconds_since = max((at - as_utc(user.previous_at)).total_seconds(), 0.0)
        have_coords = None not in (txn.latitude, txn.longitude,
                                   user.previous_latitude, user.previous_longitude)
        if have_coords:
            km = haversine_km(user.previous_latitude, user.previous_longitude,
                              txn.latitude, txn.longitude)
            speed = km / (max(seconds_since, MIN_ELAPSED_SECONDS) / 3600.0)
    impossible = int(km >= IMPOSSIBLE_MIN_KM and speed > IMPOSSIBLE_SPEED_KMH)

    # Temporal. is_night uses UTC: the customer's local time zone is not known.
    # typical_hour_share is what makes timing meaningful per customer.
    account_age = UNKNOWN
    if user.account_created_at is not None:
        account_age = max((at - as_utc(user.account_created_at)).total_seconds() / 86400.0, 0.0)

    # Amount shape
    if cold or user.amount_std <= 0:
        zscore = 0.0
    else:
        zscore = (txn.amount - user.avg_amount) / user.amount_std
    percentile = population.amount_rank / population.total_count if population.total_count else 0.5
    ratio_to_max = 1.0 if cold or user.max_amount <= 0 else txn.amount / user.max_amount

    # Device
    device_age = 0.0
    if device.first_seen_at is not None:
        device_age = max((at - as_utc(device.first_seen_at)).total_seconds() / 86400.0, 0.0)

    # Merchant: smoothed toward the prior so thin evidence stays near it.
    merchant_rate = (
        (population.merchant_fraud + MERCHANT_PRIOR_WEIGHT * MERCHANT_PRIOR_RATE)
        / (population.merchant_labelled + MERCHANT_PRIOR_WEIGHT)
    )
    first_time_merchant = int(not cold and txn.merchant_id is not None and user.merchant_count == 0)
    new_category = int(not cold and txn.merchant_category is not None and user.category_count == 0)
    card_not_present = int(txn.channel in CARD_NOT_PRESENT_CHANNELS)

    counts = dict(zip((f"txn_count_{n}" for n, _ in VELOCITY_WINDOWS), user.window_counts))
    sums = dict(zip((f"amount_sum_{n}" for n, _ in VELOCITY_WINDOWS), user.window_sums))

    return FeatureVector(
        amount=float(txn.amount),
        amount_deviation=float(amount_deviation),
        is_new_location=is_new_location,
        is_flagged_device=1 if device.user_count >= FLAGGED_DEVICE_MIN_USERS else 0,
        velocity_2m=user.count_last_2m,
        **{k: int(v) for k, v in counts.items()},
        **{k: float(v) for k, v in sums.items()},
        km_from_previous=float(km),
        travel_speed_kmh=float(speed),
        is_impossible_travel=impossible,
        distinct_locations_24h=user.distinct_locations_24h,
        hour_of_day=at.hour,
        day_of_week=at.weekday(),
        is_night=int(at.hour in NIGHT_HOURS),
        seconds_since_previous=float(seconds_since),
        account_age_days=float(account_age),
        amount_zscore=float(zscore),
        amount_population_percentile=float(percentile),
        is_round_amount=_is_round(txn.amount),
        ratio_to_lifetime_max=float(ratio_to_max),
        device_age_days=float(device_age),
        devices_per_user=user.distinct_devices,
        users_per_device=device.user_count,
        is_first_use_of_device=int(not cold and txn.device_id is not None and user.device_count == 0),
        merchant_fraud_rate=float(merchant_rate),
        is_first_time_merchant=first_time_merchant,
        is_new_category=new_category,
        is_card_not_present=card_not_present,
        typical_hour_share=_share(user.typical_hour_count, user.txn_count, cold),
        location_share=_share(user.location_count, user.txn_count, cold),
        amount_band_share=_share(user.amount_band_count, user.txn_count, cold),
        category_share=(
            1.0 if txn.merchant_category is None
            else _share(user.category_count, user.txn_count, cold)
        ),
    )
