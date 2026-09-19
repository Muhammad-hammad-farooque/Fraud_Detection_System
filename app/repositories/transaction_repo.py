"""Bounded SQL reads for the scoring path.

Scoring used to load a user's entire transaction history into Python on every
request (A7). Every aggregate here is computed by the database instead, so the
work per score does not grow with the customer's history: four statements per
decision, whatever the history size.

Every query is point-in-time: it sees only rows strictly before the candidate's
timestamp. Serving evaluates them at request time; retraining evaluates them at
each historical transaction's own timestamp. Because both go through these
same functions, training features cannot drift from serving features, and no
training row can see the future (T-16).

Statements are built once and reused with bound parameters (T-18). Rebuilding
the 35-expression user aggregate on every call cost four times more than
running it, which made retraining on 50,000 rows take most of an hour.
"""
from bisect import bisect_left
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import lru_cache

from sqlalchemy import and_, bindparam, case, distinct, extract, func, literal, select
from sqlalchemy.orm import Session

from .. import models
from ..features import (
    VELOCITY_WINDOW,
    VELOCITY_WINDOWS,
    DeviceAggregates,
    PopulationAggregates,
    TransactionInput,
    UserAggregates,
    amount_band,
    as_utc,
    typical_hours,
)

Txn = models.Transaction
DAY_24H = next(span for name, span in VELOCITY_WINDOWS if name == "24h")


def _count_where(condition):
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def _sum_where(condition, value):
    return func.coalesce(func.sum(case((condition, value), else_=0.0)), 0.0)


# ── Per-customer history ────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _user_statement(has_device: bool, has_merchant: bool, has_category: bool):
    """The aggregate over one customer's history, parameterised. Candidate
    fields that are absent compile to a constant 0, so there are eight
    variants, each built once."""
    def matches(column, name, present):
        return _count_where(column == bindparam(name)) if present else literal(0)

    return select(
        func.count(Txn.id),
        func.avg(Txn.amount),
        func.avg(Txn.amount * Txn.amount),
        func.max(Txn.amount),
        _count_where(Txn.created_at >= bindparam("since_2m")),
        func.count(distinct(case((Txn.created_at >= bindparam("since_24h"), Txn.location), else_=None))),
        func.count(distinct(Txn.device_id)),
        _count_where(Txn.location == bindparam("location")),
        matches(Txn.device_id, "device_id", has_device),
        matches(Txn.merchant_id, "merchant_id", has_merchant),
        matches(Txn.merchant_category, "merchant_category", has_category),
        _count_where(extract("hour", Txn.created_at).in_(bindparam("hours", expanding=True))),
        _count_where(and_(Txn.amount >= bindparam("band_low"), Txn.amount < bindparam("band_high"))),
        select(models.User.created_at).where(models.User.id == bindparam("user_id")).scalar_subquery(),
        *[_count_where(Txn.created_at >= bindparam(f"since_{name}")) for name, _ in VELOCITY_WINDOWS],
        *[_sum_where(Txn.created_at >= bindparam(f"since_{name}"), Txn.amount) for name, _ in VELOCITY_WINDOWS],
    ).where(Txn.user_id == bindparam("user_id"), Txn.created_at < bindparam("now"))


_PREVIOUS = (
    select(Txn.created_at, Txn.latitude, Txn.longitude)
    .where(Txn.user_id == bindparam("user_id"), Txn.created_at < bindparam("now"))
    .order_by(Txn.created_at.desc(), Txn.id.desc())
    .limit(1)
)


def get_user_aggregates(db: Session, user_id: int, txn: TransactionInput) -> UserAggregates:
    """Two statements: one aggregate over the user's history, one for the
    previous transaction. Must NOT materialise the user's transaction rows."""
    now = as_utc(txn.at)
    band_low, band_high = amount_band(txn.amount)
    params = {
        "user_id": user_id,
        "now": now,
        "since_2m": now - VELOCITY_WINDOW,
        "since_24h": now - DAY_24H,
        "location": txn.location,
        "device_id": txn.device_id,
        "merchant_id": txn.merchant_id,
        "merchant_category": txn.merchant_category,
        "hours": typical_hours(now.hour),
        "band_low": band_low,
        "band_high": band_high,
        **{f"since_{name}": now - span for name, span in VELOCITY_WINDOWS},
    }
    statement = _user_statement(txn.device_id is not None, txn.merchant_id is not None,
                                txn.merchant_category is not None)
    (count, avg_amount, avg_square, max_amount, count_2m, distinct_24h, distinct_devices,
     location_count, device_count, merchant_count, category_count, hour_count,
     band_count, created_at, *windows) = db.execute(statement, params).one()
    count = count or 0
    avg_amount = float(avg_amount or 0.0)
    variance = max(float(avg_square or 0.0) - avg_amount ** 2, 0.0) if count > 1 else 0.0

    previous = db.execute(_PREVIOUS, {"user_id": user_id, "now": now}).first()

    return UserAggregates(
        txn_count=count,
        avg_amount=avg_amount,
        known_locations=frozenset({txn.location}) if location_count else frozenset(),
        count_last_2m=int(count_2m),
        is_cold_start=count == 0,
        window_counts=tuple(int(v) for v in windows[:len(VELOCITY_WINDOWS)]),
        window_sums=tuple(float(v) for v in windows[len(VELOCITY_WINDOWS):]),
        amount_std=variance ** 0.5,
        max_amount=float(max_amount or 0.0),
        distinct_locations_24h=int(distinct_24h),
        distinct_devices=int(distinct_devices),
        location_count=int(location_count),
        device_count=int(device_count),
        merchant_count=int(merchant_count),
        category_count=int(category_count),
        typical_hour_count=int(hour_count),
        amount_band_count=int(band_count),
        previous_at=previous.created_at if previous else None,
        previous_latitude=previous.latitude if previous else None,
        previous_longitude=previous.longitude if previous else None,
        account_created_at=created_at,
    )


# ── Device ──────────────────────────────────────────────────────────────────

_DEVICE = (
    select(func.count(distinct(Txn.user_id)), func.min(Txn.created_at))
    .where(Txn.device_id == bindparam("device_id"), Txn.created_at < bindparam("now"))
)


def get_device_aggregates(db: Session, device_id: str | None, now: datetime) -> DeviceAggregates:
    """One statement: how many customers have used the device, and since when."""
    if device_id is None:
        return DeviceAggregates()
    user_count, first_seen = db.execute(_DEVICE, {"device_id": device_id, "now": as_utc(now)}).one()
    return DeviceAggregates(user_count=int(user_count or 0), first_seen_at=first_seen)


# ── Population ──────────────────────────────────────────────────────────────

# How finely the population's amount distribution is kept: percentiles are
# resolved to 1/QUANTILE_POINTS (0.1%).
QUANTILE_POINTS = 1000
_SNAPSHOT_LIMIT = 512
_snapshots: "OrderedDict[tuple, tuple[float, ...]]" = OrderedDict()


def clear_population_snapshots() -> None:
    """Forget cached snapshots. Needed only when history before a day changes,
    which in practice means tests and data generation."""
    _snapshots.clear()


def population_quantiles(db: Session, day_start: datetime) -> tuple[float, ...]:
    """The amount distribution of every transaction before `day_start`, as
    QUANTILE_POINTS order statistics.

    History before the start of a day never changes once that day has begun,
    so each day's snapshot is computed once and cached: serving pays for it on
    the first decision of the day, retraining once per distinct day in the
    data. Ranking an amount against it is then a binary search, instead of the
    count over every prior row that made retraining quadratic.
    """
    key = (str(db.get_bind().url), day_start)
    if key in _snapshots:
        _snapshots.move_to_end(key)
        return _snapshots[key]
    amounts = db.execute(
        select(Txn.amount).where(Txn.created_at < day_start).order_by(Txn.amount)
    ).scalars().all()
    n = len(amounts)
    quantiles = tuple(float(amounts[(k * n) // QUANTILE_POINTS]) for k in range(QUANTILE_POINTS)) if n else ()
    _snapshots[key] = quantiles
    if len(_snapshots) > _SNAPSHOT_LIMIT:
        _snapshots.popitem(last=False)
    return quantiles


_MERCHANT_OUTCOMES = (
    select(func.count(models.TransactionOutcome.id),
           func.coalesce(func.sum(case((models.TransactionOutcome.is_fraud_confirmed.is_(True), 1), else_=0)), 0))
    .join(Txn, Txn.id == models.TransactionOutcome.transaction_id)
    .where(Txn.merchant_id == bindparam("merchant_id"),
           Txn.created_at < bindparam("now"),
           models.TransactionOutcome.confirmed_at < bindparam("now"))
)


def get_population_aggregates(db: Session, txn: TransactionInput) -> PopulationAggregates:
    """Where the amount sits among everyone's, as of the start of the candidate's
    UTC day, and what analysts had confirmed about the merchant before `now` -
    so training never reads a label that did not exist yet. One statement for
    the merchant, plus the day's snapshot the first time a day is seen."""
    now = as_utc(txn.at)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    quantiles = population_quantiles(db, day_start)
    percentile = bisect_left(quantiles, txn.amount) / len(quantiles) if quantiles else None

    if txn.merchant_id is None:
        return PopulationAggregates(amount_percentile=percentile)
    labelled, fraud = db.execute(_MERCHANT_OUTCOMES, {"merchant_id": txn.merchant_id, "now": now}).one()
    return PopulationAggregates(amount_percentile=percentile,
                                merchant_labelled=int(labelled), merchant_fraud=int(fraud))


def get_device_user_count(db: Session, device_id: str) -> int:
    """COUNT(DISTINCT user_id) for this device across all time. Returns the
    count, not a bool, so callers can threshold it themselves."""
    return db.execute(
        select(func.count(func.distinct(Txn.user_id))).where(Txn.device_id == device_id)
    ).scalar_one()
