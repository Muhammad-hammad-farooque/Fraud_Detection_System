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
"""
from datetime import datetime

from sqlalchemy import and_, case, distinct, extract, func, literal, select
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


def _count_where(condition):
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


def _sum_where(condition, value):
    return func.coalesce(func.sum(case((condition, value), else_=0.0)), 0.0)


def _count_if_set(column, value):
    """How many prior rows match `value`; zero when the candidate has no value."""
    return _count_where(column == value) if value is not None else literal(0)


def get_user_aggregates(db: Session, user_id: int, txn: TransactionInput) -> UserAggregates:
    """Two statements: one aggregate over the user's history, one for the
    previous transaction. Must NOT materialise the user's transaction rows."""
    now = as_utc(txn.at)
    band_low, band_high = amount_band(txn.amount)

    window_counts = [_count_where(Txn.created_at >= now - span) for _, span in VELOCITY_WINDOWS]
    window_sums = [_sum_where(Txn.created_at >= now - span, Txn.amount) for _, span in VELOCITY_WINDOWS]
    last_24h = next(span for name, span in VELOCITY_WINDOWS if name == "24h")
    account_created_at = (
        select(models.User.created_at).where(models.User.id == user_id).scalar_subquery()
    )

    row = db.execute(
        select(
            func.count(Txn.id),
            func.avg(Txn.amount),
            func.avg(Txn.amount * Txn.amount),
            func.max(Txn.amount),
            _count_where(Txn.created_at >= now - VELOCITY_WINDOW),
            func.count(distinct(case((Txn.created_at >= now - last_24h, Txn.location), else_=None))),
            func.count(distinct(Txn.device_id)),
            _count_where(Txn.location == txn.location),
            _count_if_set(Txn.device_id, txn.device_id),
            _count_if_set(Txn.merchant_id, txn.merchant_id),
            _count_if_set(Txn.merchant_category, txn.merchant_category),
            _count_where(extract("hour", Txn.created_at).in_(typical_hours(now.hour))),
            _count_where(and_(Txn.amount >= band_low, Txn.amount < band_high)),
            account_created_at,
            *window_counts,
            *window_sums,
        ).where(Txn.user_id == user_id, Txn.created_at < now)
    ).one()

    (count, avg_amount, avg_square, max_amount, count_2m, distinct_24h, distinct_devices,
     location_count, device_count, merchant_count, category_count, hour_count,
     band_count, created_at, *windows) = row
    count = count or 0
    avg_amount = float(avg_amount or 0.0)
    variance = max(float(avg_square or 0.0) - avg_amount ** 2, 0.0) if count > 1 else 0.0

    previous = db.execute(
        select(Txn.created_at, Txn.latitude, Txn.longitude)
        .where(Txn.user_id == user_id, Txn.created_at < now)
        .order_by(Txn.created_at.desc(), Txn.id.desc())
        .limit(1)
    ).first()

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


def get_device_aggregates(db: Session, device_id: str | None, now: datetime) -> DeviceAggregates:
    """One statement: how many customers have used the device, and since when."""
    if device_id is None:
        return DeviceAggregates()
    user_count, first_seen = db.execute(
        select(func.count(distinct(Txn.user_id)), func.min(Txn.created_at))
        .where(Txn.device_id == device_id, Txn.created_at < as_utc(now))
    ).one()
    return DeviceAggregates(user_count=int(user_count or 0), first_seen_at=first_seen)


def get_population_aggregates(db: Session, txn: TransactionInput) -> PopulationAggregates:
    """One statement: where the amount ranks among everyone's, and what analysts
    have confirmed about the merchant - only outcomes confirmed before `now`, so
    training never reads a label that did not exist yet."""
    now = as_utc(txn.at)
    prior = Txn.created_at < now
    rank = select(func.count(Txn.id)).where(prior, Txn.amount < txn.amount).scalar_subquery()
    total = select(func.count(Txn.id)).where(prior).scalar_subquery()

    if txn.merchant_id is None:
        rank_value, total_value = db.execute(select(rank, total)).one()
        return PopulationAggregates(amount_rank=int(rank_value), total_count=int(total_value))

    Outcome = models.TransactionOutcome
    at_merchant = (
        select(Outcome.is_fraud_confirmed)
        .join(Txn, Txn.id == Outcome.transaction_id)
        .where(Txn.merchant_id == txn.merchant_id, prior, Outcome.confirmed_at < now)
        .subquery()
    )
    labelled = select(func.count()).select_from(at_merchant).scalar_subquery()
    fraud = (
        select(func.coalesce(func.sum(case((at_merchant.c.is_fraud_confirmed.is_(True), 1), else_=0)), 0))
        .scalar_subquery()
    )
    rank_value, total_value, labelled_value, fraud_value = db.execute(
        select(rank, total, labelled, fraud)
    ).one()
    return PopulationAggregates(
        amount_rank=int(rank_value),
        total_count=int(total_value),
        merchant_labelled=int(labelled_value),
        merchant_fraud=int(fraud_value),
    )


def get_device_user_count(db: Session, device_id: str) -> int:
    """COUNT(DISTINCT user_id) for this device across all time. Returns the
    count, not a bool, so callers can threshold it themselves."""
    return db.execute(
        select(func.count(func.distinct(Txn.user_id))).where(Txn.device_id == device_id)
    ).scalar_one()
