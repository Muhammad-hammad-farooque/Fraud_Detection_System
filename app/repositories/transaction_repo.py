"""Bounded SQL reads for the scoring path.

Scoring used to load a user's entire transaction history into Python on every
request (A7). Every aggregate here is computed by the database instead, so the
work per score no longer grows with the customer's history.
"""
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .. import models
from ..features import VELOCITY_WINDOW, UserAggregates


def get_user_aggregates(
    db: Session, user_id: int, now: datetime, location: str
) -> UserAggregates:
    """One round trip. Uses COUNT, AVG and a bounded window filter.
    Must NOT materialise the user's transaction rows.

    `location` is the location being scored: its membership is resolved with a
    second EXISTS query, so `known_locations` comes back holding at most that
    one value. compute_features only asks whether the location is in the set,
    never what else is in it.
    """
    window_start = now - VELOCITY_WINDOW

    txn = models.Transaction
    aggregates = db.execute(
        select(
            func.count(txn.id),
            func.avg(txn.amount),
            func.sum(case((txn.created_at >= window_start, 1), else_=0)),
        ).where(txn.user_id == user_id)
    ).one()
    txn_count, avg_amount, count_last_2m = aggregates

    location_seen = db.execute(
        select(
            select(txn.id)
            .where(txn.user_id == user_id, txn.location == location)
            .exists()
        )
    ).scalar_one()

    return UserAggregates(
        txn_count=txn_count or 0,
        avg_amount=float(avg_amount or 0.0),
        known_locations=frozenset({location}) if location_seen else frozenset(),
        count_last_2m=int(count_last_2m or 0),
        is_cold_start=not txn_count,
    )


def get_device_user_count(db: Session, device_id: str) -> int:
    """COUNT(DISTINCT user_id) for this device. Returns the count, not a bool,
    so callers can threshold it themselves."""
    txn = models.Transaction
    return db.execute(
        select(func.count(func.distinct(txn.user_id))).where(txn.device_id == device_id)
    ).scalar_one()
