"""Single source of truth for feature computation.

Every consumer of features - live scoring, retraining, initial training -
MUST compute them through this module. Never recompute features elsewhere.
"""
from collections.abc import Iterable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

FEATURE_ORDER: list[str] = [
    "amount",
    "amount_deviation",
    "is_new_location",
    "is_flagged_device",
    "velocity_2m",
]

VELOCITY_WINDOW = timedelta(seconds=120)
FLAGGED_DEVICE_MIN_USERS = 3


@dataclass(frozen=True)
class UserAggregates:
    """Pre-computed history for one user, sourced either from SQL (serving)
    or from a DataFrame slice (training). Never computed inside compute_features."""
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


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def aggregates_from_history(history: Iterable[Any], now: datetime) -> UserAggregates:
    """Build UserAggregates from a user's prior transactions.

    `history` holds objects exposing `amount`, `location` and `created_at`
    (ORM rows or DataFrame itertuples). Only transactions strictly before
    `now` should be passed. Shared by serving and training so both derive
    aggregates identically; T-02 replaces the serving side with SQL.
    """
    amounts: list[float] = []
    locations: set[str] = set()
    recent = 0
    window_start = _as_utc(now) - VELOCITY_WINDOW
    for t in history:
        amounts.append(float(t.amount))
        locations.add(t.location)
        if _as_utc(t.created_at) >= window_start:
            recent += 1

    count = len(amounts)
    return UserAggregates(
        txn_count=count,
        avg_amount=sum(amounts) / count if count else 0.0,
        known_locations=frozenset(locations),
        count_last_2m=recent,
        is_cold_start=count == 0,
    )


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
    if aggregates.is_cold_start:
        amount_deviation = 1.0
        is_new_location = 0
    else:
        amount_deviation = amount / aggregates.avg_amount if aggregates.avg_amount > 0 else 1.0
        is_new_location = 0 if location in aggregates.known_locations else 1

    return FeatureVector(
        amount=float(amount),
        amount_deviation=float(amount_deviation),
        is_new_location=is_new_location,
        is_flagged_device=1 if device_user_count >= FLAGGED_DEVICE_MIN_USERS else 0,
        velocity_2m=aggregates.count_last_2m,
    )
