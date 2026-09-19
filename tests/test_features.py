"""
Unit tests for app/features.py — the single feature computation path (T-01, T-16).

compute_features is pure, so every family is tested here with hand-built
aggregates. tests/test_repositories.py checks the SQL that produces them.
"""
import inspect
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app import features
from app.features import (
    BASELINE_FEATURES,
    FEATURE_ORDER,
    IMPOSSIBLE_MIN_KM,
    IMPOSSIBLE_SPEED_KMH,
    MERCHANT_PRIOR_RATE,
    MERCHANT_PRIOR_WEIGHT,
    UNKNOWN,
    DeviceAggregates,
    FeatureVector,
    PopulationAggregates,
    TransactionInput,
    UserAggregates,
    amount_band,
    compute_features,
    haversine_km,
    typical_hours,
)

AT = datetime(2026, 3, 4, 14, 30, tzinfo=timezone.utc)          # a Wednesday afternoon
LAHORE = (31.5204, 74.3587)
KARACHI = (24.8607, 67.0011)
LONDON = (51.5074, -0.1278)


def _txn(**overrides) -> TransactionInput:
    base = dict(amount=100.0, location="Lahore", at=AT, device_id="d1",
                merchant_id="M5411-001", merchant_category="5411", channel="pos",
                latitude=LAHORE[0], longitude=LAHORE[1])
    base.update(overrides)
    return TransactionInput(**base)


def _user(**overrides) -> UserAggregates:
    base = dict(
        txn_count=4, avg_amount=100.0, known_locations=frozenset({"Lahore"}), count_last_2m=0,
        is_cold_start=False, amount_std=20.0, max_amount=150.0, distinct_devices=1,
        location_count=4, device_count=4, merchant_count=2, category_count=3,
        typical_hour_count=3, amount_band_count=4,
        previous_at=AT - timedelta(hours=2), previous_latitude=LAHORE[0], previous_longitude=LAHORE[1],
        account_created_at=AT - timedelta(days=400),
    )
    base.update(overrides)
    return UserAggregates(**base)


COLD = UserAggregates(txn_count=0, avg_amount=0.0, known_locations=frozenset(), count_last_2m=0,
                      is_cold_start=True, account_created_at=AT - timedelta(days=1))


def _fv(txn=None, user=None, device=None, population=None) -> FeatureVector:
    return compute_features(txn or _txn(), user or _user(),
                            device or DeviceAggregates(user_count=1, first_seen_at=AT - timedelta(days=30)),
                            population or PopulationAggregates(amount_percentile=0.5))


# ── Schema ───────────────────────────────────────────────────────────────────

class TestSchema:
    def test_at_least_forty_features(self):
        assert len(FEATURE_ORDER) >= 40

    def test_feature_order_matches_vector_fields(self):
        assert [f.name for f in fields(FeatureVector)] == FEATURE_ORDER

    def test_no_duplicates(self):
        assert len(set(FEATURE_ORDER)) == len(FEATURE_ORDER)

    def test_baseline_features_lead_unchanged(self):
        assert FEATURE_ORDER[:5] == BASELINE_FEATURES == [
            "amount", "amount_deviation", "is_new_location", "is_flagged_device", "velocity_2m"]

    def test_to_frame_defaults_to_every_feature(self):
        frame = _fv().to_frame()
        assert list(frame.columns) == FEATURE_ORDER and frame.shape == (1, len(FEATURE_ORDER))

    def test_to_frame_can_serve_a_model_its_own_subset(self):
        subset = ["velocity_2m", "amount"]
        assert list(_fv().to_frame(columns=subset).columns) == subset

    def test_every_feature_is_numeric(self):
        for name, value in vars(_fv()).items():
            assert isinstance(value, (int, float)) and not isinstance(value, bool), name


class TestPurity:
    def test_signature_takes_no_database_or_clock(self):
        assert list(inspect.signature(compute_features).parameters) == ["txn", "user", "device", "population"]

    def test_never_reads_the_clock(self):
        with patch.object(features, "datetime", side_effect=AssertionError("clock used")):
            _fv()

    def test_deterministic(self):
        assert _fv() == _fv()


# ── Baseline (T-01 behaviour preserved) ──────────────────────────────────────

class TestBaseline:
    def test_cold_start_is_not_penalised(self):
        fv = _fv(user=COLD)
        assert fv.is_new_location == 0 and fv.amount_deviation == 1.0

    def test_amount_deviation(self):
        assert _fv(txn=_txn(amount=350.0)).amount_deviation == pytest.approx(3.5)

    def test_zero_average_amount_gives_neutral_deviation(self):
        assert _fv(user=_user(avg_amount=0.0)).amount_deviation == 1.0

    def test_new_and_known_location(self):
        assert _fv(txn=_txn(location="Quetta")).is_new_location == 1
        assert _fv().is_new_location == 0

    @pytest.mark.parametrize("count, flagged", [(2, 0), (3, 1)])
    def test_device_flag_threshold(self, count, flagged):
        assert _fv(device=DeviceAggregates(user_count=count)).is_flagged_device == flagged


# ── Velocity ─────────────────────────────────────────────────────────────────

class TestVelocity:
    def test_windows_are_copied_in_order(self):
        fv = _fv(user=_user(window_counts=(1, 2, 3, 4, 5, 6), window_sums=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0)))
        assert [fv.txn_count_1m, fv.txn_count_5m, fv.txn_count_1h,
                fv.txn_count_24h, fv.txn_count_7d, fv.txn_count_30d] == [1, 2, 3, 4, 5, 6]
        assert [fv.amount_sum_1m, fv.amount_sum_30d] == [10.0, 60.0]

    def test_velocity_2m_still_feeds_the_rule(self):
        assert _fv(user=_user(count_last_2m=5)).velocity_2m == 5


# ── Geo-velocity ─────────────────────────────────────────────────────────────

class TestGeoVelocity:
    def test_haversine_matches_a_known_distance(self):
        assert haversine_km(*LAHORE, *KARACHI) == pytest.approx(1030, abs=15)

    def test_distance_and_speed(self):
        user = _user(previous_at=AT - timedelta(hours=2),
                     previous_latitude=KARACHI[0], previous_longitude=KARACHI[1])
        fv = _fv(user=user)
        assert fv.km_from_previous == pytest.approx(haversine_km(*KARACHI, *LAHORE))
        assert fv.travel_speed_kmh == pytest.approx(fv.km_from_previous / 2)
        assert fv.is_impossible_travel == 0                          # ~515 km/h: a flight

    def test_impossible_travel_is_flagged(self):
        """Lahore, then London half an hour later: ~13,000 km/h."""
        user = _user(previous_at=AT - timedelta(minutes=30),
                     previous_latitude=LONDON[0], previous_longitude=LONDON[1])
        fv = _fv(user=user)
        assert fv.travel_speed_kmh > IMPOSSIBLE_SPEED_KMH
        assert fv.is_impossible_travel == 1

    def test_short_hops_are_not_teleportation(self):
        """GPS jitter seconds apart is fast but not far: never flagged."""
        near = (LAHORE[0] + 0.3, LAHORE[1])                              # ~33 km away
        user = _user(previous_at=AT - timedelta(seconds=10),
                     previous_latitude=near[0], previous_longitude=near[1])
        fv = _fv(user=user)
        assert fv.km_from_previous < IMPOSSIBLE_MIN_KM
        assert fv.is_impossible_travel == 0

    def test_simultaneous_transactions_do_not_divide_by_zero(self):
        user = _user(previous_at=AT, previous_latitude=LONDON[0], previous_longitude=LONDON[1])
        fv = _fv(user=user)
        assert fv.is_impossible_travel == 1 and fv.seconds_since_previous == 0.0

    @pytest.mark.parametrize("txn_coords, previous_coords", [
        ((None, None), LAHORE),
        (LAHORE, (None, None)),
    ])
    def test_missing_coordinates_are_unknown_not_zero(self, txn_coords, previous_coords):
        fv = _fv(txn=_txn(latitude=txn_coords[0], longitude=txn_coords[1]),
                 user=_user(previous_latitude=previous_coords[0], previous_longitude=previous_coords[1]))
        assert fv.km_from_previous == UNKNOWN and fv.travel_speed_kmh == UNKNOWN
        assert fv.is_impossible_travel == 0
        assert fv.seconds_since_previous == pytest.approx(7200.0)       # time is still known

    def test_no_previous_transaction_is_unknown(self):
        fv = _fv(user=COLD)
        assert fv.km_from_previous == fv.travel_speed_kmh == fv.seconds_since_previous == UNKNOWN

    def test_distinct_locations_24h(self):
        assert _fv(user=_user(distinct_locations_24h=4)).distinct_locations_24h == 4


# ── Temporal ─────────────────────────────────────────────────────────────────

class TestTemporal:
    def test_hour_and_day_come_from_the_scored_instant(self):
        fv = _fv()
        assert (fv.hour_of_day, fv.day_of_week) == (14, 2)              # 14:30 UTC, Wednesday

    @pytest.mark.parametrize("hour, night", [(0, 1), (5, 1), (6, 0), (23, 0)])
    def test_is_night(self, hour, night):
        assert _fv(txn=_txn(at=AT.replace(hour=hour))).is_night == night

    def test_account_age(self):
        assert _fv().account_age_days == pytest.approx(400.0)

    def test_account_age_unknown_without_a_signup_date(self):
        assert _fv(user=_user(account_created_at=None)).account_age_days == UNKNOWN

    def test_account_age_never_negative(self):
        assert _fv(user=_user(account_created_at=AT + timedelta(days=1))).account_age_days == 0.0

    def test_naive_stored_timestamps_are_treated_as_utc(self):
        naive = _user(previous_at=(AT - timedelta(hours=1)).replace(tzinfo=None))
        assert _fv(user=naive).seconds_since_previous == pytest.approx(3600.0)


# ── Amount shape ─────────────────────────────────────────────────────────────

class TestAmountShape:
    def test_zscore(self):
        assert _fv(txn=_txn(amount=160.0)).amount_zscore == pytest.approx(3.0)   # (160-100)/20

    @pytest.mark.parametrize("user", [COLD, _user(amount_std=0.0)], ids=["cold", "no spread"])
    def test_zscore_is_zero_without_spread(self, user):
        assert _fv(user=user).amount_zscore == 0.0

    def test_population_percentile(self):
        assert _fv(population=PopulationAggregates(amount_percentile=0.9)).amount_population_percentile == 0.9

    def test_percentile_is_the_median_with_no_population(self):
        assert _fv(population=PopulationAggregates()).amount_population_percentile == 0.5

    @pytest.mark.parametrize("amount, round_", [(500.0, 1), (5000.0, 1), (100.0, 1),
                                                (499.99, 0), (50.0, 0), (150.0, 0), (0.0, 0)])
    def test_round_amount(self, amount, round_):
        assert _fv(txn=_txn(amount=amount)).is_round_amount == round_

    def test_ratio_to_lifetime_max(self):
        assert _fv(txn=_txn(amount=300.0)).ratio_to_lifetime_max == pytest.approx(2.0)

    def test_ratio_is_neutral_on_cold_start(self):
        assert _fv(user=COLD).ratio_to_lifetime_max == 1.0


# ── Device ───────────────────────────────────────────────────────────────────

class TestDevice:
    def test_device_age_and_sharing(self):
        fv = _fv(device=DeviceAggregates(user_count=4, first_seen_at=AT - timedelta(days=12)))
        assert fv.device_age_days == pytest.approx(12.0)
        assert fv.users_per_device == 4

    def test_a_never_seen_device_is_age_zero(self):
        assert _fv(device=DeviceAggregates()).device_age_days == 0.0

    def test_devices_per_user(self):
        assert _fv(user=_user(distinct_devices=3)).devices_per_user == 3

    def test_first_use_of_a_device(self):
        assert _fv(user=_user(device_count=0)).is_first_use_of_device == 1
        assert _fv().is_first_use_of_device == 0

    def test_first_use_is_not_flagged_on_cold_start(self):
        assert _fv(user=COLD).is_first_use_of_device == 0


# ── Merchant ─────────────────────────────────────────────────────────────────

class TestMerchant:
    def test_rate_without_evidence_is_the_prior(self):
        assert _fv().merchant_fraud_rate == pytest.approx(MERCHANT_PRIOR_RATE)

    def test_rate_is_smoothed_toward_the_prior(self):
        population = PopulationAggregates(merchant_labelled=10, merchant_fraud=5)
        expected = (5 + MERCHANT_PRIOR_WEIGHT * MERCHANT_PRIOR_RATE) / (10 + MERCHANT_PRIOR_WEIGHT)
        assert _fv(population=population).merchant_fraud_rate == pytest.approx(expected)

    def test_one_label_does_not_make_a_merchant_fraudulent(self):
        population = PopulationAggregates(merchant_labelled=1, merchant_fraud=1)
        assert _fv(population=population).merchant_fraud_rate == pytest.approx(0.1)   # not 1.0

    def test_first_time_merchant_and_category(self):
        fv = _fv(user=_user(merchant_count=0, category_count=0))
        assert (fv.is_first_time_merchant, fv.is_new_category) == (1, 1)
        assert (_fv().is_first_time_merchant, _fv().is_new_category) == (0, 0)

    def test_unknown_merchant_is_not_a_first_visit(self):
        fv = _fv(txn=_txn(merchant_id=None, merchant_category=None), user=_user(merchant_count=0, category_count=0))
        assert (fv.is_first_time_merchant, fv.is_new_category) == (0, 0)

    @pytest.mark.parametrize("channel, cnp", [("web", 1), ("mobile", 1), ("pos", 0), ("atm", 0), (None, 0)])
    def test_card_not_present(self, channel, cnp):
        assert _fv(txn=_txn(channel=channel)).is_card_not_present == cnp


# ── Behavioural ──────────────────────────────────────────────────────────────

class TestBehavioural:
    def test_shares_are_fractions_of_history(self):
        fv = _fv(user=_user(txn_count=4, typical_hour_count=1, location_count=2,
                            amount_band_count=3, category_count=4))
        assert (fv.typical_hour_share, fv.location_share, fv.amount_band_share, fv.category_share) == \
            (0.25, 0.5, 0.75, 1.0)

    def test_cold_start_reads_as_fully_typical(self):
        fv = _fv(user=COLD)
        assert fv.typical_hour_share == fv.location_share == fv.amount_band_share == fv.category_share == 1.0

    def test_no_category_is_not_unusual(self):
        assert _fv(txn=_txn(merchant_category=None), user=_user(category_count=0)).category_share == 1.0


class TestHelpers:
    @pytest.mark.parametrize("hour, expected", [(12, [10, 11, 12, 13, 14]), (0, [0, 1, 2, 22, 23]),
                                                (23, [0, 1, 21, 22, 23])])
    def test_typical_hours_wrap_round_midnight(self, hour, expected):
        assert typical_hours(hour) == expected

    @pytest.mark.parametrize("amount, band", [(0.5, (0.0, 1.0)), (1.0, (1.0, 10.0)), (99.99, (10.0, 100.0)),
                                              (100.0, (100.0, 1000.0)), (5000.0, (1000.0, 10000.0))])
    def test_amount_bands(self, amount, band):
        assert amount_band(amount) == band


class TestColdStartEndToEnd:
    def test_a_first_ever_transaction_trips_no_behavioural_flag(self):
        fv = _fv(user=COLD, device=DeviceAggregates(), population=PopulationAggregates())
        flags = [fv.is_new_location, fv.is_first_use_of_device, fv.is_first_time_merchant,
                 fv.is_new_category, fv.is_impossible_travel]
        assert flags == [0, 0, 0, 0, 0]


class TestMissingValues:
    @pytest.mark.parametrize("value, missing", [
        (None, True), (float("nan"), True), ("", False), ("M1", False), (0, False), (0.0, False),
        (["a", "b"], False),            # not a scalar: never treated as "missing"
    ])
    def test_is_missing(self, value, missing):
        assert features._is_missing(value) is missing
