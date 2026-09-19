"""
Synthetic training data — realistic, hard, and quarantined (T-18).
"""
import statistics
from collections import Counter, defaultdict
from datetime import timedelta

import pytest
from sklearn.metrics import roc_auc_score

from app import models
from app.features import haversine_km
from app.schemas import TransactionCreate
from scripts.generate_data import (
    CITIES,
    FRAUD_ORIGINS,
    SYNTHETIC_DOMAIN,
    GeneratorConfig,
    NotASyntheticDatabase,
    generate,
)

SMALL = GeneratorConfig(transactions=20_000, users=500, seed=7)


@pytest.fixture
def generated(db_session):
    report = generate(db_session, SMALL)
    db_session.expire_all()
    rows = db_session.query(models.Transaction, models.TransactionOutcome).join(
        models.TransactionOutcome, models.TransactionOutcome.transaction_id == models.Transaction.id).all()
    return report, rows


def _split(rows):
    fraud = [t for t, o in rows if o.is_fraud_confirmed]
    legit = [t for t, o in rows if not o.is_fraud_confirmed]
    return fraud, legit


class TestVolumeAndRarity:
    def test_writes_the_requested_number_of_transactions(self, generated):
        report, rows = generated
        assert report.transactions == len(rows) == SMALL.transactions

    def test_fraud_is_rare(self, generated):
        report, rows = generated
        fraud, _ = _split(rows)
        assert 0.005 <= len(fraud) / len(rows) <= 0.01
        assert report.fraud == len(fraud)

    def test_every_pattern_is_planted(self, generated):
        report, _ = generated
        assert {"card_testing", "account_takeover", "device_ring", "subtle"} <= set(report.patterns)

    def test_default_scale_meets_the_spec(self):
        assert GeneratorConfig().transactions >= 50_000
        assert 0.005 <= GeneratorConfig().fraud_rate <= 0.01


class TestHardness:
    def test_amounts_overlap(self, generated):
        """A9: amount alone must not separate the classes."""
        _, rows = generated
        amounts = [t.amount for t, _ in rows]
        labels = [int(o.is_fraud_confirmed) for _, o in rows]
        assert roc_auc_score(labels, amounts) < 0.85
        # A secondary check: a large share of fraud sits inside the range
        # legitimate customers spend in. (The bar is ours, not the spec's; the
        # AUC above is the spec's "separability must not be trivial".)
        fraud, legit = _split(rows)
        legit_p95 = sorted(t.amount for t in legit)[int(len(legit) * 0.95)]
        assert sum(t.amount <= legit_p95 for t in fraud) / len(fraud) > 0.4

    def test_no_single_fingerprint_belongs_to_fraud_alone(self, generated):
        """Every attack signal also occurs in legitimate data."""
        _, rows = generated
        fraud, legit = _split(rows)
        assert any(t.merchant_category in ("5815", "5816") and t.amount < 5 for t in legit)    # 0.99 apps
        assert any(t.merchant_category == "4829" for t in legit)                                 # remittances
        assert any(t.location in FRAUD_ORIGINS for t in legit)                                   # trips and VPNs
        assert any(t.ip_address.split(".")[0] in ("185", "45", "91", "103") for t in legit)      # VPN users
        assert any(t.device_id.startswith("dev-borrowed") for t in legit)                        # borrowed devices

    def test_some_fraud_looks_ordinary(self, generated):
        _, rows = generated
        fraud, _ = _split(rows)
        own_device = [t for t in fraud if t.device_id.startswith("dev-")]
        home_city = [t for t in fraud if t.location not in FRAUD_ORIGINS]
        assert own_device, "some fraud should come from the victim's own device"
        assert len(home_city) / len(fraud) > 0.15


class TestTemporalStructure:
    def test_diurnal_rhythm_in_local_time(self, generated):
        _, rows = generated
        _, legit = _split(rows)
        by_local_hour = Counter()
        for t in legit:
            offset = round(CITIES[t.location][1] / 15)
            by_local_hour[(t.created_at.hour + offset) % 24] += 1
        day = sum(by_local_hour[h] for h in range(10, 21))
        night = sum(by_local_hour[h] for h in range(1, 6))
        assert day / 11 > 5 * night / 5                     # daytime hours far busier per hour

    def test_weekend_mix_differs(self, generated):
        _, rows = generated
        _, legit = _split(rows)
        def share(txns, mcc):
            return sum(t.merchant_category == mcc for t in txns) / len(txns)
        weekend = [t for t in legit if t.created_at.weekday() >= 5]
        weekday = [t for t in legit if t.created_at.weekday() < 5]
        assert share(weekend, "5812") > share(weekday, "5812")          # more restaurants at weekends

    def test_window_ends_before_labels_would_be_immature(self, generated):
        """T-06 drops labels younger than 90 days; the window must end before that."""
        from datetime import datetime, timezone

        _, rows = generated
        newest = max(t.created_at for t, _ in rows).replace(tzinfo=timezone.utc)
        assert datetime.now(timezone.utc) - newest > timedelta(days=89)


class TestAttackFingerprints:
    def test_card_testing_bursts(self, generated):
        _, rows = generated
        by_card = defaultdict(list)
        for t, o in rows:
            if o.notes == "card_testing":
                by_card[t.card_token].append(t)
        burst = max(by_card.values(), key=len)
        tiny = sorted((t for t in burst if t.amount < 5), key=lambda t: t.created_at)
        assert len(tiny) >= 4
        assert tiny[-1].created_at - tiny[0].created_at < timedelta(minutes=20)

    def test_account_takeover_often_travels_impossibly(self, generated):
        _, rows = generated
        history = defaultdict(list)
        for t, o in sorted(rows, key=lambda r: r[0].created_at):
            history[t.user_id].append((t, o))
        impossible = 0
        for events in history.values():
            for (prev, _), (cur, outcome) in zip(events, events[1:]):
                if outcome.notes != "account_takeover" or prev.latitude is None:
                    continue
                km = haversine_km(prev.latitude, prev.longitude, cur.latitude, cur.longitude)
                hours = max((cur.created_at - prev.created_at).total_seconds(), 1) / 3600
                impossible += km > 100 and km / hours > 900
        assert impossible > 0

    def test_device_ring_spans_several_customers(self, generated):
        _, rows = generated
        ring_users = defaultdict(set)
        for t, o in rows:
            if o.notes == "device_ring":
                ring_users[t.device_id].add(t.user_id)
        assert max(len(users) for users in ring_users.values()) >= 4


class TestPopulatedAndValid:
    def test_every_payment_context_field_is_filled_and_valid(self, generated):
        _, rows = generated
        for t, _ in rows[:3000]:
            payload = {k: getattr(t, k) for k in ("location", "amount", "device_id", "merchant_id",
                                                  "merchant_category", "currency", "channel", "ip_address",
                                                  "card_token", "external_txn_id", "latitude", "longitude")}
            assert all(v is not None for v in payload.values())
            TransactionCreate(**payload)

    def test_accounts_exist_before_they_transact(self, generated, db_session):
        _, rows = generated
        created = {u.id: u.created_at for u in db_session.query(models.User).all()}
        assert all(t.created_at >= created[t.user_id] for t, _ in rows)


class TestLabelsAndQuarantine:
    def test_every_transaction_has_a_synthetic_label(self, generated):
        _, rows = generated
        assert {o.source for _, o in rows} == {"SYNTHETIC"}

    def test_labels_are_confirmed_after_the_fact(self, generated):
        _, rows = generated
        assert all(o.confirmed_at > t.created_at for t, o in rows)

    def test_accounts_are_marked_synthetic(self, generated, db_session):
        assert all(u.email.endswith("@" + SYNTHETIC_DOMAIN) for u in db_session.query(models.User).all())

    def test_refuses_a_database_with_real_accounts(self, client, auth_headers, db_session):
        with pytest.raises(NotASyntheticDatabase, match="real account"):
            generate(db_session, GeneratorConfig(transactions=100, users=10))

    def test_refuses_to_overwrite_without_reset(self, db_session):
        generate(db_session, GeneratorConfig(transactions=500, users=20))
        with pytest.raises(NotASyntheticDatabase, match="reset"):
            generate(db_session, GeneratorConfig(transactions=500, users=20))
        generate(db_session, GeneratorConfig(transactions=500, users=20), reset=True)
        assert db_session.query(models.Transaction).count() == 500

    def test_deterministic_for_a_seed(self, db_session):
        cfg = GeneratorConfig(transactions=800, users=40, seed=11)
        first = generate(db_session, cfg)
        snapshot = [(t.amount, t.device_id, t.created_at) for t in
                    db_session.query(models.Transaction).order_by(models.Transaction.id).limit(200)]
        second = generate(db_session, cfg, reset=True)
        again = [(t.amount, t.device_id, t.created_at) for t in
                 db_session.query(models.Transaction).order_by(models.Transaction.id).limit(200)]
        assert dict(first.patterns) == dict(second.patterns)
        assert statistics.mean(a for a, _, _ in snapshot) == statistics.mean(a for a, _, _ in again)
        assert [d for _, d, _ in snapshot] == [d for _, d, _ in again]
