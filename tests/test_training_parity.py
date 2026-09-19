"""
Training reproduces serving exactly (T-16).

The audit trail stores the feature vector each live decision used. Retraining
recomputes features for labelled transactions through the same repository
queries, as of each transaction's own timestamp. The two must match field for
field - and the two training bugs this replaced must stay fixed.
"""
import random

import pytest

from app import models
from app.features import FEATURE_ORDER
from app.models import OutcomeSource
from scripts.retrain import build_features, load_labeled_data

CITIES = {"Lahore": (31.52, 74.36), "Karachi": (24.86, 67.00), "London": (51.51, -0.13)}


def _label_everything(db, is_fraud=lambda txn: False):
    db.expire_all()
    for txn in db.query(models.Transaction).all():
        db.add(models.TransactionOutcome(transaction_id=txn.id, is_fraud_confirmed=is_fraud(txn),
                                         source=OutcomeSource.ANALYST.value))
    db.commit()


def _audited_vector(db, txn_id):
    return db.query(models.DecisionAudit).filter_by(transaction_id=txn_id).one().feature_vector


class TestParity:
    def test_retraining_reproduces_every_served_feature(self, client, auth_headers, second_auth_headers,
                                                        db_session):
        rng = random.Random(3)
        for i in range(40):
            city = rng.choice(list(CITIES))
            payload = {
                "amount": rng.choice([12.5, 80.0, 250.0, 1200.0, 6000.0]),
                "location": city,
                "device_id": rng.choice(["d1", "d2", "shared"]),
                "merchant_id": rng.choice(["M5411-001", "M4829-003", None]),
                "merchant_category": rng.choice(["5411", "4829", None]),
                "channel": rng.choice(["pos", "web", "mobile", None]),
            }
            if rng.random() < 0.7:
                payload["latitude"], payload["longitude"] = CITIES[city]
            payload = {k: v for k, v in payload.items() if v is not None}
            headers = auth_headers if i % 3 else second_auth_headers
            assert client.post("/transactions/", json=payload, headers=headers).status_code == 200

        # Label half as fraud so merchant_fraud_rate has evidence to read.
        _label_everything(db_session, is_fraud=lambda txn: txn.id % 2 == 0)

        trained = build_features(db_session, load_labeled_data(db_session))
        assert len(trained) == 40
        for row in trained.itertuples(index=False):
            served = _audited_vector(db_session, row.id)
            recomputed = {name: getattr(row, name) for name in FEATURE_ORDER}
            assert recomputed == pytest.approx(served), row.id

    def test_merchant_labels_are_point_in_time_in_training(self, client, auth_headers, db_session):
        """A label confirmed after a transaction must not feed that transaction's features."""
        for _ in range(3):
            client.post("/transactions/", json={"amount": 50.0, "location": "Lahore", "device_id": "d1",
                                                "merchant_id": "M4829-003"}, headers=auth_headers)
        _label_everything(db_session, is_fraud=lambda txn: True)     # all confirmed now, afterwards

        trained = build_features(db_session, load_labeled_data(db_session))
        # Every label was confirmed after every transaction, so none was visible.
        assert list(trained["merchant_fraud_rate"]) == pytest.approx([0.01, 0.01, 0.01])


class TestMissingValuesFromDataFrames:
    def test_nan_is_missing_not_a_value(self):
        """pandas 3 reports a missing string as NaN, which passes `is not None`."""
        import math

        import pandas as pd

        from app.features import TransactionInput

        frame = pd.DataFrame([{"amount": 1.0, "location": "L", "device_id": "d", "merchant_id": "M1",
                               "merchant_category": "5411", "channel": "web", "latitude": 1.0, "longitude": 2.0},
                              {"amount": 1.0, "location": "L", "device_id": "d", "merchant_id": None,
                               "merchant_category": None, "channel": None, "latitude": None, "longitude": None}])
        row = list(frame.itertuples(index=False))[1]
        assert isinstance(row.latitude, float) and math.isnan(row.latitude)     # the hazard is real
        candidate = TransactionInput.from_request(row, at=None)
        assert (candidate.merchant_id, candidate.merchant_category, candidate.channel,
                candidate.latitude, candidate.longitude) == (None, None, None, None, None)


class TestTrainingBugsStayFixed:
    def test_history_includes_unlabelled_transactions(self, client, auth_headers, db_session):
        """Before T-16, training rebuilt history from labelled rows only."""
        for _ in range(4):
            client.post("/transactions/", json={"amount": 40.0, "location": "Lahore", "device_id": "d1"},
                        headers=auth_headers)
        last = client.post("/transactions/", json={"amount": 40.0, "location": "Lahore", "device_id": "d1"},
                           headers=auth_headers).json()
        db_session.add(models.TransactionOutcome(transaction_id=last["id"], is_fraud_confirmed=False,
                                                 source=OutcomeSource.ANALYST.value))
        db_session.commit()

        trained = build_features(db_session, load_labeled_data(db_session))
        assert len(trained) == 1                                     # one labelled row...
        assert trained.iloc[0]["velocity_2m"] == 4                   # ...whose history is all four before it
        assert trained.iloc[0]["txn_count_1h"] == 4

    def test_device_sharing_cannot_see_later_users(self, client, auth_headers, second_auth_headers, db_session):
        """Before T-16, training counted device users across the whole dataset, future included."""
        first = client.post("/transactions/", json={"amount": 40.0, "location": "Lahore", "device_id": "shared"},
                            headers=auth_headers).json()
        # Other customers use the device only *after* the labelled transaction.
        client.post("/transactions/", json={"amount": 40.0, "location": "Lahore", "device_id": "shared"},
                    headers=second_auth_headers)
        db_session.add(models.TransactionOutcome(transaction_id=first["id"], is_fraud_confirmed=False,
                                                 source=OutcomeSource.ANALYST.value))
        db_session.commit()

        trained = build_features(db_session, load_labeled_data(db_session))
        assert trained.iloc[0]["users_per_device"] == 0              # nobody had used it yet
        assert trained.iloc[0]["is_flagged_device"] == 0
