"""
Unit tests for the retraining split — chronological, with label maturity (T-06).
"""
import inspect
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts import retrain
from scripts.retrain import (
    LABEL_MATURITY_DAYS,
    apply_label_maturity,
    chronological_split,
)

NOW = datetime.now(timezone.utc)


def _frame(offsets_days, fraud=None):
    """One row per offset, oldest first. Offsets are days before now."""
    rows = [
        {
            "id": i,
            "user_id": 1,
            "amount": 100.0 + i,
            "location": "Lahore",
            "device_id": "d1",
            "created_at": NOW - timedelta(days=days),
            "fraud": (fraud[i] if fraud else i % 2),
        }
        for i, days in enumerate(offsets_days)
    ]
    return pd.DataFrame(rows)


class TestChronologicalSplit:
    def test_no_temporal_overlap(self):
        df = _frame(list(range(200, 0, -1)))
        train, test = chronological_split(df)
        assert train["created_at"].max() < test["created_at"].min()

    def test_every_training_row_precedes_every_test_row(self):
        df = _frame(list(range(100, 0, -1)))
        train, test = chronological_split(df)
        for train_ts in train["created_at"]:
            assert all(train_ts < test_ts for test_ts in test["created_at"])

    def test_split_respects_the_requested_fraction(self):
        df = _frame(list(range(100, 0, -1)))
        train, test = chronological_split(df, test_fraction=0.2)
        assert len(test) == pytest.approx(20, abs=2)
        assert len(train) + len(test) == len(df)

    def test_unsorted_input_is_sorted_first(self):
        df = _frame(list(range(100, 0, -1))).sample(frac=1, random_state=7)
        train, test = chronological_split(df)
        assert train["created_at"].is_monotonic_increasing
        assert train["created_at"].max() < test["created_at"].min()

    def test_rows_sharing_the_boundary_timestamp_stay_together(self):
        """A tie at the cut must not straddle both sides."""
        shared = NOW - timedelta(days=5)
        df = pd.DataFrame([
            {"id": i, "user_id": 1, "amount": 1.0, "location": "L", "device_id": "d",
             "created_at": (NOW - timedelta(days=10 - i) if i < 4 else shared),
             "fraud": i % 2}
            for i in range(10)
        ])
        train, test = chronological_split(df)
        assert train["created_at"].max() < test["created_at"].min()
        assert len(train) + len(test) == len(df)

    def test_no_random_split_remains_in_the_module(self):
        """A2: train_test_split on time-ordered data trains on the future."""
        source = inspect.getsource(retrain)
        assert "train_test_split" not in source


class TestLabelMaturity:
    def test_immature_labels_are_dropped(self):
        df = _frame([200, 150, 100, 10, 1])
        mature = apply_label_maturity(df, maturity_days=90)
        assert list(mature["id"]) == [0, 1, 2]

    def test_default_window_is_ninety_days(self):
        assert LABEL_MATURITY_DAYS == 90
        df = _frame([200, 10])
        assert list(apply_label_maturity(df)["id"]) == [0]

    def test_boundary_row_is_kept(self):
        df = _frame([91, 89])
        assert list(apply_label_maturity(df, maturity_days=90)["id"]) == [0]

    def test_empty_frame_is_returned_unchanged(self):
        assert apply_label_maturity(pd.DataFrame()).empty

    def test_all_immature_yields_nothing_to_train_on(self):
        df = _frame([5, 3, 1])
        assert apply_label_maturity(df).empty
