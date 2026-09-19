"""
LightGBM with calibration, monotone in amount, scored fast (T-19).

Most tests train on a small synthetic matrix built here, so they run in
seconds; tests/test_model_training.py covers the full pipeline end to end.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

from app.features import FEATURE_ORDER, FeatureVector
from app.ML.scorer import build_scorer
from scripts.retrain import (
    CANDIDATES,
    ISOTONIC_MIN_POSITIVES,
    MAX_TREES,
    MAX_AUC_REGRESSION,
    MODEL_FEATURES,
    MONOTONE_INCREASING,
    NOT_MONOTONE_IN_AMOUNT,
    challenger_wins,
    expected_calibration_error,
    make_booster,
    measure_latency,
    reliability,
    train_challenger,
)


def _matrix(rows: int, fraud_rate: float, seed: int = 0):
    """Features with real signal: fraud tends to larger amounts, new devices, online."""
    rng = np.random.default_rng(seed)
    y = (rng.random(rows) < fraud_rate).astype(int)
    X = pd.DataFrame(rng.random((rows, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    X["amount"] = rng.lognormal(4, 1, rows) * np.where(y, rng.uniform(1, 6, rows), 1)
    X["amount_deviation"] = X["amount"] / 60
    X["is_card_not_present"] = np.where(y, rng.random(rows) < 0.8, rng.random(rows) < 0.3).astype(int)
    X["device_age_days"] = np.where(y, rng.uniform(0, 2, rows), rng.uniform(0, 400, rows))
    return X, y


@pytest.fixture(scope="module")
def trained():
    X, y = _matrix(6000, 0.03, seed=1)
    model, info = train_challenger(X, y)
    return model, info, X, y


class TestFeatureSet:
    def test_model_features_drop_only_what_cannot_be_monotone(self):
        assert set(FEATURE_ORDER) - set(MODEL_FEATURES) == set(NOT_MONOTONE_IN_AMOUNT)
        assert [f for f in FEATURE_ORDER if f in MODEL_FEATURES] == MODEL_FEATURES

    def test_every_constrained_feature_is_in_the_model(self):
        assert set(MONOTONE_INCREASING) <= set(MODEL_FEATURES)

    def test_constraints_are_set_per_feature(self):
        booster = make_booster(10.0)
        constraints = dict(zip(MODEL_FEATURES, booster.get_params()["monotone_constraints"]))
        assert {f for f, c in constraints.items() if c == 1} == set(MONOTONE_INCREASING)
        assert all(c in (0, 1) for c in constraints.values())

    def test_scale_pos_weight_comes_from_the_class_ratio(self, trained):
        _, info, X, y = trained
        cut = int(len(y) * 0.8)
        expected = (cut - y[:cut].sum()) / y[:cut].sum()
        assert info["scale_pos_weight"] == pytest.approx(expected)


class TestModelSelection:
    def test_the_choice_is_recorded(self, trained):
        _, info, _, _ = trained
        chosen = info["selection"]
        capacities = [{k: v for k, v in c.items()} for c in CANDIDATES]
        assert {k: chosen[k] for k in capacities[0] if k in chosen}   # has capacity settings
        assert any(all(chosen.get(k) == v for k, v in c.items()) for c in capacities)
        assert 0.0 <= chosen["validation_average_precision"] <= 1.0

    def test_early_stopping_stops_early(self, trained):
        _, info, _, _ = trained
        assert 1 <= info["selection"]["trees"] < MAX_TREES

    def test_the_spec_contract_is_a_candidate(self):
        """The spec's num_leaves=31 configuration is always in the running."""
        assert CANDIDATES[0]["num_leaves"] == 31

    def test_selection_never_sees_the_test_window(self):
        """train_challenger takes only the training window; the test split happens
        outside it, so the selection cannot depend on test rows."""
        import inspect

        from scripts import retrain

        assert list(inspect.signature(retrain.train_challenger).parameters) == ["X_train", "y_train"]


class TestCalibration:
    def test_few_fraud_cases_get_platt_scaling(self, trained):
        _, info, _, _ = trained
        assert info["calibration_fraud"] < ISOTONIC_MIN_POSITIVES
        assert info["calibration_method"] == "sigmoid"

    def test_plenty_of_fraud_cases_get_isotonic(self):
        X, y = _matrix(20_000, 0.35, seed=2)       # ~1,400 fraud in the calibration slice
        _, info = train_challenger(X, y)
        assert info["calibration_fraud"] >= ISOTONIC_MIN_POSITIVES
        assert info["calibration_method"] == "isotonic"

    def test_calibration_beats_the_raw_booster_on_unseen_data(self, trained):
        model, _, _, _ = trained
        X_new, y_new = _matrix(6000, 0.03, seed=9)
        raw = model.calibrated_classifiers_[0].estimator.estimator.predict_proba(X_new)[:, 1]
        calibrated = model.predict_proba(X_new)[:, 1]
        # scale_pos_weight inflates raw probabilities by design; calibration undoes it.
        assert brier_score_loss(y_new, calibrated) < brier_score_loss(y_new, raw)
        assert calibrated.mean() == pytest.approx(y_new.mean(), abs=0.02)

    def test_reliability_curve(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        p = np.array([0.05, 0.1, 0.72, 0.68, 0.71, 0.95])
        curve = reliability(y, p)
        by_bin = {(round(r["low"], 1)): r for r in curve}
        assert by_bin[0.0]["count"] == 1 and by_bin[0.1]["count"] == 1
        assert by_bin[0.6]["count"] == 1 and by_bin[0.7]["count"] == 2
        assert by_bin[0.7]["observed"] == 0.5
        assert sum(r["count"] for r in curve) == len(y)

    def test_expected_calibration_error(self):
        perfect = [{"count": 10, "predicted": 0.2, "observed": 0.2}]
        off = [{"count": 10, "predicted": 0.2, "observed": 0.5}, {"count": 30, "predicted": 0.9, "observed": 0.9}]
        assert expected_calibration_error(perfect) == 0.0
        assert expected_calibration_error(off) == pytest.approx(0.075)


class TestMonotonicity:
    @pytest.mark.parametrize("feature", MONOTONE_INCREASING)
    def test_raising_a_constrained_feature_never_lowers_the_probability(self, trained, feature):
        model, _, X, _ = trained
        base = X.sample(200, random_state=3).reset_index(drop=True)
        low, high = X[feature].quantile(0.01), X[feature].quantile(0.99) * 3
        previous = None
        for value in np.linspace(low, high, 25):
            probe = base.copy()
            probe[feature] = value
            current = model.predict_proba(probe)[:, 1]
            if previous is not None:
                assert (current >= previous - 1e-12).all(), feature
            previous = current


class TestFastScorer:
    @staticmethod
    def _vectors(X):
        defaults = {name: 0.0 for name in FEATURE_ORDER}
        return [FeatureVector(**{**defaults, **row}) for row in X.to_dict("records")]

    def test_matches_predict_proba_exactly_with_platt(self, trained):
        model, _, X, _ = trained
        sample = X.sample(2000, random_state=4)
        score = build_scorer(model, MODEL_FEATURES)
        fast = np.array([score(fv) for fv in self._vectors(sample)])
        assert np.abs(fast - model.predict_proba(sample)[:, 1]).max() < 1e-12

    def test_matches_predict_proba_exactly_with_isotonic(self):
        X, y = _matrix(20_000, 0.35, seed=2)
        model, info = train_challenger(X, y)
        assert info["calibration_method"] == "isotonic"
        sample = X.sample(1000, random_state=5)
        score = build_scorer(model, MODEL_FEATURES)
        fast = np.array([score(fv) for fv in self._vectors(sample)])
        assert np.abs(fast - model.predict_proba(sample)[:, 1]).max() < 1e-12

    def test_columns_are_taken_by_name_in_manifest_order(self, trained):
        """The fast path builds its row from the manifest, not from FeatureVector order."""
        model, _, X, _ = trained
        row = X.iloc[[0]]
        fv = self._vectors(row)[0]
        assert build_scorer(model, MODEL_FEATURES)(fv) == pytest.approx(model.predict_proba(row)[0, 1], abs=1e-12)

    def test_other_models_fall_back_to_predict_proba(self):
        X, y = _matrix(500, 0.2, seed=6)
        forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        score = build_scorer(forest, MODEL_FEATURES)
        fv = self._vectors(X.iloc[[0]])[0]
        assert score(fv) == pytest.approx(forest.predict_proba(X.iloc[[0]])[0, 1])

    def test_inference_is_under_ten_milliseconds(self, trained):
        model, _, X, _ = trained
        timing = measure_latency(model, X, samples=300)
        assert timing["inference_ms_p50"] < 10.0
        assert timing["inference_ms_p99"] < 10.0


class TestPromotionGate:
    CHAMPION = {"auc": 0.985, "average_precision": 0.87}

    def test_better_pr_auc_wins(self):
        wins, reason = challenger_wins({"auc": 0.986, "average_precision": 0.90}, self.CHAMPION)
        assert wins and "PR-AUC" in reason

    def test_better_roc_auc_alone_does_not_win(self):
        """The T-19 trial: +0.0016 ROC AUC, -0.03 PR-AUC. An AUC-only gate promoted it."""
        wins, _ = challenger_wins({"auc": 0.987, "average_precision": 0.84}, self.CHAMPION)
        assert not wins

    def test_pr_auc_gain_cannot_cost_too_much_roc_auc(self):
        wins, reason = challenger_wins({"auc": 0.985 - MAX_AUC_REGRESSION - 0.001, "average_precision": 0.95},
                                       self.CHAMPION)
        assert not wins and "fell" in reason

    def test_a_small_roc_auc_dip_is_tolerated(self):
        wins, _ = challenger_wins({"auc": 0.980, "average_precision": 0.90}, self.CHAMPION)
        assert wins

    def test_no_champion_means_the_challenger_wins(self):
        wins, _ = challenger_wins({"auc": 0.9, "average_precision": 0.5}, {"auc": 0.0})
        assert wins
