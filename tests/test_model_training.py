"""
Training on synthetic data through the real retraining pipeline (T-18).

One small generated dataset is trained on once per module, through
scripts/retrain.run - point-in-time features, chronological split,
champion/challenger - into a temporary registry, so the shipped model is never
touched.
"""
import json

import pytest

from app.features import BASELINE_FEATURES, FEATURE_ORDER
from app.ML.registry import load_model
from scripts.generate_data import GeneratorConfig, generate
from scripts.retrain import MODEL_FEATURES, recall_at_fpr, run


def _tree_depths(calibrated_model) -> list[int]:
    """Depth of every tree in the LightGBM booster under the calibration wrapper."""
    booster = calibrated_model.calibrated_classifiers_[0].estimator.estimator.booster_

    def depth(node) -> int:
        if "leaf_index" in node or "leaf_value" in node and "split_index" not in node:
            return 0
        return 1 + max(depth(node["left_child"]), depth(node["right_child"]))

    return [depth(tree["tree_structure"]) for tree in booster.dump_model()["tree_info"]]

# Small enough to build features in seconds, with enough fraud to measure.
CONFIG = GeneratorConfig(transactions=2500, users=150, fraud_rate=0.04, seed=5)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Generate, then train through retrain.run, once for the whole module."""
    from tests.conftest import TestingSessionLocal, engine
    from app.database import Base
    from app.repositories.transaction_repo import clear_population_snapshots

    clear_population_snapshots()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    root = tmp_path_factory.mktemp("registry")
    db = TestingSessionLocal()
    try:
        generate(db, CONFIG)
        result = run(db=db, root=root)
    finally:
        db.close()
    return result, root


@pytest.fixture(autouse=True)
def reset_db():
    """This module builds its own data once; the per-test reset would wipe it."""
    yield


class TestTheChallenger:
    def test_it_trains_on_every_monotone_safe_feature(self, trained):
        """All of FEATURE_ORDER except the two that cannot be monotone in amount (T-19)."""
        result, _ = trained
        assert list(result.challenger.feature_names_in_) == MODEL_FEATURES
        assert len(MODEL_FEATURES) == len(FEATURE_ORDER) - 2

    def test_trees_reach_a_meaningful_depth(self, trained):
        """A9: the old forest collapsed to depth-1 stumps on separable data."""
        result, _ = trained
        depths = _tree_depths(result.challenger)
        assert len(depths) > 1
        assert max(depths) >= 4
        assert sum(depths) / len(depths) >= 2

    def test_it_learns_real_signal(self, trained):
        result, _ = trained
        assert result.challenger_metrics["auc"] > 0.8
        assert 0.0 < result.challenger_metrics["average_precision"] <= 1.0

    def test_it_beats_the_baseline_trained_on_twenty_rows(self, trained):
        result, _ = trained
        # No champion in the temporary registry: the challenger wins by default.
        assert result.promoted is True


class TestTheManifest:
    def test_importance_is_recorded_for_every_model_feature(self, trained):
        _, root = trained
        _, manifest = load_model(root=root)
        assert set(manifest.feature_importance) == set(MODEL_FEATURES)

    def test_importance_is_informative(self, trained):
        """Some features matter; permutation importance is not uniformly zero."""
        _, root = trained
        _, manifest = load_model(root=root)
        assert max(manifest.feature_importance.values()) > 0.001

    def test_fraud_metrics_are_recorded(self, trained):
        _, root = trained
        _, manifest = load_model(root=root)
        assert {"auc", "average_precision", "recall_at_1pct_fpr", "test_rows", "brier",
                "expected_calibration_error", "inference_ms_p50", "inference_ms_p99",
                "scale_pos_weight"} <= set(manifest.metrics)
        assert manifest.training_rows > manifest.metrics["test_rows"]
        assert manifest.algorithm in ("LGBMClassifier+sigmoid", "LGBMClassifier+isotonic")

    def test_the_reliability_curve_is_stored(self, trained):
        _, root = trained
        _, manifest = load_model(root=root)
        assert manifest.reliability
        assert sum(row["count"] for row in manifest.reliability) == manifest.metrics["test_rows"]

    def test_the_new_model_is_active_in_its_registry(self, trained):
        result, root = trained
        _, manifest = load_model(root=root)
        assert manifest.version == result.manifest.version

    def test_older_manifests_without_importance_still_load(self, tmp_path):
        """Manifests written before T-18 have no feature_importance key."""
        from app.ML.registry import ModelManifest

        legacy = {"version": "v0", "trained_at": "2026-01-01T00:00:00+00:00", "feature_order": BASELINE_FEATURES,
                  "metrics": {}, "training_rows": 20, "algorithm": "RandomForestClassifier"}
        assert ModelManifest.from_json(json.dumps(legacy)).feature_importance == {}


class TestRecallAtFixedFpr:
    def test_perfect_separation(self):
        assert recall_at_fpr([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9]) == 1.0

    def test_no_recall_without_false_positives(self):
        # Fraud scored below every legit transaction: catching any means blocking all.
        assert recall_at_fpr([1, 0, 0, 0], [0.1, 0.5, 0.6, 0.7], max_fpr=0.01) == 0.0

    def test_partial(self):
        y = [0] * 100 + [1, 1]
        scores = [i / 100 for i in range(100)] + [0.995, 0.5]
        assert recall_at_fpr(y, scores, max_fpr=0.01) == 0.5
