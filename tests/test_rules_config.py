"""
config/rules.yaml — validation, hot reload, and failing fast (T-13).
"""
import os
import pathlib
import subprocess
import sys

import pytest
import yaml

from app.config import DEFAULT_PATH, ConfigError, get_rules_config, load_rules_config

REPO = pathlib.Path(__file__).resolve().parent.parent
SHIPPED = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
BASE_TX = {"location": "Lahore", "amount": 100.0, "device_id": "device-001"}


class TestShippedConfig:
    def test_shipped_file_is_valid(self):
        config = load_rules_config(DEFAULT_PATH)
        assert config.scoring.w_rules + config.scoring.w_model == pytest.approx(1.0)

    def test_shipped_values_preserve_the_pre_t13_behaviour(self):
        """Moving the numbers into a file must not change a single score."""
        config = load_rules_config(DEFAULT_PATH)
        weights = {rule_id: spec.weight for rule_id, spec in config.scoring.rules.items()}
        assert weights == {"R1_HIGH_AMOUNT": 0.25, "R2_AMOUNT_DEVIATION": 0.25,
                           "R3_NEW_LOCATION": 0.2, "R4_FLAGGED_DEVICE": 0.3, "R5_VELOCITY": 0.5}
        assert (config.scoring.w_rules, config.scoring.w_model) == (0.7, 0.3)
        policy = config.policy
        assert (policy.allow_below, policy.step_up_below, policy.review_below) == (0.3, 0.5, 0.7)
        assert (policy.high_value_tightening, policy.trusted_tier_loosening) == (0.8, 1.2)


# Each case: (description, override applied to the shipped config, expected message fragment)
MALFORMED = [
    ("weights do not sum to 1", {"scoring": {"w_rules": 0.7, "w_model": 0.4}}, "must equal 1.0"),
    ("negative rule weight", {"scoring": {"rules": {"R3_NEW_LOCATION": {"weight": -0.1}}}}, "greater than 0"),
    ("unknown rule id", {"scoring": {"rules": {"R9_MADE_UP": {"weight": 0.1}}}}, "unknown rule ids"),
    ("threshold missing", {"scoring": {"rules": {"R1_HIGH_AMOUNT": {"weight": 0.25, "threshold": None}}}}, "needs a threshold"),
    ("bands out of order", {"policy": {"allow_below": 0.6, "step_up_below": 0.5}}, "must be ordered"),
    ("tightening above 1", {"policy": {"high_value_tightening": 1.5}}, "less than or equal to 1"),
    ("misspelled key", {"policy": {"alow_below": 0.3}}, "Extra inputs are not permitted"),
    ("empty version", {"version": ""}, "at least 1 character"),
]


def _merged(overrides):
    import copy

    data = copy.deepcopy(SHIPPED)
    for section, values in overrides.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if key == "rules":
                    for rule_id, spec in value.items():
                        data["scoring"]["rules"][rule_id] = spec
                else:
                    data[section][key] = value
        else:
            data[section] = values
    return data


class TestValidation:
    @pytest.mark.parametrize("name, overrides, message", MALFORMED, ids=[m[0] for m in MALFORMED])
    def test_malformed_config_is_rejected_with_a_reason(self, tmp_path, name, overrides, message):
        path = tmp_path / "rules.yaml"
        path.write_text(yaml.safe_dump(_merged(overrides)), encoding="utf-8")
        with pytest.raises(ConfigError, match=message):
            load_rules_config(path)

    def test_missing_rule_is_rejected(self, tmp_path):
        data = _merged({})
        del data["scoring"]["rules"]["R4_FLAGGED_DEVICE"]
        path = tmp_path / "rules.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="rules missing from config"):
            load_rules_config(path)

    def test_invalid_yaml(self, tmp_path):
        path = tmp_path / "rules.yaml"
        path.write_text("scoring: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_rules_config(path)

    def test_not_a_mapping(self, tmp_path):
        path = tmp_path / "rules.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_rules_config(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_rules_config(tmp_path / "absent.yaml")


class TestNoCodeChangeRetuning:
    def test_editing_a_threshold_changes_the_next_decision(self, client, auth_headers, rules_config):
        """The acceptance criterion, end to end: edit the file, the API follows."""
        assert client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()["decision"] == "ALLOW"
        rules_config(policy={"version": "policy-strict", "allow_below": 0, "step_up_below": 0, "review_below": 0})
        txn = client.post("/transactions/", json=BASE_TX, headers=auth_headers).json()
        assert txn["decision"] == "REJECT"
        assert txn["policy_version"] == "policy-strict"

    def test_same_file_edited_in_place_is_reloaded(self, tmp_path, monkeypatch):
        path = tmp_path / "rules.yaml"
        path.write_text(yaml.safe_dump(SHIPPED), encoding="utf-8")
        monkeypatch.setenv("RULES_CONFIG_PATH", str(path))
        assert get_rules_config().policy.allow_below == 0.3

        edited = _merged({"policy": {"allow_below": 0.25}})
        path.write_text(yaml.safe_dump(edited), encoding="utf-8")
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert get_rules_config().policy.allow_below == 0.25

    def test_a_bad_edit_keeps_the_last_good_config(self, tmp_path, monkeypatch):
        """A typo during a retune must not take payments down."""
        path = tmp_path / "rules.yaml"
        path.write_text(yaml.safe_dump(SHIPPED), encoding="utf-8")
        monkeypatch.setenv("RULES_CONFIG_PATH", str(path))
        good = get_rules_config()

        path.write_text("scoring: [broken\n", encoding="utf-8")
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert get_rules_config() is good

    def test_a_missing_file_at_startup_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RULES_CONFIG_PATH", str(tmp_path / "absent.yaml"))
        with pytest.raises(ConfigError, match="not found"):
            get_rules_config()

    def test_with_no_good_config_the_error_propagates(self, tmp_path, monkeypatch):
        path = tmp_path / "never-valid.yaml"
        path.write_text("scoring: [broken\n", encoding="utf-8")
        monkeypatch.setenv("RULES_CONFIG_PATH", str(path))
        with pytest.raises(ConfigError):
            get_rules_config()


def _start_api(extra_env):
    env = {**os.environ, "SECRET_KEY": "x", "DATABASE_URL": "sqlite:///:memory:", **extra_env}
    return subprocess.run([sys.executable, "-c", "import app.main"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


class TestFailFast:
    def test_api_will_not_start_on_a_malformed_config(self, tmp_path):
        """Malformed config fails at startup, not at the first request."""
        path = tmp_path / "rules.yaml"
        path.write_text(yaml.safe_dump(_merged({"scoring": {"w_rules": 0.9, "w_model": 0.9}})), encoding="utf-8")
        result = _start_api({"RULES_CONFIG_PATH": str(path)})
        assert result.returncode != 0
        assert "ConfigError" in result.stderr
        assert "must equal 1.0" in result.stderr

    def test_api_starts_on_the_shipped_config(self):
        result = _start_api({"RULES_CONFIG_PATH": str(DEFAULT_PATH)})
        assert result.returncode == 0, result.stderr
