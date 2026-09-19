"""Rule weights and policy thresholds, loaded from config/rules.yaml.

The file is validated in full before it is used. The API loads it at startup
and stops if it is malformed; after that, an edited file is picked up on the
next request, and an edit that fails validation is rejected while the last
good configuration stays in force. Retuning never needs a code change.
"""
import logging
import os
import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.yaml"

# Rules whose predicate reads a threshold from config. The rule logic itself
# stays in app/scoring.py.
RULES_WITH_THRESHOLDS = {"R1_HIGH_AMOUNT", "R2_AMOUNT_DEVIATION", "R5_VELOCITY"}
KNOWN_RULE_IDS = {
    "R1_HIGH_AMOUNT",
    "R2_AMOUNT_DEVIATION",
    "R3_NEW_LOCATION",
    "R4_FLAGGED_DEVICE",
    "R5_VELOCITY",
}

_WEIGHT_TOLERANCE = 1e-9


class _Strict(BaseModel):
    # A misspelled key is a silent misconfiguration; refuse it.
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuleSpec(_Strict):
    weight: float = Field(gt=0)
    threshold: float | None = None


class ScoringSection(_Strict):
    w_rules: float = Field(ge=0, le=1)
    w_model: float = Field(ge=0, le=1)
    rules: dict[str, RuleSpec]

    @model_validator(mode="after")
    def _check(self) -> "ScoringSection":
        if abs(self.w_rules + self.w_model - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(
                f"w_rules + w_model must equal 1.0 (got {self.w_rules + self.w_model}); "
                "that is what keeps the score in [0, 1] without clamping"
            )
        missing = KNOWN_RULE_IDS - set(self.rules)
        unknown = set(self.rules) - KNOWN_RULE_IDS
        if missing:
            raise ValueError(f"rules missing from config: {sorted(missing)}")
        if unknown:
            raise ValueError(f"unknown rule ids (no logic in app/scoring.py): {sorted(unknown)}")
        for rule_id in RULES_WITH_THRESHOLDS:
            if self.rules[rule_id].threshold is None:
                raise ValueError(f"{rule_id} needs a threshold")
        return self


class PolicySection(_Strict):
    version: str = Field(min_length=1)
    allow_below: float = Field(ge=0)
    step_up_below: float = Field(ge=0)
    review_below: float = Field(ge=0)
    high_value_amount: float = Field(gt=0)
    high_value_tightening: float = Field(gt=0, le=1)
    trusted_tier_loosening: float = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> "PolicySection":
        if not self.allow_below <= self.step_up_below <= self.review_below:
            raise ValueError(
                "policy bands must be ordered: allow_below <= step_up_below <= review_below"
            )
        return self


class RulesConfig(_Strict):
    version: str = Field(min_length=1)
    scoring: ScoringSection
    policy: PolicySection


class ConfigError(RuntimeError):
    """The rules file could not be read or failed validation."""


def config_path() -> Path:
    return Path(os.getenv("RULES_CONFIG_PATH", str(DEFAULT_PATH)))


def load_rules_config(path: Path | None = None) -> RulesConfig:
    """Read and validate a rules file. Raises ConfigError with the reason."""
    path = path or config_path()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"rules config not found: {path}")
    except yaml.YAMLError as exc:
        raise ConfigError(f"rules config is not valid YAML ({path}): {exc}")
    if not isinstance(raw, dict):
        raise ConfigError(f"rules config must be a mapping at the top level: {path}")
    try:
        return RulesConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"rules config failed validation ({path}):\n{exc}")


_lock = threading.Lock()
_cache: dict[str, object] = {"key": None, "config": None}


def get_rules_config() -> RulesConfig:
    """The current configuration, reloaded when the file changes.

    A file that fails validation on reload is logged and ignored, keeping the
    last good configuration: a typo during a retune must not take payments down.
    With no good configuration loaded yet, the error propagates.
    """
    path = config_path()
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        key = (str(path), None)

    with _lock:
        if key == _cache["key"] and _cache["config"] is not None:
            return _cache["config"]
        try:
            config = load_rules_config(path)
        except ConfigError:
            last_good = _cache["config"]
            if last_good is None or _cache["key"] is None or _cache["key"][0] != str(path):
                raise
            logger.error("Rejected edited rules config; keeping %s", last_good.version, exc_info=True)
            _cache["key"] = key            # do not re-parse the same bad file every request
            return last_good
        _cache["key"] = key
        _cache["config"] = config
        return config
