"""What 'safe to promote' means, written down: per-metric tolerances, SLOs and how much evidence to demand."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .stats import STATISTICS

TOP_KEYS = {"alpha", "warmup_points", "min_samples", "bootstrap_rounds", "seed", "block", "metrics"}
RULE_KEYS = {"direction", "statistic", "max_regression", "mode", "gating", "slo_max", "slo_min", "min_samples"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    name: str
    direction: str = "lower_is_better"          # lower_is_better: latency, errors.  higher_is_better: throughput, success rate
    statistic: str = "median"
    max_regression: float = 0.10                # how much worse than the baseline is still acceptable
    mode: str = "relative"                      # relative: a fraction of the baseline (0.10 = 10%). absolute: in the metric's own unit
    gating: bool = True                         # a failing non-gating metric is reported but cannot roll back a release
    slo_max: float | None = None                # the canary itself must stay at or below this
    slo_min: float | None = None                # ... or at or above this
    min_samples: int | None = None

    @property
    def sign(self) -> int:
        return 1 if self.direction == "lower_is_better" else -1      # +1: a higher canary value is worse


@dataclass(frozen=True)
class Config:
    rules: tuple[Rule, ...]
    alpha: float = 0.05                         # overall false-alarm budget, shared across the gating metrics (Bonferroni)
    warmup_points: int = 0                      # leading samples to ignore (cold caches, JIT, connection pools)
    min_samples: int = 30
    bootstrap_rounds: int = 1000
    seed: int = 7
    block: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _rule(name: str, raw: Any) -> Rule:
    if not isinstance(raw, dict):
        raise ConfigError(f"metric '{name}': settings must be an object")
    unknown = set(raw) - RULE_KEYS
    if unknown:
        raise ConfigError(f"metric '{name}': unknown setting(s) {sorted(unknown)}. Allowed: {sorted(RULE_KEYS)}")
    rule = Rule(name, **raw)
    if rule.direction not in ("lower_is_better", "higher_is_better"):
        raise ConfigError(f"metric '{name}': direction must be lower_is_better or higher_is_better")
    if rule.statistic not in STATISTICS:
        raise ConfigError(f"metric '{name}': statistic must be one of {', '.join(STATISTICS)}")
    if rule.mode not in ("relative", "absolute"):
        raise ConfigError(f"metric '{name}': mode must be relative or absolute")
    if isinstance(rule.max_regression, bool) or not isinstance(rule.max_regression, (int, float)) or rule.max_regression < 0:
        raise ConfigError(f"metric '{name}': max_regression must be a number of at least 0")
    if rule.slo_max is not None and rule.slo_min is not None and rule.slo_min > rule.slo_max:
        raise ConfigError(f"metric '{name}': slo_min is above slo_max")
    if rule.min_samples is not None and (not isinstance(rule.min_samples, int) or rule.min_samples < 5):
        raise ConfigError(f"metric '{name}': min_samples must be a whole number of at least 5")
    return rule


def config_from_dict(data: dict[str, Any]) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("the config must be a JSON object")
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise ConfigError(f"unknown config key(s) {sorted(unknown)}. Allowed: {sorted(TOP_KEYS)}")
    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ConfigError("'metrics' must list at least one metric")
    rules = tuple(_rule(name, raw) for name, raw in metrics.items())
    if not any(r.gating for r in rules):
        raise ConfigError("at least one metric must be gating, or nothing can ever block a release")
    kwargs = {k: data[k] for k in TOP_KEYS - {"metrics"} if k in data}
    config = Config(rules, **kwargs)
    if not 0 < config.alpha < 0.5:
        raise ConfigError("alpha must be between 0 and 0.5 (0.05 means a 5% overall chance of a false alarm)")
    if config.warmup_points < 0 or config.min_samples < 5:
        raise ConfigError("warmup_points must be >= 0 and min_samples >= 5")
    if config.bootstrap_rounds < 200:
        raise ConfigError("bootstrap_rounds must be at least 200")
    if config.block is not None and config.block < 1:
        raise ConfigError("block must be at least 1")
    return config


def load_config(path: Path) -> Config:
    try:
        return config_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from None
    except ValueError as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{path.name} is not valid JSON: {exc}") from None
