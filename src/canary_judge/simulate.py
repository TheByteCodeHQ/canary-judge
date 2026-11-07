"""Test the judge itself on data where the truth is known: how often does it roll back a good canary, and how often does it catch a bad one?

Latency is simulated as a log-normal series with autocorrelation (a slow moment tends to be followed by another), because comparing two
calm bell curves would flatter the judge. A regression of r multiplies every canary value by (1 + r).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import Config, Rule
from .judge import judge


@dataclass
class Outcome:
    regression: float
    trials: int
    promote: int
    extend: int
    rollback: int

    @property
    def rollback_rate(self) -> float:
        return self.rollback / self.trials


def latency_series(n: int, rng: random.Random, *, median: float = 100.0, spread: float = 0.35, phi: float = 0.5) -> list[float]:
    z = rng.gauss(0, 1)
    out = []
    for _ in range(n):
        z = phi * z + math.sqrt(1 - phi * phi) * rng.gauss(0, 1)
        out.append(median * math.exp(spread * z))
    return out


def simulate(*, samples: int, regressions: list[float], trials: int, tolerance: float = 0.10, phi: float = 0.5, seed: int = 1,
             rounds: int = 400, statistic: str = "median") -> list[Outcome]:
    config = Config((Rule("latency", statistic=statistic, max_regression=tolerance),), min_samples=min(30, samples), bootstrap_rounds=rounds, seed=seed)
    rng = random.Random(seed)
    results = []
    for regression in regressions:
        counts = {"promote": 0, "extend": 0, "rollback": 0}
        for _ in range(trials):
            baseline = latency_series(samples, rng, phi=phi)
            canary = [v * (1 + regression) for v in latency_series(samples, rng, phi=phi)]
            counts[judge({"latency": baseline}, {"latency": canary}, config).decision] += 1
        results.append(Outcome(regression, trials, **counts))
    return results
