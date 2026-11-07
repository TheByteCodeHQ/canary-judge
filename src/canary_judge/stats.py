"""The statistics, in plain Python. A canary has tens of noisy samples, not millions, so say how sure you are instead of comparing two numbers."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

STATISTICS = ("median", "mean", "p90", "p95", "p99")


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def statistic(name: str) -> Callable[[Sequence[float]], float]:
    if name == "mean":
        return lambda v: sum(v) / len(v)
    if name == "median":
        return lambda v: percentile(v, 0.5)
    if name in STATISTICS:
        fraction = int(name[1:]) / 100
        return lambda v: percentile(v, fraction)
    raise ValueError(f"unknown statistic '{name}'. Use one of {', '.join(STATISTICS)}")


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test (normal approximation with tie and continuity correction). Returns (U for a, p-value).

    It asks whether one group tends to be larger than the other without assuming a bell curve, which latency never is.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        raise ValueError("both groups need values")
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(combined)
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = average
        t = j - i + 1
        tie_term += t**3 - t
        i = j + 1
    rank_sum_a = sum(r for r, (_, group) in zip(ranks, combined, strict=True) if group == 0)
    u1 = rank_sum_a - n1 * (n1 + 1) / 2
    n = n1 + n2
    mean = n1 * n2 / 2
    variance = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if variance <= 0:
        return u1, 1.0
    z = (abs(u1 - mean) - 0.5) / math.sqrt(variance)
    return u1, math.erfc(max(z, 0.0) / math.sqrt(2))


def lag1_autocorrelation(values: Sequence[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values)
    if var == 0:
        return 0.0
    return sum((values[i] - mean) * (values[i + 1] - mean) for i in range(n - 1)) / var


def effective_samples(values: Sequence[float]) -> float:
    """How many independent samples the series is worth. 60 points with strong memory (rho 0.8) carry about the information of 7."""
    rho = min(max(lag1_autocorrelation(values), 0.0), 0.95)
    return len(values) * (1 - rho) / (1 + rho)


def choose_block(values: Sequence[float]) -> int:
    """Samples from a running service are not independent (a slow minute is followed by another slow minute). Resampling in blocks keeps
    that, so the interval is not falsely tight. The block is as long as the series' own memory: about three times 1 / (1 - rho)."""
    n = len(values)
    rho = min(max(lag1_autocorrelation(values), 0.0), 0.95)
    return max(1, round(n ** (1 / 3)), min(math.ceil(3 / (1 - rho)), max(1, n // 4)))


def block_resample(values: Sequence[float], block: int, rng: random.Random) -> list[float]:
    n = len(values)
    block = max(1, min(block, n))
    out: list[float] = []
    while len(out) < n:
        start = rng.randrange(0, n - block + 1)
        out.extend(values[start:start + block])
    return out[:n]


def bootstrap_change(
    baseline: Sequence[float],
    canary: Sequence[float],
    stat: Callable[[Sequence[float]], float],
    *,
    relative: bool,
    sign: int,
    confidence: float = 0.95,
    rounds: int = 1000,
    seed: int = 7,
    block: int | None = None,
) -> tuple[float, float, float]:
    """(point, low, high) of how much WORSE the canary is than the baseline. Positive means worse; ``sign`` is +1 when higher is worse
    and -1 when lower is worse. Relative changes are divided by the baseline's statistic.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = random.Random(seed)
    block_b = block or choose_block(baseline)
    block_c = block or choose_block(canary)

    def change(b: Sequence[float], c: Sequence[float]) -> float:
        sb, sc = stat(b), stat(c)
        diff = sign * (sc - sb)
        if not relative:
            return diff
        if sb == 0:
            return math.inf if diff > 0 else (0.0 if diff == 0 else -math.inf)
        return diff / abs(sb)

    point = change(baseline, canary)
    draws = sorted(change(block_resample(baseline, block_b, rng), block_resample(canary, block_c, rng)) for _ in range(rounds))
    tail = (1 - confidence) / 2
    low = draws[min(rounds - 1, max(0, int(tail * rounds)))]
    high = draws[min(rounds - 1, max(0, int((1 - tail) * rounds) - 1))]
    return point, low, high


def bootstrap_value(values: Sequence[float], stat: Callable[[Sequence[float]], float], *, confidence: float = 0.95, rounds: int = 1000,
                    seed: int = 7, block: int | None = None) -> tuple[float, float, float]:
    """(point, low, high) of a single group's statistic, for checking a canary against an absolute SLO."""
    rng = random.Random(seed)
    size = block or choose_block(values)
    draws = sorted(stat(block_resample(values, size, rng)) for _ in range(rounds))
    tail = (1 - confidence) / 2
    return stat(values), draws[max(0, int(tail * rounds))], draws[min(rounds - 1, int((1 - tail) * rounds) - 1)]
