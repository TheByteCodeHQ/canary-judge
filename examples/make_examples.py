"""Regenerates the example data in this folder:  python examples/make_examples.py

Three scenarios (good, bad, thin) for the same config, plus the same bad canary in Prometheus query_range format. Everything is seeded,
so the files and the verdicts they produce are reproducible.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).parent


def latency(n: int, rng: random.Random, scale: float = 1.0) -> list[float]:
    z, out = rng.gauss(0, 1), []
    for _ in range(n):
        z = 0.5 * z + math.sqrt(0.75) * rng.gauss(0, 1)
        out.append(round(180 * scale * math.exp(0.12 * z), 2))
    return out


def errors(n: int, rng: random.Random, level: float = 0.002) -> list[float]:
    return [round(max(0.0, rng.gauss(level, 0.0012)), 5) for _ in range(n)]


def cpu(n: int, rng: random.Random, scale: float = 1.0) -> list[float]:
    return [round(min(100.0, max(0.0, rng.gauss(55 * scale, 6))), 2) for _ in range(n)]


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def scenario(name: str, n: int, canary_latency: float, canary_errors: float, canary_cpu: float, seed: int) -> dict:
    rng = random.Random(seed)
    base = {"latency_p95_ms": latency(n, rng), "error_rate": errors(n, rng), "cpu_percent": cpu(n, rng)}
    cand = {"latency_p95_ms": latency(n, rng, canary_latency), "error_rate": errors(n, rng, canary_errors), "cpu_percent": cpu(n, rng, canary_cpu)}
    write(HERE / name / "baseline.json", {"metrics": base})
    write(HERE / name / "canary.json", {"metrics": cand})
    return {"baseline": base, "canary": cand}


def prometheus(path: Path, metrics: dict[str, list[float]]) -> None:
    start = 1_700_000_000
    result = [{"metric": {"__name__": name, "job": "checkout"}, "values": [[start + 60 * i, str(v)] for i, v in enumerate(values)]} for name, values in metrics.items()]
    write(path, {"status": "success", "data": {"resultType": "matrix", "result": result}})


if __name__ == "__main__":
    write(HERE / "config.json", {
        "alpha": 0.05,
        "warmup_points": 10,
        "min_samples": 30,
        "metrics": {
            "latency_p95_ms": {"statistic": "median", "max_regression": 0.10, "slo_max": 300},
            "error_rate": {"statistic": "mean", "mode": "absolute", "max_regression": 0.002},
            "cpu_percent": {"statistic": "mean", "max_regression": 0.25, "gating": False},
        },
    })
    scenario("good", 240, 1.0, 0.002, 1.0, seed=1)
    bad = scenario("bad", 240, 1.35, 0.006, 1.4, seed=2)
    scenario("thin", 140, 1.08, 0.003, 1.0, seed=3)
    prometheus(HERE / "prometheus" / "baseline.json", bad["baseline"])
    prometheus(HERE / "prometheus" / "canary.json", bad["canary"])
    write(HERE / "prometheus" / "config.json", json.loads((HERE / "config.json").read_text(encoding="utf-8")))
    print("examples written")
