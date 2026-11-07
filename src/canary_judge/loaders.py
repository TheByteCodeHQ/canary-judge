"""Read metric samples from the shapes people actually have: a JSON map, a CSV, or a raw Prometheus ``query_range`` response."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

MAX_POINTS = 1_000_000


class LoadError(ValueError):
    pass


def _number(value: Any, where: str) -> float | None:
    """A finite float, or None for NaN (Prometheus uses it for gaps). Infinity and non-numbers are errors, never silently dropped."""
    if isinstance(value, bool):
        raise LoadError(f"{where}: a boolean is not a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LoadError(f"{where}: {str(value)[:30]!r} is not a number") from None
    if math.isnan(number):
        return None
    if math.isinf(number):
        raise LoadError(f"{where}: infinite values cannot be compared; fix the query")
    return number


def _clean(name: str, raw: list[Any], skipped: dict[str, int]) -> list[float]:
    if len(raw) > MAX_POINTS:
        raise LoadError(f"{name}: more than {MAX_POINTS} points")
    values = []
    for index, item in enumerate(raw):
        number = _number(item, f"{name}[{index}]")
        if number is None:
            skipped[name] = skipped.get(name, 0) + 1
        else:
            values.append(number)
    return values


def _prometheus(data: dict[str, Any], skipped: dict[str, int]) -> dict[str, list[float]]:
    if data.get("status") != "success" or data.get("data", {}).get("resultType") != "matrix":
        raise LoadError("a Prometheus response must be a successful 'matrix' result (use the query_range API)")
    out: dict[str, list[float]] = {}
    for series in data["data"]["result"]:
        labels = series.get("metric", {})
        name = labels.get("__name__") or labels.get("name")
        if not name:
            raise LoadError("a Prometheus series has no __name__ or name label. Name it in the query, e.g. label_replace(..., 'name', 'latency', '', '')")
        if name in out:
            raise LoadError(f"two Prometheus series are both called '{name}'. Aggregate them in the query (sum, avg, quantile) so each metric is one series")
        points = sorted(series.get("values", []), key=lambda p: float(p[0]))
        out[name] = _clean(name, [p[1] for p in points], skipped)
    return out


def load_series(path: Path) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Returns (metric name -> samples in time order, count of NaN gaps dropped per metric)."""
    if not path.is_file():
        raise LoadError(f"File not found: {path}")
    skipped: dict[str, int] = {}
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames or not {"metric", "value"} <= set(reader.fieldnames):
            raise LoadError(f"{path.name}: a CSV needs 'metric' and 'value' columns (and optionally 'timestamp')")
        rows = list(reader)
        if "timestamp" in (reader.fieldnames or []):
            try:
                rows.sort(key=lambda r: float(r["timestamp"]))
            except ValueError:
                raise LoadError(f"{path.name}: 'timestamp' must be numeric (epoch seconds)") from None
        grouped: dict[str, list[Any]] = {}
        for row in rows:
            grouped.setdefault(row["metric"], []).append(row["value"])
        return {name: _clean(name, raw, skipped) for name, raw in grouped.items()}, skipped
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise LoadError(f"{path.name} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise LoadError(f"{path.name}: expected a JSON object")
    if "status" in data and "data" in data:
        return _prometheus(data, skipped), skipped
    metrics = data.get("metrics", data)
    if not isinstance(metrics, dict) or not metrics:
        raise LoadError(f"{path.name}: expected {{\"metrics\": {{\"name\": [numbers]}}}} or a Prometheus response")
    out = {}
    for name, raw in metrics.items():
        if not isinstance(raw, list):
            raise LoadError(f"{path.name}: '{name}' must be a list of numbers")
        out[name] = _clean(name, raw, skipped)
    return out, skipped
