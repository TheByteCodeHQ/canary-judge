import json
import math
import random

import pytest

from canary_judge.config import ConfigError, config_from_dict
from canary_judge.loaders import LoadError, load_series
from canary_judge.stats import (
    block_resample,
    bootstrap_change,
    bootstrap_value,
    choose_block,
    effective_samples,
    lag1_autocorrelation,
    mann_whitney,
    percentile,
    statistic,
)

# -- statistics ----------------------------------------------------------------------------------------

def test_percentile_and_named_statistics():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert statistic("median")([1, 2, 3]) == 2
    assert statistic("mean")([1, 2, 6]) == 3
    assert statistic("p95")(list(range(101))) == 95
    with pytest.raises(ValueError):
        statistic("p50")
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_mann_whitney_known_values():
    u, p = mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert u == 0 and p == pytest.approx(0.0122, abs=0.002)          # exact test says 0.0079; the normal approximation is a little looser
    assert mann_whitney([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])[1] == 1.0
    assert mann_whitney([3, 3, 3], [3, 3, 3])[1] == 1.0              # all ties: no variance, no evidence
    assert mann_whitney([1, 2, 3], [1.5, 2.5, 3.5])[1] > 0.5
    with pytest.raises(ValueError):
        mann_whitney([], [1])


def test_mann_whitney_is_symmetric_in_p():
    a, b = [1.1, 2.3, 3.7, 4.1, 9.0], [2.2, 5.5, 6.1, 7.3, 8.8, 9.9]
    assert mann_whitney(a, b)[1] == pytest.approx(mann_whitney(b, a)[1])


def series(n, seed, scale=1.0, spread=0.3):
    rng = random.Random(seed)
    return [scale * math.exp(spread * rng.gauss(0, 1)) for _ in range(n)]


def test_bootstrap_interval_covers_zero_in_most_no_change_comparisons_and_excludes_it_for_a_real_change():
    stat = statistic("median")
    covered = 0
    for seed in range(40):
        _, low, high = bootstrap_change(series(100, 2 * seed), series(100, 2 * seed + 1), stat, relative=True, sign=1, rounds=400)
        covered += low < 0 < high
    assert covered >= 34                                        # nominally 95% of 40 = 38; allow for sampling noise
    point, low, high = bootstrap_change(series(100, 1), series(100, 2, scale=1.5), stat, relative=True, sign=1)
    assert low > 0.2 and point == pytest.approx(0.5, abs=0.15)


def test_sign_flips_the_meaning_of_worse():
    stat = statistic("mean")
    base, cand = series(80, 1), series(80, 2, scale=1.5)
    worse_if_higher = bootstrap_change(base, cand, stat, relative=True, sign=1)[0]
    worse_if_lower = bootstrap_change(base, cand, stat, relative=True, sign=-1)[0]
    assert worse_if_higher > 0 > worse_if_lower


def test_bootstrap_is_repeatable_and_seed_dependent():
    base, cand = series(50, 1), series(50, 2)
    stat = statistic("median")
    assert bootstrap_change(base, cand, stat, relative=True, sign=1, seed=3) == bootstrap_change(base, cand, stat, relative=True, sign=1, seed=3)
    assert bootstrap_change(base, cand, stat, relative=True, sign=1, seed=3) != bootstrap_change(base, cand, stat, relative=True, sign=1, seed=4)


def test_zero_baseline_relative_change_is_infinite_not_a_crash():
    point, _, _ = bootstrap_change([0.0] * 30, [1.0] * 30, statistic("mean"), relative=True, sign=1)
    assert point == math.inf
    assert bootstrap_change([0.0] * 30, [1.0] * 30, statistic("mean"), relative=False, sign=1)[0] == 1.0


def test_bootstrap_rejects_bad_confidence():
    with pytest.raises(ValueError):
        bootstrap_change([1, 2], [1, 2], statistic("mean"), relative=True, sign=1, confidence=1.0)


def test_bootstrap_value_brackets_the_statistic():
    point, low, high = bootstrap_value(series(100, 5), statistic("median"))
    assert low <= point <= high


def test_block_resample_keeps_runs_together():
    rng = random.Random(1)
    out = block_resample(list(range(100)), 10, rng)
    assert len(out) == 100
    runs = sum(1 for a, b in zip(out, out[1:], strict=False) if b == a + 1)
    assert runs >= 80                                                  # mostly consecutive values, unlike an ordinary resample


def test_autocorrelation_and_effective_samples():
    rng = random.Random(2)
    white = [rng.gauss(0, 1) for _ in range(400)]
    sticky, z = [], 0.0
    for _ in range(400):
        z = 0.9 * z + rng.gauss(0, 0.4)
        sticky.append(z)
    assert abs(lag1_autocorrelation(white)) < 0.15 and lag1_autocorrelation(sticky) > 0.7
    assert effective_samples(white) > 250 and effective_samples(sticky) < 60
    assert choose_block(sticky) > choose_block(white)
    assert lag1_autocorrelation([5.0] * 10) == 0.0 and lag1_autocorrelation([1, 2]) == 0.0


# -- loaders -------------------------------------------------------------------------------------------

def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_json_map_with_and_without_wrapper(tmp_path):
    plain = write(tmp_path, "a.json", json.dumps({"lat": [1, 2, 3]}))
    wrapped = write(tmp_path, "b.json", json.dumps({"metrics": {"lat": [1, 2, 3]}}))
    assert load_series(plain)[0] == load_series(wrapped)[0] == {"lat": [1.0, 2.0, 3.0]}


def test_csv_is_sorted_by_timestamp_when_present(tmp_path):
    path = write(tmp_path, "a.csv", "timestamp,metric,value\n3,lat,30\n1,lat,10\n2,lat,20\n1,err,0.1\n")
    assert load_series(path)[0] == {"lat": [10.0, 20.0, 30.0], "err": [0.1]}


def test_csv_without_the_right_columns_is_an_error(tmp_path):
    with pytest.raises(LoadError, match="'metric' and 'value'"):
        load_series(write(tmp_path, "a.csv", "name,val\nx,1\n"))
    with pytest.raises(LoadError, match="numeric"):
        load_series(write(tmp_path, "b.csv", "timestamp,metric,value\nsoon,lat,1\n"))


def prom(*series_list):
    return json.dumps({"status": "success", "data": {"resultType": "matrix", "result": list(series_list)}})


def test_prometheus_matrix_is_read_in_time_order_with_nan_gaps_dropped(tmp_path):
    body = prom({"metric": {"__name__": "lat"}, "values": [[3, "30"], [1, "10"], [2, "NaN"]]})
    series_map, skipped = load_series(write(tmp_path, "p.json", body))
    assert series_map == {"lat": [10.0, 30.0]} and skipped == {"lat": 1}


def test_prometheus_problems_are_explained(tmp_path):
    with pytest.raises(LoadError, match="no __name__"):
        load_series(write(tmp_path, "a.json", prom({"metric": {}, "values": []})))
    with pytest.raises(LoadError, match="both called 'lat'"):
        load_series(write(tmp_path, "b.json", prom({"metric": {"__name__": "lat"}, "values": []}, {"metric": {"__name__": "lat"}, "values": []})))
    with pytest.raises(LoadError, match="matrix"):
        load_series(write(tmp_path, "c.json", json.dumps({"status": "success", "data": {"resultType": "vector", "result": []}})))


def test_bad_numbers_are_errors_not_silent_drops(tmp_path):
    for bad in ([1, True], [1, "fast"], [1, None]):
        with pytest.raises(LoadError):
            load_series(write(tmp_path, "a.json", json.dumps({"lat": bad})))
    with pytest.raises(LoadError, match="infinite"):
        load_series(write(tmp_path, "b.json", '{"lat": [1, Infinity]}'))


def test_loader_file_errors(tmp_path):
    with pytest.raises(LoadError, match="not found"):
        load_series(tmp_path / "nope.json")
    with pytest.raises(LoadError, match="not valid JSON"):
        load_series(write(tmp_path, "a.json", "{oops"))
    with pytest.raises(LoadError):
        load_series(write(tmp_path, "b.json", "[1, 2]"))
    with pytest.raises(LoadError, match="list of numbers"):
        load_series(write(tmp_path, "c.json", json.dumps({"lat": 5})))


# -- config --------------------------------------------------------------------------------------------

def test_config_defaults_and_strictness():
    cfg = config_from_dict({"metrics": {"lat": {}}})
    assert cfg.rules[0].direction == "lower_is_better" and cfg.rules[0].sign == 1 and cfg.alpha == 0.05
    assert config_from_dict({"metrics": {"qps": {"direction": "higher_is_better"}}}).rules[0].sign == -1
    bad = [
        ({"metrics": {"a": {"tolerence": 1}}}, "tolerence"),
        ({"metrics": {"a": {}}, "alfa": 1}, "alfa"),
        ({"metrics": {}}, "at least one"),
        ({"metrics": {"a": {"gating": False}}}, "gating"),
        ({"metrics": {"a": {"direction": "up"}}}, "direction"),
        ({"metrics": {"a": {"statistic": "p50"}}}, "statistic"),
        ({"metrics": {"a": {"mode": "percent"}}}, "mode"),
        ({"metrics": {"a": {"max_regression": -1}}}, "max_regression"),
        ({"metrics": {"a": {"slo_min": 5, "slo_max": 1}}}, "slo_min"),
        ({"metrics": {"a": {"min_samples": 2}}}, "min_samples"),
        ({"metrics": {"a": {}}, "alpha": 0.9}, "alpha"),
        ({"metrics": {"a": {}}, "bootstrap_rounds": 10}, "bootstrap_rounds"),
        ({"metrics": {"a": {}}, "block": 0}, "block"),
        ("nope", "object"),
    ]
    for data, message in bad:
        with pytest.raises(ConfigError, match=message):
            config_from_dict(data)
