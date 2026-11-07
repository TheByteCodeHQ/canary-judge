import json
import math
import random
from pathlib import Path

import pytest

from canary_judge.cli import main
from canary_judge.config import config_from_dict
from canary_judge.judge import judge
from canary_judge.report import render_json, render_markdown, render_text
from canary_judge.simulate import latency_series, simulate


def lat(n, seed, scale=1.0, spread=0.3, phi=0.0):
    rng = random.Random(seed)
    z, out = rng.gauss(0, 1), []
    for _ in range(n):
        z = phi * z + math.sqrt(1 - phi * phi) * rng.gauss(0, 1)
        out.append(100 * scale * math.exp(spread * z))
    return out


def wave(n, factor=1.0, phase=0):
    """Deterministic, bounded, noise-like data (90..110 before scaling), so a test does not depend on a lucky random seed."""
    return [factor * (100 + ((i * 37 + phase) % 21) - 10) for i in range(n)]


def cfg(**metric):
    return config_from_dict({"bootstrap_rounds": 400, "metrics": {"lat": metric}})


def decide(base, cand, **metric):
    return judge({"lat": base}, {"lat": cand}, cfg(**metric))


# -- the three verdicts --------------------------------------------------------------------------------

def test_promotes_when_the_evidence_shows_no_meaningful_change():
    v = decide(lat(600, 1), lat(600, 2))
    assert v.decision == "promote" and v.results[0].status == "ok"


def test_rolls_back_on_a_clear_regression_and_explains_it():
    v = decide(lat(300, 1), lat(300, 2, scale=1.5))
    assert v.decision == "rollback" and v.results[0].status == "regressed"
    assert "beyond the allowed 10%" in v.reasons[0]


def test_extends_when_the_data_cannot_settle_it():
    """A canary exactly at the tolerance: the evidence can neither prove it fine nor prove it too slow."""
    v = decide(wave(80), wave(80, 1.10, phase=5))
    assert v.decision == "extend" and v.results[0].status == "inconclusive"
    assert "straddles" in v.reasons[0]


def test_too_few_samples_is_extend_not_a_guess():
    v = decide(lat(10, 1), lat(10, 2, scale=3))
    assert v.decision == "extend" and v.results[0].status == "insufficient_data"


def test_a_regression_hidden_in_a_big_mean_is_found_with_the_right_statistic():
    base = lat(300, 1)
    tail_heavy = [v * (3 if i % 10 == 0 else 1) for i, v in enumerate(lat(300, 2))]
    assert decide(base, tail_heavy, statistic="median").decision != "rollback"
    assert decide(base, tail_heavy, statistic="p95").decision == "rollback"


# -- direction, modes, SLOs ----------------------------------------------------------------------------

def test_higher_is_better_flips_what_counts_as_a_regression():
    assert decide(lat(300, 1), lat(300, 2, scale=0.6), direction="higher_is_better").decision == "rollback"
    assert decide(lat(300, 1), lat(300, 2, scale=1.6), direction="higher_is_better").decision != "rollback"
    assert decide(lat(300, 1), lat(300, 2, scale=0.6)).decision != "rollback"          # lower latency is not a problem


def test_absolute_mode_handles_a_zero_baseline():
    v = judge({"err": [0.0] * 60}, {"err": [0.02] * 60}, config_from_dict({"metrics": {"err": {"mode": "absolute", "max_regression": 0.01, "statistic": "mean"}}}))
    assert v.decision == "rollback"
    zero = judge({"err": [0.0] * 60}, {"err": [0.02] * 60}, config_from_dict({"metrics": {"err": {"statistic": "mean"}}}))
    assert zero.decision == "extend" and "mode \"absolute\"" in zero.reasons[0]


def test_slo_violation_rolls_back_even_if_the_baseline_was_equally_bad():
    v = decide(lat(300, 1, scale=3), lat(300, 2, scale=3), slo_max=200)
    assert v.decision == "rollback" and v.results[0].status == "slo_violated" and "above the SLO" in v.reasons[0]


def test_slo_min_for_higher_is_better_metrics():
    v = decide(lat(300, 1), lat(300, 2), direction="higher_is_better", slo_min=200)
    assert v.decision == "rollback" and "below the SLO" in v.reasons[0]


def test_an_unproven_slo_blocks_promotion():
    """The canary's median is about 100 and the SLO is 101: close enough that the interval cannot clear it, so do not promote."""
    v = decide(lat(300, 1), lat(300, 2), slo_max=101)
    assert v.decision == "extend" and v.results[0].status == "inconclusive" and "does not yet clear the SLO" in v.reasons[0]


# -- warm-up, missing data, several metrics ------------------------------------------------------------

def test_warmup_points_are_ignored():
    cold = [10_000.0] * 20 + lat(600, 2)
    config = config_from_dict({"warmup_points": 20, "bootstrap_rounds": 400, "metrics": {"lat": {}}})
    assert judge({"lat": lat(620, 1)}, {"lat": cold}, config).decision == "promote"
    no_trim = config_from_dict({"bootstrap_rounds": 400, "metrics": {"lat": {}}})
    assert judge({"lat": lat(620, 1)}, {"lat": cold}, no_trim).decision != "promote"


def test_a_metric_missing_on_either_side_means_extend():
    config = cfg()
    for base, cand in (({}, {"lat": lat(100, 1)}), ({"lat": lat(100, 1)}, {})):
        v = judge(base, cand, config)
        assert v.decision == "extend" and v.results[0].status == "missing"


def test_several_gating_metrics_share_the_false_alarm_budget():
    config = config_from_dict({"alpha": 0.05, "bootstrap_rounds": 400, "metrics": {"a": {}, "b": {}, "c": {}, "d": {"gating": False}}})
    v = judge({n: lat(200, i) for i, n in enumerate("abcd")}, {n: lat(200, i + 9) for i, n in enumerate("abcd")}, config)
    assert v.confidence == pytest.approx(1 - 2 * 0.05 / 3)


def test_non_gating_metrics_warn_but_cannot_roll_back():
    config = config_from_dict({"bootstrap_rounds": 400, "metrics": {"lat": {}, "cpu": {"gating": False}}})
    v = judge({"lat": lat(600, 1), "cpu": lat(300, 1)}, {"lat": lat(600, 2), "cpu": lat(300, 2, scale=2)}, config)
    assert v.decision == "promote" and any("non-gating metric 'cpu'" in w for w in v.warnings)


def test_one_failing_gating_metric_is_enough_to_roll_back():
    config = config_from_dict({"bootstrap_rounds": 400, "metrics": {"lat": {}, "err": {"statistic": "mean"}}})
    v = judge({"lat": lat(300, 1), "err": lat(300, 3)}, {"lat": lat(300, 2), "err": lat(300, 4, scale=2)}, config)
    assert v.decision == "rollback" and [r.name for r in v.results if r.status == "regressed"] == ["err"]


def test_suspicious_inputs_produce_warnings():
    same = lat(100, 1)
    v = judge({"lat": same, "extra": [1.0] * 40}, {"lat": list(same), "extra": [1.0] * 40}, cfg())
    text = " ".join(v.warnings)
    assert "identical sample for sample" in text and "'extra' is in the data but has no rule" in text
    v = judge({"lat": lat(300, 1)}, {"lat": lat(60, 2)}, cfg())
    assert any("three times the samples" in w for w in v.warnings)


def test_strong_autocorrelation_with_few_points_refuses_to_decide():
    sticky = lat(60, 1, phi=0.9)
    v = decide(sticky, lat(60, 2, scale=2.0, phi=0.9))
    assert v.decision == "extend" and "strongly correlated" in v.reasons[0]


# -- reports and command line --------------------------------------------------------------------------

def test_reports_carry_the_verdict_and_the_reasons():
    v = decide(lat(300, 1), lat(300, 2, scale=1.5))
    assert "ROLL BACK" in render_text(v) and "FAIL" in render_text(v)
    assert "## Canary verdict: **ROLL BACK**" in render_markdown(v)
    data = json.loads(render_json(v))
    assert data["decision"] == "rollback" and data["metrics"][0]["status"] == "regressed" and data["metrics"][0]["ci_low"] > 0.1


def files(tmp_path, base, cand, **metric):
    (tmp_path / "base.json").write_text(json.dumps({"lat": base}), encoding="utf-8")
    (tmp_path / "cand.json").write_text(json.dumps({"lat": cand}), encoding="utf-8")
    (tmp_path / "cfg.json").write_text(json.dumps({"bootstrap_rounds": 400, "metrics": {"lat": metric}}), encoding="utf-8")
    return [str(tmp_path / "base.json"), str(tmp_path / "cand.json"), "--config", str(tmp_path / "cfg.json")]


def test_exit_codes_let_a_pipeline_branch_three_ways(tmp_path, capsys):
    assert main(["judge", *files(tmp_path, lat(600, 1), lat(600, 2))]) == 0
    assert main(["judge", *files(tmp_path, lat(300, 1), lat(300, 2, scale=1.6))]) == 1
    assert main(["judge", *files(tmp_path, wave(80), wave(80, 1.10, phase=5))]) == 3
    capsys.readouterr()


def test_cli_formats_output_file_and_input_errors(tmp_path, capsys):
    args = files(tmp_path, lat(300, 1), lat(300, 2, scale=1.6))
    out = tmp_path / "report.md"
    assert main(["judge", *args, "--format", "markdown", "--out", str(out)]) == 1
    assert "Canary verdict" in out.read_text(encoding="utf-8")
    capsys.readouterr()
    assert main(["judge", *args, "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["decision"] == "rollback"
    assert main(["judge", str(tmp_path / "missing.json"), args[1], "--config", args[3]]) == 2
    assert "File not found" in capsys.readouterr().err
    (tmp_path / "cfg.json").write_text(json.dumps({"metrics": {"lat": {"tolerence": 1}}}), encoding="utf-8")
    assert main(["judge", *args]) == 2


def test_nan_gaps_are_reported_as_warnings(tmp_path, capsys):
    body = {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": {"__name__": "lat"}, "values": [[i, "NaN" if i == 5 else str(v)] for i, v in enumerate(lat(300, 1))]}]}}
    (tmp_path / "p.json").write_text(json.dumps(body), encoding="utf-8")
    (tmp_path / "c.json").write_text(json.dumps({"lat": lat(300, 2)}), encoding="utf-8")
    (tmp_path / "cfg.json").write_text(json.dumps({"bootstrap_rounds": 400, "metrics": {"lat": {}}}), encoding="utf-8")
    main(["judge", str(tmp_path / "p.json"), str(tmp_path / "c.json"), "--config", str(tmp_path / "cfg.json")])
    assert "1 NaN gap(s) dropped from baseline 'lat'" in capsys.readouterr().out


def test_examples_in_the_repo_give_their_documented_verdicts(capsys):
    root = Path(__file__).resolve().parent.parent / "examples"
    cases = {"good": 0, "bad": 1, "thin": 3}
    for name, code in cases.items():
        assert main(["judge", str(root / name / "baseline.json"), str(root / name / "canary.json"), "--config", str(root / "config.json")]) == code, name
    capsys.readouterr()
    prom = root / "prometheus"
    assert main(["judge", str(prom / "baseline.json"), str(prom / "canary.json"), "--config", str(prom / "config.json")]) == 1


# -- the judge, judged ---------------------------------------------------------------------------------

def test_a_good_canary_is_almost_never_rolled_back():
    (outcome,) = simulate(samples=200, regressions=[0.0], trials=120, phi=0.0, seed=11, rounds=300)
    assert outcome.rollback_rate <= 0.05


def test_a_clearly_bad_canary_is_almost_always_caught():
    (outcome,) = simulate(samples=600, regressions=[0.30], trials=60, phi=0.0, seed=12, rounds=300)
    assert outcome.rollback / outcome.trials >= 0.95


def test_at_exactly_the_tolerance_the_false_rollback_rate_stays_near_alpha():
    (outcome,) = simulate(samples=200, regressions=[0.10], trials=200, phi=0.0, seed=13, rounds=300)
    assert outcome.rollback_rate <= 0.10               # nominal 5%; the slack covers sampling noise in 200 trials


def test_simulated_series_have_the_requested_shape():
    rng = random.Random(1)
    values = latency_series(5000, rng, median=100, spread=0.35, phi=0.5)
    ordered = sorted(values)
    assert ordered[len(ordered) // 2] == pytest.approx(100, rel=0.06)
