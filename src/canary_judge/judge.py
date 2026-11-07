"""The decision. For each metric: how much worse is the canary, and how sure are we? Then one verdict for the release.

    promote   every gating metric is shown to be within tolerance
    rollback  at least one gating metric is shown to be worse than tolerance (or breaks its SLO)
    extend    the evidence is not strong enough either way: keep the canary running, do not guess

'Extend' is a first-class answer. A tool that must always say yes or no will be wrong most often exactly when the data is thinnest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config, Rule
from .stats import bootstrap_change, bootstrap_value, effective_samples, mann_whitney, statistic

OK, REGRESSED, SLO, INCONCLUSIVE, INSUFFICIENT, MISSING = "ok", "regressed", "slo_violated", "inconclusive", "insufficient_data", "missing"
FAILED = {REGRESSED, SLO}
UNSURE = {INCONCLUSIVE, INSUFFICIENT, MISSING}


@dataclass
class MetricResult:
    name: str
    status: str
    gating: bool
    reasons: list[str] = field(default_factory=list)
    baseline_stat: float | None = None
    canary_stat: float | None = None
    change: float | None = None                 # how much worse the canary is (relative fraction, or absolute units)
    ci_low: float | None = None
    ci_high: float | None = None
    tolerance: float | None = None
    relative: bool = True
    p_value: float | None = None                # Mann-Whitney: is there any shift at all? Informational, not the decision
    n_baseline: int = 0
    n_canary: int = 0


@dataclass
class Verdict:
    decision: str                                # promote | extend | rollback
    results: list[MetricResult]
    confidence: float                            # the per-metric interval level actually used
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def evaluate_metric(rule: Rule, base: list[float] | None, cand: list[float] | None, config: Config, confidence: float) -> MetricResult:
    result = MetricResult(rule.name, INSUFFICIENT, rule.gating, tolerance=rule.max_regression, relative=rule.mode == "relative")
    if base is None or cand is None:
        side = "baseline" if base is None else "canary"
        result.status, result.reasons = MISSING, [f"the {side} has no data for this metric"]
        return result
    base, cand = base[config.warmup_points:], cand[config.warmup_points:]
    result.n_baseline, result.n_canary = len(base), len(cand)
    need = rule.min_samples or config.min_samples
    if len(base) < need or len(cand) < need:
        result.reasons = [f"needs {need} samples on each side after warm-up, has {len(base)} baseline and {len(cand)} canary"]
        return result

    ess_b, ess_c = effective_samples(base), effective_samples(cand)
    if min(ess_b, ess_c) < need:
        result.reasons = [f"the samples are strongly correlated over time: {len(base)} baseline and {len(cand)} canary points are worth about "
                          f"{ess_b:.0f} and {ess_c:.0f} independent ones, and {need} are needed. Keep the canary running or sample less often"]
        return result

    stat = statistic(rule.statistic)
    result.baseline_stat, result.canary_stat = stat(base), stat(cand)
    if rule.mode == "relative" and result.baseline_stat == 0:
        result.status = INCONCLUSIVE
        result.reasons = [f"the baseline {rule.statistic} is 0, so a relative change is undefined. Use mode \"absolute\" for this metric"]
        return result
    point, low, high = bootstrap_change(base, cand, stat, relative=rule.mode == "relative", sign=rule.sign, confidence=confidence,
                                        rounds=config.bootstrap_rounds, seed=config.seed, block=config.block)
    result.change, result.ci_low, result.ci_high = point, low, high
    result.p_value = mann_whitney(base, cand)[1]

    unit = "%" if rule.mode == "relative" else ""
    scale = 100 if rule.mode == "relative" else 1
    allowed = f"{rule.max_regression * scale:g}{unit}"
    if low > rule.max_regression:
        result.status = REGRESSED
        result.reasons.append(f"worse by {point * scale:+.3g}{unit} (interval {low * scale:+.3g} to {high * scale:+.3g}{unit}), clearly beyond the allowed {allowed}")
    elif high <= rule.max_regression:
        result.status = OK
    else:
        result.status = INCONCLUSIVE
        result.reasons.append(f"worse by {point * scale:+.3g}{unit} (interval {low * scale:+.3g} to {high * scale:+.3g}{unit}); the interval straddles the allowed {allowed}")

    if rule.slo_max is not None or rule.slo_min is not None:
        value, v_low, v_high = bootstrap_value(cand, stat, confidence=confidence, rounds=config.bootstrap_rounds, seed=config.seed, block=config.block)
        if rule.slo_max is not None and v_low > rule.slo_max:
            result.status = SLO
            result.reasons.append(f"canary {rule.statistic} is {value:.4g}, above the SLO of {rule.slo_max:g} (interval {v_low:.4g} to {v_high:.4g})")
        elif rule.slo_min is not None and v_high < rule.slo_min:
            result.status = SLO
            result.reasons.append(f"canary {rule.statistic} is {value:.4g}, below the SLO of {rule.slo_min:g} (interval {v_low:.4g} to {v_high:.4g})")
        elif result.status == OK and ((rule.slo_max is not None and v_high > rule.slo_max) or (rule.slo_min is not None and v_low < rule.slo_min)):
            result.status = INCONCLUSIVE
            result.reasons.append(f"canary {rule.statistic} is {value:.4g}; its interval ({v_low:.4g} to {v_high:.4g}) does not yet clear the SLO")
    return result


def judge(baseline: dict[str, list[float]], canary: dict[str, list[float]], config: Config) -> Verdict:
    gating = [r for r in config.rules if r.gating]
    # Several metrics means several chances for noise to look like a regression. Splitting the false-alarm budget keeps the whole
    # verdict's chance of a spurious rollback near alpha instead of near alpha times the number of metrics.
    # Each decision is one-sided ("is it worse than tolerance?"), so a two-sided interval at 1 - 2 * alpha_metric gives alpha_metric per side.
    confidence = 1 - 2 * config.alpha / len(gating)
    results = [evaluate_metric(r, baseline.get(r.name), canary.get(r.name), config, confidence if r.gating else 1 - 2 * config.alpha) for r in config.rules]

    warnings: list[str] = []
    named = {r.name for r in config.rules}
    for extra in sorted((set(baseline) | set(canary)) - named):
        warnings.append(f"'{extra}' is in the data but has no rule, so it was ignored")
    for res in results:
        if not res.gating and res.status in FAILED:
            warnings.append(f"non-gating metric '{res.name}' is {res.status.replace('_', ' ')}: {'; '.join(res.reasons)}")
        if res.n_baseline and res.n_canary and max(res.n_baseline, res.n_canary) > 3 * min(res.n_baseline, res.n_canary):
            warnings.append(f"'{res.name}': one side has over three times the samples of the other ({res.n_baseline} vs {res.n_canary}); are the windows comparable?")
    for name in named & set(baseline) & set(canary):
        if len(baseline[name]) > 5 and baseline[name] == canary[name]:
            warnings.append(f"'{name}': baseline and canary are identical sample for sample. Both queries probably read the same series")

    failed = [r for r in results if r.gating and r.status in FAILED]
    unsure = [r for r in results if r.gating and r.status in UNSURE]
    if failed:
        decision, reasons = "rollback", [f"{r.name}: {'; '.join(r.reasons)}" for r in failed]
    elif unsure:
        decision, reasons = "extend", [f"{r.name}: {'; '.join(r.reasons) or r.status}" for r in unsure]
    else:
        decision, reasons = "promote", ["every gating metric is within tolerance, with the evidence to show it"]
    return Verdict(decision, results, confidence, reasons, warnings)
