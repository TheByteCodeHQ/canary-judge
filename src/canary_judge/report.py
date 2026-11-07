"""Render a verdict for a terminal, a pull request or another tool."""

from __future__ import annotations

import json

from .judge import MetricResult, Verdict

ICON = {"ok": "PASS", "regressed": "FAIL", "slo_violated": "FAIL", "inconclusive": "????", "insufficient_data": "????", "missing": "????"}
HEADLINE = {"promote": "PROMOTE", "rollback": "ROLL BACK", "extend": "EXTEND THE CANARY"}


def _change(r: MetricResult) -> str:
    if r.change is None:
        return "-"
    scale, unit = (100, "%") if r.relative else (1, "")
    return f"{r.change * scale:+.3g}{unit} ({r.ci_low * scale:+.3g} to {r.ci_high * scale:+.3g})"


def _allowed(r: MetricResult) -> str:
    scale, unit = (100, "%") if r.relative else (1, "")
    return f"{(r.tolerance or 0) * scale:g}{unit}"


def render_text(v: Verdict) -> str:
    lines = [f"Verdict: {HEADLINE[v.decision]}", f"({v.confidence:.1%} intervals per metric, shared across the gating metrics)", ""]
    lines.append(f"{'':<5}{'metric':<28}{'baseline':>11}{'canary':>11}  {'worse by (interval)':<30}{'allowed':>9}")
    for r in v.results:
        b = "-" if r.baseline_stat is None else f"{r.baseline_stat:.4g}"
        c = "-" if r.canary_stat is None else f"{r.canary_stat:.4g}"
        mark = ICON[r.status] + ("" if r.gating else "*")
        lines.append(f"{mark:<5}{r.name:<28}{b:>11}{c:>11}  {_change(r):<30}{_allowed(r):>9}")
    if any(not r.gating for r in v.results):
        lines.append("* not gating: reported, cannot block a release")
    lines += ["", "Why:"] + [f"  - {reason}" for reason in v.reasons]
    lines += [f"Warning: {w}" for w in v.warnings]
    return "\n".join(lines)


def render_markdown(v: Verdict) -> str:
    out = [f"## Canary verdict: **{HEADLINE[v.decision]}**", "", "| | Metric | Baseline | Canary | Worse by (interval) | Allowed |", "|---|---|---|---|---|---|"]
    for r in v.results:
        b = "-" if r.baseline_stat is None else f"{r.baseline_stat:.4g}"
        c = "-" if r.canary_stat is None else f"{r.canary_stat:.4g}"
        out.append(f"| {ICON[r.status]}{'' if r.gating else '*'} | `{r.name}` | {b} | {c} | {_change(r)} | {_allowed(r)} |")
    out += ["", "**Why**", ""] + [f"- {reason}" for reason in v.reasons]
    out += [f"- Warning: {w}" for w in v.warnings]
    return "\n".join(out) + "\n"


def to_dict(v: Verdict) -> dict:
    return {
        "decision": v.decision, "confidence": v.confidence, "reasons": v.reasons, "warnings": v.warnings,
        "metrics": [{"name": r.name, "status": r.status, "gating": r.gating, "baseline": r.baseline_stat, "canary": r.canary_stat,
                     "worse_by": r.change, "ci_low": r.ci_low, "ci_high": r.ci_high, "allowed": r.tolerance,
                     "relative": r.relative, "p_value": r.p_value, "n_baseline": r.n_baseline, "n_canary": r.n_canary, "reasons": r.reasons}
                    for r in v.results],
    }


def render_json(v: Verdict) -> str:
    return json.dumps(to_dict(v), indent=2)
