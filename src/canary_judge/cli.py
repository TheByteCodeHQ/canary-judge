"""Command line: canary-judge judge | simulate.

Exit codes for judge: 0 promote, 1 roll back, 3 extend (not enough evidence), 2 bad input. So a pipeline can branch on all three.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, load_config
from .judge import judge
from .loaders import LoadError, load_series
from .report import render_json, render_markdown, render_text
from .simulate import simulate

EXIT = {"promote": 0, "rollback": 1, "extend": 3}
USAGE = 2


def cmd_judge(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    baseline, skipped_b = load_series(Path(args.baseline))
    canary, skipped_c = load_series(Path(args.canary))
    verdict = judge(baseline, canary, config)
    for side, skipped in (("baseline", skipped_b), ("canary", skipped_c)):
        for name, count in sorted(skipped.items()):
            verdict.warnings.append(f"{count} NaN gap(s) dropped from {side} '{name}'")
    text = {"text": render_text, "markdown": render_markdown, "json": render_json}[args.format](verdict)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Verdict: {verdict.decision} (report written to {args.out})")
    else:
        print(text)
    return EXIT[verdict.decision]


def cmd_simulate(args: argparse.Namespace) -> int:
    regressions = [float(x) for x in args.regressions.split(",")]
    print(f"{args.trials} trials per row, {args.samples} samples per side, tolerance {args.tolerance:.0%}, autocorrelation {args.phi}")
    print(f"{'true regression':>16} {'promote':>8} {'extend':>8} {'rollback':>9}")
    for o in simulate(samples=args.samples, regressions=regressions, trials=args.trials, tolerance=args.tolerance, phi=args.phi, seed=args.seed):
        print(f"{o.regression:>16.0%} {o.promote / o.trials:>8.0%} {o.extend / o.trials:>8.0%} {o.rollback / o.trials:>9.0%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="canary-judge", description="Is this canary safe to promote? Promote, roll back, or keep watching.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    j = sub.add_parser("judge", help="compare canary metrics with the baseline")
    j.add_argument("baseline", help="baseline samples (JSON map, CSV or Prometheus query_range response)")
    j.add_argument("canary")
    j.add_argument("--config", required=True)
    j.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    j.add_argument("--out", help="write the report to a file")
    j.set_defaults(func=cmd_judge)
    s = sub.add_parser("simulate", help="measure the judge's false-alarm and detection rates on synthetic data")
    s.add_argument("--samples", type=int, default=60)
    s.add_argument("--regressions", default="0,0.05,0.10,0.20,0.30")
    s.add_argument("--trials", type=int, default=100)
    s.add_argument("--tolerance", type=float, default=0.10)
    s.add_argument("--phi", type=float, default=0.5, help="autocorrelation of the simulated series (0 = independent)")
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_simulate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LoadError, ConfigError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE


if __name__ == "__main__":
    raise SystemExit(main())
