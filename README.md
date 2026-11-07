# canary-judge

**Is this canary safe to promote?** Compare a canary deployment's metrics with the baseline and get one of three answers, with the
evidence: **promote**, **roll back**, or **extend** (keep watching, because the data cannot settle it yet). Standard library only.

```
$ canary-judge judge examples/bad/baseline.json examples/bad/canary.json --config examples/config.json
Verdict: ROLL BACK

     metric                         baseline     canary  worse by (interval)             allowed
FAIL latency_p95_ms                    179.4      240.8  +34.2% (+27.6 to +39.3)             10%
FAIL error_rate                     0.001986   0.006089  +0.0041 (+0.0039 to +0.0043)      0.002
FAIL*cpu_percent                       55.04      76.95  +39.8% (+37.5 to +41.5)             25%
* not gating: reported, cannot block a release
```

## Why not just compare the two averages?

A canary has tens or hundreds of noisy samples. "p95 went from 180 to 186ms" might be a regression or might be Tuesday. This tool
answers the question that matters: **how much worse is the canary, and how sure are we?**

| Problem with the naive check | What this does |
|---|---|
| A bare number says nothing about noise | Every change comes with a **confidence interval** (bootstrap) |
| Latency is not bell-shaped | Medians and percentiles, resampled; no normality assumption. Mann-Whitney is reported too |
| Samples over time are not independent | **Block bootstrap** sized to the series' own memory, and an **effective sample size** check: 60 points with strong autocorrelation are worth far fewer |
| "Not significantly different" is read as "fine" | Promotion needs the interval to sit **inside** the tolerance. Unclear is *extend*, never a quiet yes |
| Five metrics means five chances for a false alarm | The false-alarm budget `alpha` is **shared** across gating metrics |
| Cold caches and JIT skew the start | `warmup_points` drops the first N samples |
| The baseline was already over the limit | Absolute **SLO** checks on the canary itself (`slo_max` / `slo_min`) |
| A zero baseline makes percentages meaningless | `mode: absolute`, and a clear message if you forget |
| A broken query hides a problem | Warnings for identical series, wildly unequal windows, metrics with no rule, NaN gaps |

## The three verdicts

| Verdict | Meaning | Exit code |
|---|---|---|
| **promote** | Every gating metric is *shown* to be within tolerance | `0` |
| **rollback** | At least one gating metric is *shown* to be worse than tolerance, or breaks its SLO | `1` |
| **extend** | Not enough evidence either way (too few samples, interval straddles the limit, metric missing) | `3` |
| (bad input) | Config or data could not be read | `2` |

A pipeline can branch on all three: promote on 0, roll back on 1, wait and re-run on 3.

## Input

Baseline and canary files can be any of:

```jsonc
{"metrics": {"latency_p95_ms": [181.2, 179.8, ...]}}          // a JSON map of samples in time order
```
```csv
timestamp,metric,value                                         // CSV (sorted by timestamp if present)
1700000000,latency_p95_ms,181.2
```
or a raw **Prometheus `query_range` response** (`resultType: matrix`). Each series needs a `__name__` (or `name`) label; NaN gaps
are dropped and counted, infinities are rejected.

## Config

```json
{
  "alpha": 0.05,
  "warmup_points": 10,
  "min_samples": 30,
  "metrics": {
    "latency_p95_ms": {"statistic": "median", "max_regression": 0.10, "slo_max": 300},
    "error_rate":     {"statistic": "mean", "mode": "absolute", "max_regression": 0.002},
    "cpu_percent":    {"statistic": "mean", "max_regression": 0.25, "gating": false}
  }
}
```

Per metric: `direction` (`lower_is_better` default, or `higher_is_better`), `statistic` (`median`, `mean`, `p90`, `p95`, `p99`),
`max_regression` and `mode` (`relative` fraction or `absolute` units), `gating`, `slo_max`, `slo_min`, `min_samples`.
`min_samples` counts **effective independent** samples. A misspelt key is an error, because a typo that silently weakens a gate is
worse than no gate.

## Use it in a pipeline

```bash
canary-judge judge baseline.json canary.json --config canary.json --format markdown --out verdict.md
case $? in
  0) kubectl argo rollouts promote checkout ;;
  1) kubectl argo rollouts abort checkout ;;
  3) echo "not enough evidence yet; waiting" ;;
  *) echo "judge could not run"; exit 1 ;;
esac
```

`--format json` gives a machine-readable verdict; `--format markdown` is ready to post on a pull request or in chat.

## How good is the judge? It tests itself

The decision rule is worth nothing if it is not calibrated, so `canary-judge simulate` measures it on synthetic latency where the
truth is known (log-normal, with optional autocorrelation). Each row is 300 trials, tolerance 10%.

**200 samples per side, independent** (the textbook case):

| True regression | Promote | Extend | Roll back |
|---|---|---|---|
| 0% | 69% | 31% | **0%** |
| 5% | 24% | 76% | 0% |
| 10% (the tolerance itself) | 8% | 88% | 4% |
| 20% | 0% | 41% | 59% |
| 30% | 0% | 1% | **99%** |

**600 samples per side, autocorrelation 0.5** (a realistic time series):

| True regression | Promote | Extend | Roll back |
|---|---|---|---|
| 0% | 82% | 18% | **0%** |
| 5% | 34% | 66% | 0% |
| 10% | 7% | 85% | 7% |
| 20% | 0% | 23% | 77% |
| 30% | 0% | 0% | **100%** |

What this says, plainly:

- **A good canary is never rolled back** (0% at no regression, in all six settings above).
- **At exactly the tolerance, the false rollback rate is between 1% and 7%** across the six settings I ran, around the 5% it is designed for.
- **It would rather wait than guess.** With 60 samples at autocorrelation 0.5 it says *extend* about 99% of the time, because that
  data cannot prove anything. Proving "within 10%" takes hundreds of samples. If you want faster answers, loosen the tolerance or
  sample more often; do not expect a smaller sample to be more certain.
- **A 20% regression with 200 samples is caught only 59% of the time** (38% at autocorrelation 0.5); the rest are *extend*, not *promote*. The failure mode is delay,
  not a bad release.

The tool changed because of these runs: my first version used two-sided 95% intervals and almost never promoted anything; it is now
one-sided (which is what "is it worse than X?" is). With strongly autocorrelated data it was also too eager to roll back, which is
why the effective-sample-size check exists.

## Tested

45 tests on Python 3.10 and 3.12, Linux and Windows. They include statistical properties (interval coverage over 40 paired
comparisons, calibration of the false-rollback and detection rates through the simulator) and failure paths (malformed
Prometheus responses, NaN and infinite values, zero baselines, strong autocorrelation). Where a test needs a borderline case it
uses deterministic data, not a lucky random seed. I also broke the code on purpose in sixteen ways (comparing the point estimate
instead of the interval, dropping the shared false-alarm budget, ignoring warm-up, letting a non-gating metric block, accepting
infinities, ...) and confirmed a test fails each time.

## Limits

- **Validated on simulated data.** The calibration above is on log-normal series I generated. Real traffic has daily cycles, deploys,
  noisy neighbours and heavy tails, and the numbers there will differ. Run `simulate` with your own noise level, and treat the first
  weeks as a shadow run next to your current process.
- **The two groups must be comparable.** The judge cannot know that the canary got a different mix of customers. Compare like with
  like (same time window, same traffic share, a baseline from the *same* time, not last week).
- **Autocorrelation handling is an approximation** (block bootstrap plus an effective-sample-size floor). It held up in my
  simulations at autocorrelation 0.5. At 0.8 with a few hundred samples it declines to decide (the safe failure); with 600 samples it decides,
  and rolled back about 8% of canaries sitting exactly on the tolerance, a bit above the design's 5%.
- **It judges metrics, not releases.** A canary can pass every metric and still break a feature nobody measures.
- **It does not shift traffic or roll anything back.** It returns a verdict; your pipeline acts on it.
- No live Prometheus connection: you give it files (a `query_range` response saved with `curl`).
- One baseline, one canary. No multi-variant or sequential testing.

## Layout

```
src/canary_judge/
  stats.py      percentiles, Mann-Whitney, block bootstrap, effective sample size
  loaders.py    JSON, CSV and Prometheus query_range, with strict number checks
  config.py     rules and their validation
  judge.py      the decision and its reasons
  report.py     text, Markdown, JSON
  simulate.py   the self-test on synthetic data
  cli.py        canary-judge judge | simulate
examples/       good, bad and thin canaries (+ a Prometheus-format one) and the generator
```

## License

MIT
