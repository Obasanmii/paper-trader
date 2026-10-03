"""The research protocol.

1. Split the history once, up front: in-sample (IS) and out-of-sample (OOS).
2. Try every parameter combination on IS only. Count them: that count is
   the number of trials, and it matters.
3. Pick the best by IS Sharpe, then *deflate* that Sharpe for the number of
   trials (DSR). Picking the best of 20 noise strategies looks impressive
   and means nothing; the DSR knows that.
4. Run the winner, untouched, on OOS. Bootstrap a confidence interval and
   run the timing test. Compare with the benchmark over the same window.
5. Write down the verdict whatever it is. Don't re-split, don't re-sweep
   until OOS looks good: that just turns OOS into IS.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import pandas as pd

from papertrader.engine.backtest import BacktestResult, evaluation_start, run_backtest
from papertrader.metrics import sharpe
from papertrader.strategies import build_strategy
from papertrader.validation import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    timing_test,
)


@dataclass
class Trial:
    params: dict
    sharpe: float
    sharpe_per_period: float
    total_return: float
    max_drawdown: float
    kill_switch: str | None


@dataclass
class ResearchResult:
    strategy: str
    split_date: pd.Timestamp
    eval_start: pd.Timestamp
    trials: list[Trial]
    best: Trial
    best_params: dict
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    benchmark_in: BacktestResult
    benchmark_out: BacktestResult
    psr_in: float
    dsr_in: float
    oos_sharpe_ci: tuple
    oos_timing: dict
    verdict: list[str] = field(default_factory=list)
    passed: int = 0
    checks: int = 0
    check_results: dict = field(default_factory=dict)  # check name -> passed?


def run_research(data, cfg, strategy_name: str | None = None, param_grid: dict | None = None) -> ResearchResult:
    rc = cfg.research
    name = strategy_name or cfg.strategy.name
    grid = param_grid if param_grid is not None else rc.param_grid
    if not rc.split_date:
        raise ValueError("research.split_date is required")
    split = pd.Timestamp(rc.split_date)
    eval_start = evaluation_start(data, rc.warmup_days)
    oos_dates = data.dates[data.dates > split]
    if split <= eval_start:
        raise ValueError(f"split_date {split.date()} is before the evaluation start {eval_start.date()}")
    if len(oos_dates) < 126:
        raise ValueError("need at least ~6 months of out-of-sample data after split_date")
    in_data = data.truncate(split)
    run = dict(portfolio=cfg.portfolio, costs=cfg.costs, limits=cfg.risk)

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys))) if keys else [()]
    trials, results = [], []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        strategy = build_strategy(name, {**cfg.strategy.params, **overrides})
        res = run_backtest(in_data, strategy, start=eval_start, **run)
        r = res.returns()
        s = res.summary()
        trials.append(
            Trial(overrides, sharpe(r), per_period_sharpe(r), s["total_return"], s["max_drawdown"], res.kill_switch_reason)
        )
        results.append(res)

    ranked = [(t.sharpe if math.isfinite(t.sharpe) else -math.inf, k) for k, t in enumerate(trials)]
    best_idx = max(ranked)[1]
    best = trials[best_idx]
    best_params = {**cfg.strategy.params, **best.params}
    in_sample = results[best_idx]
    out_of_sample = run_backtest(data, build_strategy(name, best_params), start=oos_dates[0], **run)
    bench = build_strategy(rc.benchmark, {})
    benchmark_in = run_backtest(in_data, bench, start=eval_start, **run)
    benchmark_out = run_backtest(data, bench, start=oos_dates[0], **run)

    psr = probabilistic_sharpe_ratio(in_sample.returns())
    dsr = deflated_sharpe_ratio(in_sample.returns(), [t.sharpe_per_period for t in trials])
    ci = bootstrap_sharpe_ci(out_of_sample.returns(), n_boot=rc.bootstrap_samples)
    timing = timing_test(out_of_sample.weights, data, start=oos_dates[0], n_perm=rc.timing_permutations)

    result = ResearchResult(
        strategy=name,
        split_date=split,
        eval_start=eval_start,
        trials=trials,
        best=best,
        best_params=best_params,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        benchmark_in=benchmark_in,
        benchmark_out=benchmark_out,
        psr_in=psr,
        dsr_in=dsr,
        oos_sharpe_ci=ci,
        oos_timing=timing,
    )
    _write_verdict(result)
    return result


def _write_verdict(res: ResearchResult) -> None:
    n = len(res.trials)
    is_sr = res.best.sharpe
    oos_sr = sharpe(res.out_of_sample.returns())
    bench_sr = sharpe(res.benchmark_out.returns())
    lo, hi = res.oos_sharpe_ci
    p = res.oos_timing.get("p_value", math.nan)
    lines, passed, checks = [], 0, 0
    results = {}

    lines.append(f"Best of {n} in-sample trial(s): Sharpe {is_sr:.2f}, probabilistic Sharpe {res.psr_in:.2f}.")
    if n >= 2:
        checks += 1
        results["deflated_sharpe"] = res.dsr_in >= 0.95
        if res.dsr_in >= 0.95:
            passed += 1
            lines.append(f"PASS  Deflated for {n} trials, the in-sample result still stands (DSR {res.dsr_in:.2f}).")
        else:
            lines.append(
                f"FAIL  Deflated for {n} trials, the in-sample result is consistent with luck "
                f"(DSR {res.dsr_in:.2f}, want >= 0.95)."
            )
    checks += 1
    results["oos_interval_above_zero"] = bool(math.isfinite(lo) and lo > 0)
    if math.isfinite(lo) and lo > 0:
        passed += 1
        lines.append(f"PASS  Out-of-sample Sharpe {oos_sr:.2f}; 95% interval [{lo:.2f}, {hi:.2f}] excludes zero.")
    else:
        lines.append(f"FAIL  Out-of-sample Sharpe {oos_sr:.2f}; 95% interval [{lo:.2f}, {hi:.2f}] includes zero.")
    checks += 1
    results["beats_benchmark_oos"] = bool(math.isfinite(oos_sr) and math.isfinite(bench_sr) and oos_sr > bench_sr)
    if results["beats_benchmark_oos"]:
        passed += 1
        lines.append(f"PASS  Beat the benchmark's out-of-sample Sharpe ({bench_sr:.2f}).")
    else:
        lines.append(f"FAIL  Did not beat the benchmark's out-of-sample Sharpe ({bench_sr:.2f}).")
    checks += 1
    results["timing"] = bool(math.isfinite(p) and p < 0.05)
    if results["timing"]:
        passed += 1
        lines.append(f"PASS  Timing beats randomly time-shifted copies of itself (p = {p:.3f}).")
    else:
        lines.append(f"FAIL  Timing does not beat randomly time-shifted copies of itself (p = {p:.2f}).")
    if res.out_of_sample.kill_switch_reason:
        lines.append(f"NOTE  The kill switch tripped out-of-sample: {res.out_of_sample.kill_switch_reason}.")
    if passed == checks:
        lines.append("Survives every check here. That earns it paper trading, not trust.")
    else:
        lines.append(f"Passed {passed} of {checks} checks. Treat the in-sample numbers as an upper bound, not an estimate.")
    res.verdict, res.passed, res.checks, res.check_results = lines, passed, checks, results
