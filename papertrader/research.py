"""The research protocol.

1. Split the history once, up front: in-sample (IS) and out-of-sample (OOS).
2. Try every parameter combination on IS only. Count them: that count is
   the number of trials, and it matters.
3. Pick the best by IS Sharpe, then *deflate* that Sharpe for the number of
   trials (DSR). Picking the best of 20 noise strategies looks impressive
   and means nothing; the DSR knows that.
4. Run the winner, untouched, on OOS. Bootstrap a confidence interval and
   run the timing test. Compare with the benchmark over the same window
   with a paired bootstrap: a bare "higher Sharpe" is a coin flip on noise.
5. Write down the verdict whatever it is. Don't re-split, don't re-sweep
   until OOS looks good: that just turns OOS into IS.

With a ledger (see ledger.py), step 2's count includes every parameter set
ever tried on this strategy and matching data, and a look at an OOS window
that an earlier run already looked at is flagged.
"""
from __future__ import annotations

import contextlib
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from papertrader.engine.backtest import BacktestResult, evaluation_start, run_backtest
from papertrader.ledger import LedgerTrial, Prior, TrialLedger, params_key, study
from papertrader.metrics import sharpe
from papertrader.strategies import build_strategy
from papertrader.validation import (
    bootstrap_sharpe_ci,
    bootstrap_sharpe_difference,
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
    oos_vs_benchmark: dict  # bootstrap_sharpe_difference(strategy, benchmark) out-of-sample
    n_trials_total: int = 0  # distinct parameter sets ever tried on this strategy and data (= len(trials) without a ledger)
    n_trials_prior: int = 0  # ...of which only earlier runs tried
    oos_prior_evaluations: int = 0  # earlier runs whose out-of-sample window overlaps this one
    ledger_run_id: str | None = None  # this run's record in the trial ledger; None = no ledger
    verdict: list[str] = field(default_factory=list)
    passed: int = 0
    checks: int = 0
    check_results: dict = field(default_factory=dict)  # check name -> passed?


def run_research(
    data, cfg, strategy_name: str | None = None, param_grid: dict | None = None, ledger: str | Path | None = None
) -> ResearchResult:
    """Run the protocol. `ledger` is a TrialLedger path; None remembers nothing between runs."""
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

    # Opened before the sweep, so a broken ledger fails in seconds rather than after it.
    with TrialLedger(ledger) if ledger is not None else contextlib.nullcontext() as book:
        trials, results = _sweep(in_data, name, cfg.strategy.params, grid, eval_start, run)
        ranked = [(t.sharpe if math.isfinite(t.sharpe) else -math.inf, k) for k, t in enumerate(trials)]
        best_idx = max(ranked)[1]
        best = trials[best_idx]
        best_params = {**cfg.strategy.params, **best.params}
        in_sample = results[best_idx]
        out_of_sample = run_backtest(data, build_strategy(name, best_params), start=oos_dates[0], **run)
        bench = build_strategy(rc.benchmark, {})
        benchmark_in = run_backtest(in_data, bench, start=eval_start, **run)
        benchmark_out = run_backtest(data, bench, start=oos_dates[0], **run)

        mine = _ledger_trials(trials, cfg.strategy.params)
        this, prior = None, None
        if book is not None:
            this = study(name, cfg.data.source, data, split, rc.warmup_days, oos_dates[0], oos_dates[-1])
            prior = book.prior(this)

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
            psr_in=probabilistic_sharpe_ratio(in_sample.returns()),
            dsr_in=math.nan,  # set by _count_trials
            oos_sharpe_ci=bootstrap_sharpe_ci(out_of_sample.returns(), n_boot=rc.bootstrap_samples),
            oos_timing=timing_test(out_of_sample.weights, data, start=oos_dates[0], n_perm=rc.timing_permutations),
            oos_vs_benchmark=bootstrap_sharpe_difference(
                out_of_sample.returns(), benchmark_out.returns(), n_boot=rc.bootstrap_samples
            ),
        )
        _count_trials(result, mine, prior)
        _write_verdict(result)
        if book is not None:  # recorded before anyone sees the verdict: no result without its record
            result.ledger_run_id, seen = book.record_run(this, list(mine.values()), best_params)
            if seen.run_ids != prior.run_ids:  # another run was recorded meanwhile: count it too
                _count_trials(result, mine, seen)
                _write_verdict(result)
    return result


def _count_trials(res: ResearchResult, mine: dict[str, LedgerTrial], prior: Prior | None) -> None:
    """Deflate for every distinct parameter set tried on this strategy and data, earlier runs included."""
    every = {**prior.trials, **mine} if prior is not None else mine  # one entry per parameter set; this run's values win
    res.n_trials_total, res.n_trials_prior = len(every), len(every) - len(mine)
    res.oos_prior_evaluations = prior.oos_overlaps if prior is not None else 0
    res.dsr_in = deflated_sharpe_ratio(res.in_sample.returns(), [t.sharpe_per_period for t in every.values()])


def _sweep(in_data, name: str, base_params: dict, grid: dict, eval_start, run: dict) -> tuple[list[Trial], list[BacktestResult]]:
    """Backtest every grid combination on the in-sample data only."""
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys))) if keys else [()]
    trials, results = [], []
    for combo in combos:
        overrides = dict(zip(keys, combo, strict=True))
        res = run_backtest(in_data, build_strategy(name, {**base_params, **overrides}), start=eval_start, **run)
        r = res.returns()
        s = res.summary()
        trials.append(
            Trial(overrides, sharpe(r), per_period_sharpe(r), s["total_return"], s["max_drawdown"], res.kill_switch_reason)
        )
        results.append(res)
    return trials, results


def _ledger_trials(trials: list[Trial], base_params: dict) -> dict[str, LedgerTrial]:
    """This run's trials keyed by the full parameter set each one ran with, not just the grid's part,
    so a change to strategy.params can't make a new strategy look like an old trial."""
    out = {}
    for t in trials:
        p = params_key({**base_params, **t.params})
        out[p] = LedgerTrial(p, t.sharpe, t.sharpe_per_period)
    return out


def _signed(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:+.2f}"


def _write_verdict(res: ResearchResult) -> None:
    n, earlier = res.n_trials_total, res.n_trials_prior
    is_sr = res.best.sharpe
    oos_sr = sharpe(res.out_of_sample.returns())
    bench_sr = sharpe(res.benchmark_out.returns())
    lo, hi = res.oos_sharpe_ci
    p = res.oos_timing.get("p_value", math.nan)
    lines, passed, checks = [], 0, 0
    results = {}

    if res.oos_prior_evaluations:
        # Not a pass/fail check: the numbers below are still computed the same way, but
        # they are no longer an honest test. Said first so nobody reads past it.
        lines.append(
            f"WARN  The out-of-sample period, or one overlapping it, was already evaluated {res.oos_prior_evaluations} "
            "time(s) for this strategy and data; it is no longer a clean holdout."
        )
    # The best comes from this run's grid only; earlier runs' trials only raise the bar it is deflated against.
    best_of = f"this run's {len(res.trials)}" if earlier else f"{len(res.trials)}"
    tried = f" (deflated for {n} tried on this strategy and data, {earlier} in earlier runs)" if earlier else ""
    lines.append(f"Best of {best_of} in-sample trial(s){tried}: Sharpe {is_sr:.2f}, probabilistic Sharpe {res.psr_in:.2f}.")
    if n >= 2:
        checks += 1
        results["deflated_sharpe"] = res.dsr_in >= 0.95
        if res.dsr_in >= 0.95:
            passed += 1
            lines.append(f"PASS  Deflated for {n} trials, the in-sample result still stands (DSR {res.dsr_in:.2f}).")
        elif math.isnan(res.dsr_in):
            lines.append(
                f"FAIL  Deflated for {n} trials, no DSR can be computed (fewer than 2 distinct results among them, "
                "e.g. most never traded), so the in-sample result doesn't stand."
            )
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
    diff, bound = res.oos_vs_benchmark.get("difference", math.nan), res.oos_vs_benchmark.get("lower", math.nan)
    results["beats_benchmark_oos"] = bool(math.isfinite(bound) and bound > 0)
    detail = f"Sharpe difference {_signed(diff)}, one-sided 95% lower bound {_signed(bound)}"
    flat = res.oos_vs_benchmark.get("n_degenerate", 0)
    note = (
        f" {flat} of {res.oos_vs_benchmark.get('n_boot')} resamples had a series with no variance "
        "(all zeros scored Sharpe 0, anything else a loss)."
        if flat
        else ""
    )
    if results["beats_benchmark_oos"]:
        passed += 1
        lines.append(f"PASS  Beat the benchmark out-of-sample: {detail} (benchmark Sharpe {bench_sr:.2f}).{note}")
    else:
        lines.append(
            f"FAIL  Did not beat the benchmark out-of-sample: {detail} (want > 0; benchmark Sharpe {bench_sr:.2f}).{note}"
        )
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
