"""Markdown reports with charts. Plain files you can commit, diff and share."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

_PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

ROWS = [
    ("Total return", "total_return", "pct"),
    ("CAGR", "cagr", "pct"),
    ("Volatility", "volatility", "pct"),
    ("Sharpe", "sharpe", "num"),
    ("Sortino", "sortino", "num"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Longest drawdown (days)", "max_drawdown_days", "int"),
    ("Avg gross exposure", "avg_gross_exposure", "pct"),
    ("Turnover (x equity / yr)", "turnover_per_year", "num"),
    ("Fills", "fills", "int"),
    ("Costs paid (% of start)", "costs_pct_of_start", "pct"),
    ("Orders rejected by risk", "rejected_orders", "int"),
    ("Kill switch", "kill_switch", "kill"),
]


def fmt(value, kind: str) -> str:
    if kind == "kill":
        return "tripped" if value else "never tripped"
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "int":
        return f"{int(value):,}"
    return f"{value:.2f}"


def metrics_table(summaries: dict[str, dict]) -> str:
    names = list(summaries)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key, kind in ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(summaries[n].get(key), kind) for n in names) + " |")
    return "\n".join(lines)


def plot_equity(curves: dict[str, pd.Series], path: Path, title: str, marks: dict[str, pd.Timestamp] | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for label, eq in curves.items():
        ax1.plot(eq.index, eq / eq.iloc[0], label=label, linewidth=1.3)
        ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, linewidth=1.0)
    for label, when in (marks or {}).items():
        for ax in (ax1, ax2):
            ax.axvline(when, color="grey", linestyle="--", linewidth=0.9)
        ax1.annotate(
            label, (when, ax1.get_ylim()[1]), fontsize=8, color="grey", va="top", ha="left",
            xytext=(3, -3), textcoords="offset points",
        )
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(_PLAIN)
    ax1.yaxis.set_minor_formatter(_PLAIN)
    ax1.tick_params(axis="y", which="minor", labelsize=7)
    ax1.set_ylabel("Growth of 1 (log scale)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def backtest_report(cfg, result, bench, validation: dict, cleaning, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curves = {result.name: result.equity, f"benchmark: {bench.name}": bench.equity}
    plot_equity(curves, out / "equity.png", f"{cfg.name}: backtest")
    s, b = result.summary(), bench.summary()
    lo, hi = validation["sharpe_ci"]
    timing = validation["timing"]
    lines = [
        f"# Backtest: {cfg.name}",
        "",
        f"Strategy `{result.name}` vs benchmark `{bench.name}`, {s['start']} to {s['end']} ({s['days']:,} trading days).",
        f"Data source: `{cfg.data.source}` · symbols: {', '.join(cfg.data.symbols)} · "
        f"costs: {cfg.costs.commission_bps:g} bps commission + {cfg.costs.slippage_bps:g} bps slippage.",
        "",
        "![equity](equity.png)",
        "",
        metrics_table({"Strategy": s, "Benchmark": b}),
        "",
        "## Should we believe it?",
        "",
        f"- Probabilistic Sharpe ratio (P[true Sharpe > 0]): **{fmt(validation['psr'], 'num')}** (want at least 0.95).",
        f"- 95% bootstrap interval for the Sharpe ratio: [{fmt(lo, 'num')}, {fmt(hi, 'num')}].",
        f"- Timing test: gross Sharpe {fmt(timing.get('sharpe'), 'num')} vs randomly time-shifted copies "
        f"(95th percentile {fmt(timing.get('null_95th'), 'num')}), p = {fmt(timing.get('p_value'), 'num')}.",
        "- This is a single backtest with no multiple-testing correction. Use `research` to sweep parameters honestly.",
        "",
        "## Risk layer",
        "",
        f"- Strategy kill switch: {result.kill_switch_reason or 'never tripped'}.",
        f"- Benchmark kill switch: {bench.kill_switch_reason or 'never tripped'}.",
        f"- Orders rejected: {len(result.rejections)}.",
    ]
    if len(result.rejections):
        top = result.rejections["reasons"].str.split(";").str[0].str.replace(r"[\d.,%]+", "#", regex=True).value_counts().head(5)
        for reason, count in top.items():
            lines.append(f"  - {count} x {reason.strip()}")
    lines += ["", "## Data cleaning", "", "```", cleaning.summary(), "```", ""]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path


def research_report(cfg, res, cleaning, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_curve = res.in_sample.equity
    oos_curve = res.out_of_sample.equity
    # One continuous line: IS curve, then the OOS run rescaled to start where IS ended.
    joined = pd.concat([is_curve, oos_curve / oos_curve.iloc[0] * is_curve.iloc[-1]])
    bench_in, bench_out = res.benchmark_in.equity, res.benchmark_out.equity
    bench = pd.concat([bench_in, bench_out / bench_out.iloc[0] * bench_in.iloc[-1]])
    plot_equity(
        {f"best: {_params(res.best.params)}": joined, "benchmark": bench},
        out / "research.png",
        f"{cfg.name}: {res.strategy}, {_best_of(res)}, then out-of-sample",
        marks={"out-of-sample starts": oos_curve.index[0]},
    )
    trials = sorted(res.trials, key=lambda t: -t.sharpe if math.isfinite(t.sharpe) else math.inf)
    lines = [
        f"# Research: {cfg.name}",
        "",
        f"Strategy `{res.strategy}`. In-sample {res.eval_start.date()} to {res.split_date.date()}; "
        f"out-of-sample {oos_curve.index[0].date()} to {oos_curve.index[-1].date()}.",
        "",
        *_ledger_note(res),
        "![research](research.png)",
        "",
        "## Verdict",
        "",
        *[f"- **{line}**" if line.startswith("WARN") else f"- {line}" for line in res.verdict],
        "",
        "## In-sample vs out-of-sample",
        "",
        metrics_table(
            {
                "Best, in-sample": res.in_sample.summary(),
                "Best, out-of-sample": res.out_of_sample.summary(),
                "Benchmark, out-of-sample": res.benchmark_out.summary(),
            }
        ),
        "",
        *_trials_heading(res),
        "| Params | Sharpe | Total return | Max drawdown | Kill switch |",
        "|---|---|---|---|---|",
        *[
            f"| {_params(t.params)} | {fmt(t.sharpe, 'num')} | {fmt(t.total_return, 'pct')} | "
            f"{fmt(t.max_drawdown, 'pct')} | {'tripped' if t.kill_switch else ''} |"
            for t in trials
        ],
        "",
        "## Data cleaning",
        "",
        "```",
        cleaning.summary(),
        "```",
        "",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path


def _best_of(res) -> str:
    """The best is picked from this run's grid; earlier runs' trials only raise the deflation bar."""
    if not res.n_trials_prior:
        return f"best of {len(res.trials)} in-sample"
    return f"best of this run's {len(res.trials)} in-sample (deflated for {res.n_trials_total} tried)"


def _ledger_note(res) -> list[str]:
    if res.ledger_run_id is None:
        return ["Trial ledger: off. Trials and out-of-sample looks from earlier runs are not counted.", ""]
    return [
        f"Trial ledger: {res.n_trials_total} distinct parameter set(s) tried on this strategy and data so far; "
        f"earlier runs had evaluated an overlapping out-of-sample period {res.oos_prior_evaluations} time(s). "
        f"This run is `{res.ledger_run_id}`.",
        "",
    ]


def _trials_heading(res) -> list[str]:
    if not res.n_trials_prior:
        return [f"## All {len(res.trials)} in-sample trials", ""]
    return [
        f"## This run's {len(res.trials)} in-sample trials",
        "",
        f"The best is chosen from these only. Earlier runs on the same strategy and data tried "
        f"{res.n_trials_prior} more parameter set(s), recorded in the trial ledger, and the deflated Sharpe ratio "
        f"counts all {res.n_trials_total}.",
        "",
    ]


def _params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "defaults"
