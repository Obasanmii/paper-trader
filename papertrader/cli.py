"""Command-line interface.

    python -m papertrader data        -c CONFIG     clean + summarise the data
    python -m papertrader backtest    -c CONFIG     strategy vs benchmark, with a report
    python -m papertrader research    -c CONFIG     IS sweep -> deflated Sharpe -> OOS verdict
    python -m papertrader paper-step  -c CONFIG     run today's paper-trading step
    python -m papertrader status      -c CONFIG     account, pending orders, kill switch
    python -m papertrader kill        -c CONFIG --reason "..."
    python -m papertrader reset-kill  -c CONFIG --yes-i-checked
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from papertrader.config import ConfigError, config_to_dict, load_config


def _cmd_data(cfg, args) -> int:
    from papertrader.data import load_market_data

    data, report = load_market_data(cfg.data)
    print(f"{len(data.symbols)} symbols, {len(data):,} days, {data.dates[0].date()} to {data.dates[-1].date()}")
    print(report.summary())
    if len(report):
        print()
        print(report.to_frame().head(25).to_string(index=False))
    return 0


def _cmd_backtest(cfg, args) -> int:
    from papertrader.data import load_market_data
    from papertrader.engine.backtest import evaluation_start, run_backtest
    from papertrader.journal import Journal
    from papertrader.reporting import backtest_report, metrics_table
    from papertrader.strategies import build_strategy
    from papertrader.validation import bootstrap_sharpe_ci, probabilistic_sharpe_ratio, timing_test

    out = Path(args.out or Path("reports") / cfg.name / "backtest")
    data, cleaning = load_market_data(cfg.data)
    start = evaluation_start(data, cfg.research.warmup_days)
    strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
    journal = Journal(out / "journal.sqlite")
    common = dict(portfolio=cfg.portfolio, costs=cfg.costs, limits=cfg.risk, start=start)
    try:  # close (which commits) even on failure: the rows up to the error are what a post-mortem needs
        result = run_backtest(data, strategy, journal=journal, config=config_to_dict(cfg), **common)
    finally:
        journal.close()
    bench = run_backtest(data, build_strategy(cfg.research.benchmark, {}), **common)
    r = result.returns()
    validation = {
        "psr": probabilistic_sharpe_ratio(r),
        "sharpe_ci": bootstrap_sharpe_ci(r, n_boot=cfg.research.bootstrap_samples),
        "timing": timing_test(result.weights, data, start=start, n_perm=cfg.research.timing_permutations),
    }
    path = backtest_report(cfg, result, bench, validation, cleaning, out)
    print(metrics_table({result.name: result.summary(), bench.name: bench.summary()}))
    lo, hi = validation["sharpe_ci"]
    t = validation["timing"]
    print(f"\nPSR {validation['psr']:.2f} | Sharpe 95% CI [{lo:.2f}, {hi:.2f}] | timing p = {t['p_value']:.3f}")
    print(f"Kill switch: {result.kill_switch_reason or 'never tripped'}")
    print(f"Report: {path}\nJournal: {out / 'journal.sqlite'}")
    return 0


def _cmd_research(cfg, args) -> int:
    from papertrader.data import load_market_data
    from papertrader.reporting import research_report
    from papertrader.research import run_research

    out = Path(args.out or Path("reports") / cfg.name / "research")
    data, cleaning = load_market_data(cfg.data)
    res = run_research(data, cfg, ledger=cfg.research.ledger_path)
    path = research_report(cfg, res, cleaning, out)
    earlier = f"; earlier runs tried {res.n_trials_prior} more" if res.n_trials_prior else ""
    print(f"{res.strategy}: {len(res.trials)} trials in-sample, best = {res.best.params}{earlier}")
    for line in res.verdict:
        print("  " + line)
    print(f"Report: {path}")
    if res.ledger_run_id is None:
        print("Trial ledger: off (research.ledger_path is null)")
    else:
        print(f"Trial ledger: {cfg.research.ledger_path} (this run: {res.ledger_run_id})")
    return 0


def _cmd_paper_step(cfg, args) -> int:
    from papertrader.paper import PaperRunner
    from papertrader.utils import is_valid_price

    runner = PaperRunner(cfg)
    results = runner.step(as_of=args.as_of, force=args.force, accept_config_change=args.accept_config_change)
    for event in runner.step_events:  # found before the first day: config changes, corporate actions
        print(f"RISK EVENT: {event.kind}: {event.detail}")
    if not results:
        print("Nothing to do: already processed the latest complete trading day.")
        return 0
    for day in results:
        flags = " [KILL SWITCH ON]" if day.kill_switch else ""
        # A wiped-out or unvaluable account is exactly when this line must still print.
        gross = f"{day.gross_exposure / day.equity:6.1%}" if is_valid_price(day.equity) else "   n/a"
        print(
            f"{day.date.date()}  equity {day.equity:>12,.2f}  gross {gross}  "
            f"fills {len(day.fills)}  approved {len(day.approved)}  rejected {len(day.rejected)}{flags}"
        )
        for event in day.events:
            print(f"    RISK EVENT: {event.kind}: {event.detail}")
        for order, reasons in day.rejected:
            print(f"    rejected {order.side} {abs(order.quantity):g} {order.symbol}: {'; '.join(reasons)}")
        for order, why in day.cancelled:
            print(f"    cancelled {order.side} {abs(order.quantity):g} {order.symbol}: {why}")
    return 0


def _cmd_status(cfg, args) -> int:
    from papertrader.paper import PaperRunner

    print(json.dumps(PaperRunner(cfg).status(), indent=2, default=str))
    return 0


def _cmd_kill(cfg, args) -> int:
    from papertrader.paper import PaperRunner

    switch = PaperRunner(cfg).kill_switch()
    if switch.trip(f"manual: {args.reason}"):
        print(f"Kill switch TRIPPED: {args.reason}")
    else:
        print(f"Kill switch was already tripped: {switch.reason()}")
    print("Queued orders that add risk will be cancelled at the next step; positions will be flattened if risk.flatten_on_kill.")
    return 0


def _cmd_reset_kill(cfg, args) -> int:
    from papertrader.paper import PaperRunner

    if not args.yes_i_checked:
        print("Refusing: pass --yes-i-checked once you've looked at why it tripped (see the journal's risk_events).")
        return 1
    before = PaperRunner(cfg).reset_kill_switch()  # under the run lock: never in the middle of a step
    if before["tripped"]:
        print(f"Kill switch reset. It had tripped at {before['at']}: {before['reason']}")
    else:
        print("Kill switch was not tripped; nothing to reset.")
    return 0


COMMANDS = {
    "data": (_cmd_data, "load, clean and summarise the configured market data"),
    "backtest": (_cmd_backtest, "backtest the configured strategy against the benchmark"),
    "research": (_cmd_research, "sweep parameters in-sample, then judge the winner out-of-sample"),
    "paper-step": (_cmd_paper_step, "run the paper-trading day(s) since the last run (safe to re-run)"),
    "status": (_cmd_status, "show the paper account, queued orders and kill switch"),
    "kill": (_cmd_kill, "trip the kill switch: no new risk until a human resets it"),
    "reset-kill": (_cmd_reset_kill, "reset the kill switch after checking why it tripped"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="papertrader", description="Careful paper-trading research stack.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("-c", "--config", required=True, help="path to a YAML config file")
        if name in ("backtest", "research"):
            p.add_argument("--out", help="output directory for the report")
        if name == "paper-step":
            p.add_argument("--as-of", help="simulate running on this date (YYYY-MM-DD)")
            p.add_argument("--force", action="store_true", help="allow catching up more than paper.max_catchup_days")
            p.add_argument(
                "--accept-config-change",
                action="store_true",
                help="adopt a changed data/strategy/portfolio/costs/risk config (journaled as a risk event)",
            )
        if name == "kill":
            p.add_argument("--reason", required=True)
        if name == "reset-kill":
            p.add_argument("--yes-i-checked", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        return COMMANDS[args.command][0](cfg, args)
    except BrokenPipeError:  # e.g. `papertrader data ... | head`: stop quietly
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:  # OSError: missing file, a directory, no permission
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:  # e.g. a held position with no valid mark: a data problem, not a bug
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)  # str(KeyError) adds quotes
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
