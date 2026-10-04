# Working rules for this repo

This is a paper-trading research stack. No real money, ever, in v1.

Run `python -m pytest` and `ruff check .` before and after every change. All tests must pass. CI runs both
on every push and pull request.

Hard rules:
- Every order goes through `RiskManager.check_orders`. Never call `broker.submit` from anywhere else. Queued orders
  also pass `RiskManager.check_open` (the price collar) at the next open; never fill them without it.
- `papertrader/risk/` must not import from `strategies`, `engine` or `research` (a test enforces this).
- Risk checks fail closed: missing or invalid data means reject, never skip the check.
- Strategies obey the no-lookahead contract in `papertrader/strategies/base.py`. New strategies are
  tested automatically by `tests/test_no_lookahead.py`; never weaken that test to make one pass.
- Never loosen a limit in `config/*.yaml` to make a backtest look better.
- Config fields are validated from their type annotations (`papertrader/config.py`). Give every new field an
  accurate annotation and a range check in its dataclass's `__post_init__`.
- Research protocol: sweep parameters in-sample only, report the number of trials, judge once
  out-of-sample. Do not re-sweep after looking at out-of-sample results. Every run is recorded in the trial
  ledger (`research.ledger_path`); never delete or bypass it to reset the trial count. A `WARN ... already
  evaluated` line means the holdout is spent: get new data; don't re-split or re-sweep.
- Report results honestly, including failures. "Passed 1 of 4 checks" is a useful result.

Layout: data -> strategies -> engine (portfolio, session) -> risk -> execution -> journal.
Backtests and paper trading share `TradingSession.process_day`; keep it that way.

Paper trading:
- Paper data is prepared per day with `prepare_market_data(raw, cfg.data, as_of=day, hold_unconfirmed=True)` from
  one `load_raw` download, so catch-up equals on-time. Research and backtests must not use `hold_unconfirmed`.
- A paper step is one transaction: write account state only through `Journal.save_paper_state` inside it, and
  never commit the journal mid-step.
- Automatic kill-switch trips in a paper step are deferred (`KillSwitch(state_dir, deferred=True)`) and written
  to `kill_switch.json` only after the commit (`flush()`); never write the file mid-step. `reset-kill` goes
  through `PaperRunner.reset_kill_switch()`, under the run lock.
- `.lock` is an OS lock (`utils.run_lock`): never delete it, and don't add pid-based stale-lock logic.
