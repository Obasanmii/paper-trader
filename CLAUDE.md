# Working rules for this repo

This is a paper-trading research stack. No real money, ever, in v1.

Run `python -m pytest` before and after every change. All tests must pass.

Hard rules:
- Every order goes through `RiskManager.check_orders`. Never call `broker.submit` from anywhere else.
- `papertrader/risk/` must not import from `strategies`, `engine` or `research` (a test enforces this).
- Risk checks fail closed: missing or invalid data means reject, never skip the check.
- Strategies obey the no-lookahead contract in `papertrader/strategies/base.py`. New strategies are
  tested automatically by `tests/test_no_lookahead.py`; never weaken that test to make one pass.
- Never loosen a limit in `config/*.yaml` to make a backtest look better.
- Research protocol: sweep parameters in-sample only, report the number of trials, judge once
  out-of-sample. Do not re-sweep after looking at out-of-sample results.
- Report results honestly, including failures. "Passed 1 of 4 checks" is a useful result.

Layout: data -> strategies -> engine (portfolio, session) -> risk -> execution -> journal.
Backtests and paper trading share `TradingSession.process_day`; keep it that way.
