# papertrader

A small, careful paper-trading research stack, built as version 1 of a team project. It collects and cleans market data, tries quantitative and machine-learning ideas, and tests them without fooling itself. Every order passes an independent risk layer with hard limits and a kill switch. Paper trading keeps a full audit log. It is paper only: nothing here touches real money, and nothing here is financial advice.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest            # 61 tests, about 5 seconds
bash scripts/demo.sh        # the whole tour, about a minute
```

Or one step at a time:

```bash
python -m papertrader data       -c config/demo_dirty_data.yaml   # cleaning catches planted errors
python -m papertrader research   -c config/demo_random_walk.yaml  # pure noise: should say no
python -m papertrader research   -c config/demo_trending.yaml     # a real edge: should mostly say yes
python -m papertrader backtest   -c config/demo_trending.yaml     # full report with charts
python -m papertrader paper-step -c config/demo_paper.yaml --as-of 2023-06-01
python -m papertrader status     -c config/demo_paper.yaml
```

Reports are written to `reports/`. Paper-trading state and its SQLite journal go in `state/`. Example output is committed in `docs/examples/`, so you can see what the reports look like before running anything.

## How it's organised

| Goal | Where | What it does |
|---|---|---|
| Collect and clean market data | `papertrader/data/` | CSV, Yahoo (yfinance) and synthetic sources, plus a cleaning pass that records every fix |
| Explore quant and ML ideas | `papertrader/strategies/` | Benchmark, trend, mean reversion and a walk-forward ML classifier, all behind one contract |
| Test carefully | `papertrader/research.py`, `validation.py` | In-sample sweep, deflated Sharpe ratio, out-of-sample bootstrap, timing test |
| Independent risk layer | `papertrader/risk/` | Hard limits, fail-closed checks, persistent kill switch |
| Paper trading with logs | `papertrader/paper.py`, `journal.py`, `execution/` | Simulated broker, daily runner, SQLite audit log |

Every trading day runs through one function, `TradingSession.process_day`, in backtests and paper trading alike:

```
open    fill yesterday's approved orders at today's open (with slippage and commission)
close   mark positions to market; check daily-loss and drawdown limits (may trip the kill switch)
decide  strategy target weights -> proposed orders (or flatten everything if the kill switch is on)
gate    every proposal goes through the risk manager; approved orders queue for tomorrow's open
record  decisions, risk verdicts, orders, fills and equity go to the journal
```

Because there is one code path, paper trading can't quietly drift away from the backtest. A test steps through 30 trading days of paper trading one day at a time and checks the equity matches the backtest to the cent.

## The strategy contract

A strategy turns market data into target weights: the fraction of equity to hold in each symbol. There is one rule. The weights for day t may only use data dated on or before t. Orders are decided after the close and filled at the next open, so a strategy that follows the rule can't trade on a price it hasn't seen.

The rule is enforced by a test. `tests/test_no_lookahead.py` recomputes every registered strategy on data cut off at several dates and checks that the answers match the full run. New strategies are picked up automatically. A deliberately cheating strategy is included in the tests to prove the check catches one.

The ML classifier adds its own leakage controls. Its features use past data only, and its retrain dates are fixed from the start of the data. A training sample is only used once its label is fully known, with an extra embargo gap on top.

## Testing ideas without fooling ourselves

`python -m papertrader research` runs a fixed protocol:

1. It splits history once into an in-sample and an out-of-sample period.
2. It tries every parameter combination in the config on the in-sample period only, and picks the best.
3. It deflates the winner's Sharpe ratio for the number of combinations tried, using the deflated Sharpe ratio of Bailey and López de Prado.
4. It runs the winner, untouched, on the out-of-sample period and bootstraps a confidence interval for its Sharpe ratio.
5. It compares the winner with an equal-weight benchmark.
6. It runs a timing test: does the strategy beat randomly time-shifted copies of its own positions?

The report states which checks passed and which failed.

To check the protocol itself, `scripts/calibrate_protocol.py` runs it on ten synthetic markets with no edge at all and ten with a real but modest one. In the edge markets, prices follow hidden trends, and the winning trend strategy's median out-of-sample Sharpe came out at about 0.7. That is roughly the range long-run trend-following has historically reported.

| | Pure noise | Real edge |
|---|---|---|
| Best in-sample backtest looks "significant" (no correction) | 30% | 70% |
| Still significant after deflating for 16 trials | 20% | 30% |
| Out-of-sample Sharpe interval above zero | 0% | 50% |
| Beats the benchmark out-of-sample | 50% | 100% |
| Timing beats shifted copies | 0% | 30% |
| Median best in-sample Sharpe | 0.13 | 0.60 |
| Median out-of-sample Sharpe | -0.15 | 0.69 |

Two lessons come out of this.

First, on pure noise the best of 16 backtests looked statistically significant almost a third of the time. In-sample Sharpe ratios reached 0.9, then fell as low as -0.96 out of sample. That is what most impressive-looking backtests are.

Second, even a genuine edge of this size is hard to confirm. With 12 years in-sample and 8 years out-of-sample, the protocol confirmed it only about half the time.

The deflated Sharpe ratio only tests whether a result beats zero, so a strategy that happened to be long during a lucky in-sample market can still pass it. The benchmark comparison and the timing test are there to catch that case.

## The risk layer

The risk manager is the only way to reach the broker, and it deliberately knows nothing about strategies. A test fails if `papertrader/risk/` ever imports strategy code. The risk manager sees proposed orders, positions, cash and prices. For each order it can approve it, reject it with reasons, or trip the kill switch.

The hard limits are set in the `risk:` section of the config:

- Position size as a share of equity.
- Gross exposure.
- Order size (a fat-finger limit).
- Orders per day.
- Short selling (off by default).
- A symbol whitelist.
- How far the price an order was decided on may be from the current market.
- Stale data, in live runs.

Orders in a batch are checked against projected positions, so several individually acceptable orders can't jointly breach a limit. Every check fails closed: a missing price, a NaN quantity or a position that can't be valued means the order is rejected, never that the check is skipped. Orders that only shrink an existing position always get through, because you must always be able to get out.

The kill switch trips in three ways:

- Automatically, when a daily loss or drawdown goes beyond its limit.
- From the command line: `papertrader kill -c CONFIG --reason "..."`.
- When a file named `KILL` appears in the state directory. This works even if the process is stuck.

Once tripped, it cancels queued orders that add risk and flattens all positions at the next open (with `flatten_on_kill: true`). It then blocks everything except exits until someone runs `papertrader reset-kill -c CONFIG --yes-i-checked`. Its state persists across runs, and it can't be reset while the `KILL` file exists.

Its first real catch was a bug in this repo. An earlier version of the portfolio code topped up positions that had drifted below target without trimming the ones above target, which would have pushed gross exposure past 100%. The risk layer rejected every one of those orders, and the fix went into the portfolio code. In the example reports, it also halted the equal-weight benchmark twice: once at the 35% drawdown limit, and once on a 7% one-day loss.

Config parsing is strict for the same reason. An unknown key is an error, so a typo like `max_postion_pct` can't silently switch a limit off.

## Paper trading

Run one step after each market close:

```bash
python -m papertrader paper-step -c config/example_yahoo.yaml
```

The runner is built to be safe to run repeatedly:

- Running it twice on the same day does nothing the second time.
- It refuses to run while another run holds the lock.
- If days were missed, it processes each one in order, up to `paper.max_catchup_days`. More than that needs `--force`, so you notice the gap.

State lives in `paper.state_dir`:

- `state.json` holds cash, positions, queued orders and risk state.
- `kill_switch.json` holds the kill switch.
- `journal.sqlite` holds the full log. It records every target weight with the signal behind it, every proposed order with the risk layer's verdict and reasons, every fill and cancellation, every risk event, and the equity curve.

Nothing in the journal is updated in place, so "why did it do that?" always has an answer:

```bash
sqlite3 state/demo_paper/journal.sqlite \
  "select date, event, symbol, quantity, reason, detail from orders order by rowid"
```

To automate it on Linux or macOS, add a cron entry that runs `paper-step` on weekdays after the close of the market you trade, in your machine's local time, from the repo directory with the virtual environment's Python.

The broker in version 1 is simulated. Orders fill at the next open with slippage and commission. Sells are processed before buys, and buys are cut to the cash available rather than silently using margin. A real paper-trading API, such as Alpaca or Interactive Brokers, would slot in behind the same `Broker` interface in `papertrader/execution/broker.py`, with nothing upstream changing.

## Real data

```bash
pip install -e ".[yahoo]"
python scripts/download_yahoo.py -c config/example_yahoo.yaml   # saves data/raw/<SYMBOL>.csv
# then set data.source: csv in the config, so every run reads the same frozen data
python -m papertrader data -c config/example_yahoo.yaml         # read the cleaning report first
```

Yahoo data through `yfinance` is unofficial. It is fine for learning, not a feed to rely on. The Yahoo adapter could not be exercised where this was built (there was no access to Yahoo), so treat your first real download as a test and read the cleaning report.

Any CSV with a date column and open, high, low, close and volume columns works. If there is an `adj_close` column, it is used to adjust prices for splits and dividends.

## Adding a strategy

1. Subclass `Strategy` in a new module under `papertrader/strategies/` and decorate it with `@register`. `trend.py` is the simplest example to copy.
2. Import the module in `papertrader/strategies/__init__.py`.
3. Run `python -m pytest`. The lookahead tests pick up the new strategy automatically.
4. Write a config with a small parameter grid, decided before looking at any results.
5. Run `research` on the random-walk config first. It should fail there. Then run it on the data you care about.

## Working on this together

The code splits along the lines the project is recruiting for:

- Data engineering: `data/`.
- Statistics and research: `validation.py` and `research.py`.
- Machine learning: `strategies/ml.py`.
- Broker APIs: `execution/`.
- Risk management: `risk/`.

Good first tasks, each small enough for one person:

- **A buffer against whipsaw for trend following.** This is the most pressing one. The trend strategy flips in and out whenever the price crosses its moving average. In the example backtest it traded about 10 times its equity a year, and costs ate 18% of starting capital. The fix is to enter only above the average plus a margin and exit only below it minus the margin, then let the research protocol judge whether it helps.
- A `Broker` adapter for Alpaca's or Interactive Brokers' paper API.
- A price collar that cancels orders when the next open gaps too far from the decision price.
- Purged k-fold cross-validation for the ML strategy's settings.
- Portfolio-level volatility targeting.
- A dashboard over `journal.sqlite`.
- Corporate-action checks beyond adjusted closes.

The rules for contributions are in `CLAUDE.md`. In short:

- Every order goes through the risk layer.
- Strategies never see the future.
- Limits never get loosened to make a backtest look better.
- Results get reported honestly, including "passed 1 of 4".

## Claude Code agents

`.claude/agents/` holds four project subagents for Claude Code, and `CLAUDE.md` holds the project rules Claude Code reads at the start of each session.

- `risk-reviewer` checks changes to the order path for anything that could bypass or weaken the risk layer.
- `lookahead-auditor` hunts for future data leaking into strategies, features and labels.
- `strategy-researcher` implements a new idea under the contract and runs the research protocol.
- `data-engineer` adds data sources and cleaning checks, with tests.

In a Claude Code session in this repo, ask for one by name, for example "have the risk-reviewer look at my changes".

## Limitations

- Only daily bars, and only market orders filled at the next open. There are no intraday data, no limit orders and no model of limited liquidity.
- No borrow costs for short positions (shorting is off by default).
- One currency.
- Synthetic results show that the machinery works. They say nothing about real markets.
- The deflated Sharpe ratio and the timing test are guides, not proofs. The timing test is conservative for slow signals.
- Paper results overstate live results. Real fills, fees and a person's own behaviour under pressure are all worse than a simulator.
