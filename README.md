# papertrader

A small, careful paper-trading research stack, built as version 1 of a team project. It collects and cleans market data, tries quantitative and machine-learning ideas, and tests them without fooling itself. Every order passes an independent risk layer with hard limits and a kill switch. Paper trading keeps a full audit log. It is paper only: nothing here touches real money, and nothing here is financial advice.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"     # or: pip install -r requirements.txt (same thing)
python -m pytest            # the full suite, under a minute
ruff check .                # lint; CI runs both on every push and pull request
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
| Test carefully | `papertrader/research.py`, `validation.py`, `ledger.py` | In-sample sweep, deflated Sharpe ratio, out-of-sample bootstrap, paired benchmark test, timing test, persistent trial ledger |
| Independent risk layer | `papertrader/risk/` | Hard limits, fail-closed checks, persistent kill switch |
| Paper trading with logs | `papertrader/paper.py`, `journal.py`, `execution/` | Simulated broker, daily runner, SQLite audit log |

Every trading day runs through one function, `TradingSession.process_day`, in backtests and paper trading alike:

```
open    cancel queued orders that add risk if today's open gapped too far from the price they were
        decided on (the price collar); fill the rest at today's open (with slippage and commission)
close   mark positions to market; check daily-loss, drawdown and daily-gain limits (may trip the kill switch)
decide  strategy target weights -> proposed orders (or flatten everything if the kill switch is on)
gate    every proposal goes through the risk manager; approved orders queue for tomorrow's open
record  decisions, risk verdicts, orders, fills and equity go to the journal
```

Because there is one code path, paper trading can't quietly drift away from the backtest. A test steps through 30 trading days of paper trading one day at a time and checks the equity matches the backtest to the cent. Another checks that catching up on missed days ends exactly where running on time would have, even on data with planted errors.

## The strategy contract

A strategy turns market data into target weights: the fraction of equity to hold in each symbol. There is one rule. The weights for day t may only use data dated on or before t. Orders are decided after the close and filled at the next open, so a strategy that follows the rule can't trade on a price it hasn't seen.

The rule is enforced by a test. `tests/test_no_lookahead.py` recomputes every registered strategy on data cut off at several dates and checks that the answers match the full run. It does this for each strategy's defaults, for every parameter combination in every shipped config's research grid, and for explicit cases that reach code paths the grids don't (trend with `vol_target` and `band`, the ML classifier with `model: gbm`). New strategies and new grids are picked up automatically. A case that is flat on every check date fails, because it would pass without proving anything. A deliberately cheating strategy is included in the tests to prove the check catches one.

The ML classifier adds its own leakage controls. Its features use past data only, and its retrain dates are fixed from the start of the data. A training sample is only used once its label is fully known, with an extra embargo gap on top.

## Testing ideas without fooling ourselves

`python -m papertrader research` runs a fixed protocol:

1. It splits history once into an in-sample and an out-of-sample period.
2. It tries every parameter combination in the config on the in-sample period only, and picks the best.
3. It deflates the winner's Sharpe ratio for the number of combinations tried, using the deflated Sharpe ratio of Bailey and López de Prado.
4. It runs the winner, untouched, on the out-of-sample period and bootstraps a confidence interval for its Sharpe ratio.
5. It compares the winner with an equal-weight benchmark, using a paired stationary bootstrap of the out-of-sample Sharpe difference (both strategies resampled on the same days). It passes only if the one-sided 95% lower bound of the difference is above zero.
6. It runs a timing test: does the strategy beat randomly time-shifted copies of its own positions?

The report states which checks passed and which failed.

Every research run is also recorded in a trial ledger, a SQLite file at `research.ledger_path` (default `state/research_ledger.sqlite`). Runs are grouped by matching data, not by an exact hash or by the parameter grid. Two runs match when they test the same strategy on the same symbols (in any order) and each symbol's in-sample daily returns line up: they correlate at least 0.99 day by day and month by month, over days covering at least 90% of each run's in-sample window. A re-download that back-adjusts a dividend or corrects a bad tick, the same bars read from a CSV copy, reordered `data.symbols`, a different `warmup_days` or a split moved by a few weeks all count as the same data. Different market histories don't. So if you run `research` again with a different grid, the deflated Sharpe ratio counts every distinct parameter set ever tried for the strategy on matching data, and the verdict says how many came from earlier runs. The best is still picked from this run's grid only, and the verdict says that too.

If an earlier run on the same market already evaluated an overlapping out-of-sample period, the verdict opens with a `WARN` line: the holdout has been looked at, so it is no longer a clean test. This also catches a large re-split, a moved start date, or both moved together as a sliding window: for the warning it is enough that the two in-sample windows share most of the shorter one, or a year of the same market. Don't delete the ledger or set `ledger_path: null` to get a fresh count; that is the re-sweeping the protocol exists to prevent. A ledger from an older version of this code is refused with a message; rename the old file to keep it.

Two details keep the statistics honest. Parameter sets that never trade still count toward N; if they pad a grid around one live result, the deflated Sharpe check fails rather than quietly becoming an undeflated one. And in both bootstraps a resample where the strategy sat in cash scores a Sharpe of 0 and counts; earlier code dropped those, which flattered strategies that rarely trade. (The demo configs use their own ledger, which `scripts/demo.sh` resets, so the tour shows the first-look verdicts every time.)

To check the protocol itself, `scripts/calibrate_protocol.py` runs it on ten synthetic markets with no edge at all and ten with a real but modest one. In the edge markets, prices follow hidden trends, and the winning trend strategy's median out-of-sample Sharpe came out at about 0.7. That is roughly the range long-run trend-following has historically reported.

| | Pure noise | Real edge |
|---|---|---|
| Best in-sample backtest looks "significant" (no correction) | 30% | 70% |
| Still significant after deflating for 16 trials | 20% | 30% |
| Out-of-sample Sharpe interval above zero | 0% | 50% |
| Beats the benchmark out-of-sample (paired bootstrap) | 0% | 80% |
| Timing beats shifted copies | 0% | 30% |
| Median best in-sample Sharpe | 0.13 | 0.60 |
| Median out-of-sample Sharpe | -0.15 | 0.69 |
| Passes every check | 0% | 0% |

Two lessons come out of this.

First, on pure noise the best of 16 backtests looked statistically significant almost a third of the time. In-sample Sharpe ratios reached 0.9, then fell as low as -0.96 out of sample. That is what most impressive-looking backtests are.

Second, even a genuine edge of this size is hard to confirm. With 12 years in-sample and 8 years out-of-sample, the protocol confirmed it only about half the time.

The deflated Sharpe ratio only tests whether a result beats zero, so a strategy that happened to be long during a lucky in-sample market can still pass it. The benchmark comparison and the timing test are there to catch that case. An earlier version compared the two Sharpe ratios as plain numbers, and on pure noise the strategy "beat" the benchmark half the time. The paired test passes on none of the noise markets and on 8 of the 10 edge markets.

## The risk layer

The risk manager is the only way to reach the broker, and it deliberately knows nothing about strategies. A test fails if `papertrader/risk/` ever imports strategy code. The risk manager sees proposed orders, positions, cash and prices. For each order it can approve it, reject it with reasons, cancel it at the next open if the price has gapped too far, or trip the kill switch.

The hard limits are set in the `risk:` section of the config:

- Position size as a share of equity.
- Gross exposure.
- Order size (a fat-finger limit).
- Orders per day. Every order that adds risk counts, rejected ones too, so a runaway loop proposing junk is still stopped. Exits never count.
- Short selling (off by default).
- A symbol whitelist.
- A price collar (`max_price_deviation_pct`, default 10%). Orders are decided on the close and filled at the next open; if that open has gapped further than this from the decision price, a queued order that adds risk is cancelled with the reason in the journal. A missing or invalid price cancels it too. Exits always go through.
- Stale data, in live runs.

Orders in a batch are checked against projected positions, so several individually acceptable orders can't jointly breach a limit. Every check fails closed: a missing price, a NaN quantity or a position that can't be valued means the order is rejected, never that the check is skipped. Orders that only shrink an existing position always get through, because you must always be able to get out.

The kill switch trips in three ways:

- Automatically, when a daily loss or drawdown goes beyond its limit, or when equity jumps by `max_daily_gain_pct` or more in one day (default 25%; `null` turns it off). A gain that large on a diversified paper account almost always means a bad price, so it does not become the new peak or the base for the next day's loss check. If a person checks it and resets the switch, the next close accepts the new level as the base, journaled as a `gain_accepted` risk event. That close is still measured against the one the person confirmed: another implausible jump is held back again, and a loss beyond the daily limit trips.
- In paper trading, when a held symbol's past prices were revised in a way that isn't a clean split or dividend adjustment (see below).
- From the command line: `papertrader kill -c CONFIG --reason "..."`.
- When a file named `KILL` appears in the state directory. This works even if the process is stuck.

Once tripped, it cancels queued orders that add risk and flattens all positions at the next open (with `flatten_on_kill: true`). It then blocks everything except exits until someone runs `papertrader reset-kill -c CONFIG --yes-i-checked`. Its state persists across runs, and it can't be reset while the `KILL` file exists. `reset-kill` waits for the run lock, so it can't land in the middle of a step, and a trip that has been reset is never re-applied, even if `kill_switch.json` is later lost. A trip from outside a step (the `kill` command or the `KILL` file) is journaled once, as `kill_switch_observed`, by the first step that acts on it.

Its first real catch was a bug in this repo. An earlier version of the portfolio code topped up positions that had drifted below target without trimming the ones above target, which would have pushed gross exposure past 100%. The risk layer rejected every one of those orders, and the fix went into the portfolio code. In the example reports, it also halted the equal-weight benchmark twice: once at the 35% drawdown limit, and once on a 7% one-day loss.

Config parsing is strict for the same reason. An unknown key is an error, so a typo like `max_postion_pct` can't silently switch a limit off. Values are checked against each setting's type and range too. Booleans must be YAML `true`/`false`: a quoted `"false"` used to be read as true, which switched short selling on. Whole-number settings reject fractions, numbers must be finite, ticker lists must hold strings (quote tickers YAML would misread, such as `'ON'`), and dates must be `YYYY-MM-DD`. Errors name the full setting, for example `config.risk.allow_short: expected true or false, got 'false'`. A key or section repeated in the YAML is an error too: with plain YAML the last copy silently wins, so a second `risk:` block would reset the first one's limits to the defaults. `research.param_grid` must map parameter names to non-empty lists, and `data.cleaning.max_ffill_days: 0` turns forward-filling off.

## Paper trading

Run one step after each market close:

```bash
python -m papertrader paper-step -c config/example_yahoo.yaml
```

The runner is built to be safe to run repeatedly:

- Running it twice on the same day does nothing the second time.
- It refuses to run while another run holds the lock. The lock is an operating-system lock on `.lock` (flock, or msvcrt on Windows), released however the run ends, so a crash never leaves a stale lock behind. A step also takes SQLite's write lock before reading the account, so two overlapping runs can never both process a day.
- If days were missed, it processes each one in order, up to `paper.max_catchup_days`. More than that needs `--force`, so you notice the gap. Each missed day is processed on the data as it stood that day, so a catch-up ends exactly where on-time runs would have.
- A crash at any point leaves the account, the journal and the kill switch as they were. A step's journal rows and the account state are committed together in one transaction, so the retry does the work once. A kill-switch trip the step makes is part of that transaction: it is recorded with the account and written to `kill_switch.json` only after the commit, so a failed step leaves no trip behind and the retry trips it again from the same data. (If the run dies between the commit and the file write, the next step writes the trip.)
- A run before the market has closed ignores today's bar, which data feeds serve while it is still forming. The close is `paper.market_close` (default `"16:00"`, quoted in YAML) plus `paper.close_buffer_minutes` (default 15) in `paper.market_timezone` (default `America/New_York`). With `--as-of DATE` that date counts as closed.
- A newest bar that moved more than `data.cleaning.spike_threshold` is held back for a day, because a bad tick can only be told from a real move once the next bar arrives. Until then the symbol is marked at its last good close and isn't traded. Next day the bar is either dropped as a bad tick or kept, so a real big move shows up one day late. Research and backtests don't do this.
- The account is tied to its config. The `data`, `strategy`, `portfolio`, `costs` and `risk` sections are stored with it. A run with any change is refused, listing each `setting: old -> new`, until you re-run with `--accept-config-change`, which journals the change as a `config_changed` risk event. A limit can't be loosened without a trace.

Paper trading expects adjusted data: the yfinance source, or CSVs with an `adj_close` column. An adjusted feed rescales all earlier prices at every split and dividend, while the account holds share counts on the old scale. So each run compares the closes it stored for its last few days with the new download. If a symbol's closes all moved by one factor, the account is re-anchored: share counts and queued quantities are divided by it, marks and decision prices multiplied, and a `corporate_action` risk event is journaled. Equity doesn't change, a 2-for-1 split no longer reads as a 50% loss, and dividends end up reinvested, which is the total-return accounting adjusted data implies. With `portfolio.allow_fractional: false` the new share count is rounded down and the fraction is paid out in cash at the new price (cash in lieu, shown in the event), so a whole-share catch-up across an ex-date can differ from on-time runs by that rounding. Only bars that actually traded are compared, each symbol's own last few, so a trading halt, a late or missing bar, or a close carried forward past a held-back bar is not mistaken for a revised history. A revision that isn't one clean factor trips the kill switch if the symbol is held, and is logged as `history_revised` if not. With an unadjusted feed a split still looks like a crash.

State lives in `paper.state_dir`:

- `journal.sqlite` holds the full log and, in its `paper_state` table, the account itself: cash, positions, queued orders, risk state, marks, recent closes, config, the account's last kill-switch trip and the last day processed. The account is stored under the key `account` (or under `paper.state_dir` as written, when `paper.journal_path` points somewhere else), so the state directory can be moved or re-mounted. A journal holding only other accounts, or paper runs but no account, is refused rather than starting a new account in it. The log records every target weight with the signal behind it, every proposed order with the risk layer's verdict and reasons, every fill and cancellation (including the unfilled part of a buy cut to the cash available), every risk event, and the equity curve.
- `state.json` is a readable copy of the account, written after each commit (and repaired by the next step if a crash left it stale). It is only read back to migrate an account from an older version, so editing it has no effect.
- `kill_switch.json` holds the kill switch, and a `generation` count of how many times a trip has been reset.
- `.lock` is locked while a run is in progress. The file stays; it only records who holds the lock.

`papertrader status -c CONFIG` shows the account. A position with no valid price shows no value, and equity is shown as unknown rather than counting it as zero.

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

Any CSV with a date column and open, high, low, close and volume columns works. If there is an `adj_close` column, it is used to adjust prices for splits and dividends. For paper trading, use adjusted data (see above).

`data.end` is inclusive for every source. (yfinance treats its own end date as exclusive; the adapter asks for one extra day and trims.)

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

- **Judge the trend strategy's whipsaw buffer.** This is the most pressing one. Without a buffer the trend strategy flips in and out whenever the price crosses its moving average: in the example backtest it traded about 10 times its equity a year, and costs ate 18% of starting capital. The strategy now has a `band` parameter: it enters only when the close is more than `band` above the average and exits only when it is more than `band` below it (`band: 0`, the default, is the old rule exactly). On the trending demo's in-sample years, a band of 0.01 to 0.05 cut turnover from about 9x to between 4x and 2x a year. Nobody has looked at returns yet. Decide a small grid of band values before looking at any results, then let the research protocol judge it.
- A `Broker` adapter for Alpaca's or Interactive Brokers' paper API.
- Purged k-fold cross-validation for the ML strategy's settings.
- Portfolio-level volatility targeting.
- A dashboard over `journal.sqlite`.
- Explicit split and dividend data, so paper trading also works on unadjusted feeds.

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
- In paper trading a genuine move bigger than the spike threshold is recognised one day late, and the symbol can't be traded on the day it is held back.
- Catching up N missed days reruns the strategy N times. That is quick for the rule-based strategies and slow for the ML classifier.
- Paper results overstate live results. Real fills, fees and a person's own behaviour under pressure are all worse than a simulator.
