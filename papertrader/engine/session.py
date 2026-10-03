"""One trading day, done the same way for backtests and paper trading.

`process_day(t)` does, in this order:
  0. if the kill switch tripped since yesterday, cancel queued orders that add risk
  1. OPEN   fill the orders queued yesterday at today's open        (broker)
  2. CLOSE  mark to market, then check loss limits                  (risk, may trip the kill switch)
  3. DECIDE read today's target weights, propose orders              (strategy -> portfolio)
            or, if the kill switch is tripped, propose flattening
  4. GATE   every proposal goes through the risk manager; approved
            orders queue for tomorrow's open                         (risk -> broker)
  5. RECORD decisions, verdicts, orders, fills, equity               (journal)

A backtest calls this for every day in a range. The paper runner calls it
for each new day since its last run. Same function, same results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from papertrader.core import OrderStatus
from papertrader.engine.portfolio import orders_from_targets
from papertrader.journal import Journal
from papertrader.risk.manager import is_reducing


@dataclass
class DayResult:
    date: pd.Timestamp
    cash: float
    equity: float
    gross_exposure: float
    net_exposure: float
    fills: list = field(default_factory=list)
    approved: list = field(default_factory=list)
    rejected: list = field(default_factory=list)  # [(order, [reasons])]
    cancelled: list = field(default_factory=list)  # [(order, why)]
    events: list = field(default_factory=list)
    kill_switch: bool = False


class TradingSession:
    def __init__(self, data, output, broker, risk, portfolio_cfg, journal: Journal | None = None, run_id: str | None = None):
        self.data = data
        self.broker, self.risk, self.cfg = broker, risk, portfolio_cfg
        self.journal = journal if journal is not None else Journal(None)
        self.run_id = run_id
        self.symbols = data.symbols
        self._index = {d: i for i, d in enumerate(data.dates)}
        self._open = data.open.to_numpy(dtype=float)
        self._close = data.close.to_numpy(dtype=float)
        self._weights = output.weights.reindex(index=data.dates, columns=self.symbols).fillna(0.0).to_numpy(dtype=float)
        self._signals = (
            None
            if output.signals is None
            else output.signals.reindex(index=data.dates, columns=self.symbols).to_numpy(dtype=float)
        )
        self.marks: dict[str, float] = {}  # last valid close per symbol
        self.peak_equity: float | None = None

    def _row(self, arr, i) -> dict[str, float]:
        return {s: float(v) for s, v in zip(self.symbols, arr[i]) if math.isfinite(v) and v > 0}

    def process_day(self, date, now=None) -> DayResult:
        date = pd.Timestamp(date)
        if date not in self._index:
            raise KeyError(f"{date.date()} is not a trading day in the data")
        i = self._index[date]
        j, run = self.journal, self.run_id
        self.risk.start_day(date)

        # 0. kill switch tripped outside the loop (CLI, sentinel file)? Don't let queued risk through.
        cancelled = []
        if self.risk.kill_switch.tripped:
            held = self.broker.snapshot({}).positions
            cancelled += self.broker.cancel_pending(
                lambda o: not is_reducing(held.get(o.symbol, 0.0), o.quantity), "kill switch active at the open"
            )

        # 1. open
        fills, open_cancels = self.broker.process_open(date, self._row(self._open, i))
        cancelled += open_cancels
        for f in fills:
            j.log_fill(run, f)
        for order, why in cancelled:
            j.log_order(run, order, date, "cancelled", [why])

        # 2. close
        closes = self._row(self._close, i)
        self.marks.update(closes)
        snap = self.broker.snapshot(self.marks)
        equity = snap.equity
        events = self.risk.end_of_day(date, equity)
        for e in events:
            j.log_risk_event(run, e)

        # 3. decide
        if self.risk.kill_switch.tripped:
            orders = self.risk.flatten_orders(snap, date) if self.risk.limits.flatten_on_kill else []
        else:
            targets = {s: float(self._weights[i, k]) for k, s in enumerate(self.symbols)}
            signals = None if self._signals is None else {s: self._signals[i, k] for k, s in enumerate(self.symbols)}
            j.log_decisions(run, date, targets, signals)
            orders = orders_from_targets(targets, snap, closes, date, self.cfg)

        # 4. gate
        approved, rejected = [], []
        for order, decision in self.risk.check_orders(orders, snap, now=now, data_as_of=date):
            if decision.approved:
                self.broker.submit(order)
                approved.append(order)
                j.log_order(run, order, date, "approved")
            else:
                order.status = OrderStatus.REJECTED
                rejected.append((order, decision.reasons))
                j.log_order(run, order, date, "rejected", decision.reasons)

        # 5. record
        self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1.0
        j.log_equity(run, date, snap.cash, equity, snap.gross_exposure, snap.net_exposure, drawdown)
        return DayResult(
            date=date,
            cash=snap.cash,
            equity=equity,
            gross_exposure=snap.gross_exposure,
            net_exposure=snap.net_exposure,
            fills=fills,
            approved=approved,
            rejected=rejected,
            cancelled=cancelled,
            events=events,
            kill_switch=self.risk.kill_switch.tripped,
        )
