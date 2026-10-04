"""Backtests: the TradingSession run over a date range with a fresh account."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from papertrader.engine.session import TradingSession
from papertrader.execution.paper_broker import SimulatedBroker
from papertrader.journal import Journal
from papertrader.metrics import PERIODS_PER_YEAR, performance_summary, returns_from_equity
from papertrader.risk.killswitch import KillSwitch
from papertrader.risk.manager import RiskManager


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    cash: pd.Series
    gross_exposure: pd.Series
    weights: pd.DataFrame
    fills: pd.DataFrame
    rejections: pd.DataFrame
    risk_events: list = field(default_factory=list)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    kill_switch_reason: str | None = None
    run_id: str | None = None

    def returns(self) -> pd.Series:
        return returns_from_equity(self.equity)

    def summary(self) -> dict:
        s = performance_summary(self.equity)
        years = max(len(self.equity) - 1, 1) / PERIODS_PER_YEAR
        traded = float(self.fills["notional"].sum()) if len(self.fills) else 0.0
        s["turnover_per_year"] = traded / float(self.equity.mean()) / years
        s["fills"] = int(len(self.fills))
        s["costs"] = self.total_commission + self.total_slippage
        s["costs_pct_of_start"] = s["costs"] / float(self.equity.iloc[0])
        s["avg_gross_exposure"] = float((self.gross_exposure / self.equity).mean())
        s["rejected_orders"] = int(len(self.rejections))
        s["kill_switch"] = self.kill_switch_reason
        return s


def run_backtest(
    data,
    strategy,
    portfolio,
    costs,
    limits,
    *,
    start=None,
    end=None,
    journal: Journal | None = None,
    name: str | None = None,
    output=None,
    config: dict | None = None,
) -> BacktestResult:
    """Trade `strategy` on `data` from `start` to `end` with a fresh account.

    The strategy sees all of `data` (it needs history before `start` for its
    lookbacks); the contract in strategies/base.py guarantees it can't use
    anything from after each decision date.
    """
    output = output if output is not None else strategy.run(data)
    name = name or strategy.describe()
    broker = SimulatedBroker(portfolio.initial_cash, costs, allow_fractional=portfolio.allow_fractional)
    risk = RiskManager(limits, KillSwitch())  # in-memory kill switch: never touches live state
    run_id = journal.start_run("backtest", name, strategy.describe(), config) if journal is not None else None
    session = TradingSession(data, output, broker, risk, portfolio, journal, run_id)

    dates = data.dates
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) == 0:
        raise ValueError("no trading days in the requested window")

    rows, fills, rejections = [], [], []
    for d in dates:
        day = session.process_day(d)
        rows.append((d, day.equity, day.cash, day.gross_exposure))
        fills.extend(
            (f.timestamp, f.symbol, f.quantity, f.price, f.commission, f.notional) for f in day.fills
        )
        rejections.extend((d, o.symbol, o.quantity, "; ".join(reasons)) for o, reasons in day.rejected)
    if journal is not None:
        journal.commit()

    curve = pd.DataFrame(rows, columns=["date", "equity", "cash", "gross"]).set_index("date")
    return BacktestResult(
        name=name,
        equity=curve["equity"],
        cash=curve["cash"],
        gross_exposure=curve["gross"],
        weights=output.weights,
        fills=pd.DataFrame(fills, columns=["date", "symbol", "quantity", "price", "commission", "notional"]),
        rejections=pd.DataFrame(rejections, columns=["date", "symbol", "quantity", "reasons"]),
        risk_events=list(risk.events),
        total_commission=broker.total_commission,
        total_slippage=broker.total_slippage,
        kill_switch_reason=risk.kill_switch.reason() if risk.kill_switch.tripped else None,
        run_id=run_id,
    )


def evaluation_start(data, warmup_days: int):
    """The common start date every strategy is judged from."""
    if len(data) <= warmup_days:
        raise ValueError(f"need more than {warmup_days} days of data, have {len(data)}")
    return data.dates[warmup_days]
