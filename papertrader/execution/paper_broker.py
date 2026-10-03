"""A simulated paper broker.

Timing: orders decided after the close of day t are queued, then filled at
the open of day t+1. A decision can never trade at a price it used.

Costs: every fill pays slippage (the fill price is moved against you by
`slippage_bps`) plus commission (`commission_bps` of notional, with an
optional minimum). Both are deliberately on by default: a strategy that
only works for free doesn't work.

Cash: sells are processed before buys. Without margin, a buy that can't be
afforded at the actual open (prices gap) is cut to what cash allows and
recorded as a partial fill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from papertrader.core import Fill, Order, OrderStatus, PortfolioSnapshot
from papertrader.execution.broker import Broker
from papertrader.utils import is_valid_price


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    min_commission: float = 0.0

    def __post_init__(self):
        if self.commission_bps < 0 or self.slippage_bps < 0 or self.min_commission < 0:
            raise ValueError("costs can't be negative")


class SimulatedBroker(Broker):
    def __init__(
        self,
        initial_cash: float,
        costs: CostModel | None = None,
        allow_fractional: bool = False,
        allow_margin: bool = False,
    ):
        self.cash = float(initial_cash)
        self.positions: dict[str, float] = {}
        self.costs = costs or CostModel()
        self.allow_fractional = allow_fractional
        self.allow_margin = allow_margin
        self._pending: list[Order] = []
        self.total_commission = 0.0
        self.total_slippage = 0.0

    # ------------------------------------------------------------------ orders
    def submit(self, order: Order) -> None:
        order.status = OrderStatus.APPROVED
        self._pending.append(order)

    @property
    def pending(self) -> list[Order]:
        return list(self._pending)

    def cancel_pending(self, predicate: Callable[[Order], bool], why: str):
        keep, cancelled = [], []
        for order in self._pending:
            if predicate(order):
                order.status = OrderStatus.CANCELLED
                cancelled.append((order, why))
            else:
                keep.append(order)
        self._pending = keep
        return cancelled

    def process_open(self, at, open_prices: dict[str, float]):
        at = pd.Timestamp(at)
        fills: list[Fill] = []
        cancels: list[tuple[Order, str]] = []
        queue, self._pending = self._pending, []
        queue.sort(key=lambda o: o.quantity > 0)  # sells first, to free up cash
        rate = self.costs.commission_bps / 1e4
        slip = self.costs.slippage_bps / 1e4
        for order in queue:
            px = open_prices.get(order.symbol)
            if not is_valid_price(px):
                order.status = OrderStatus.CANCELLED
                cancels.append((order, "no tradable open price today"))
                continue
            fill_px = px * (1 + slip) if order.quantity > 0 else px * (1 - slip)
            qty = float(order.quantity)
            if qty > 0 and not self.allow_margin:
                affordable = (self.cash - self.costs.min_commission) / (fill_px * (1 + rate))
                if not self.allow_fractional:
                    affordable = math.floor(affordable + 1e-9)
                if affordable <= 0:
                    order.status = OrderStatus.CANCELLED
                    cancels.append((order, "insufficient cash"))
                    continue
                qty = min(qty, affordable)
            commission = max(self.costs.min_commission, abs(qty) * fill_px * rate)
            self.cash -= qty * fill_px + commission
            new_position = self.positions.get(order.symbol, 0.0) + qty
            if abs(new_position) < 1e-9:
                self.positions.pop(order.symbol, None)
            else:
                self.positions[order.symbol] = new_position
            self.total_commission += commission
            self.total_slippage += abs(qty) * abs(fill_px - px)
            order.status = OrderStatus.FILLED if qty == order.quantity else OrderStatus.PARTIAL
            fills.append(Fill(order.order_id, order.symbol, qty, fill_px, commission, at))
        return fills, cancels

    # ------------------------------------------------------------------ state
    def snapshot(self, marks: dict[str, float]) -> PortfolioSnapshot:
        return PortfolioSnapshot(cash=self.cash, positions=dict(self.positions), prices=dict(marks))

    def to_state(self) -> dict:
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "pending": [o.to_dict() for o in self._pending],
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
        }

    @classmethod
    def from_state(cls, state: dict, costs: CostModel, allow_fractional: bool = False, allow_margin: bool = False):
        broker = cls(state["cash"], costs, allow_fractional=allow_fractional, allow_margin=allow_margin)
        broker.positions = {s: float(q) for s, q in state.get("positions", {}).items()}
        broker._pending = [Order.from_dict(d) for d in state.get("pending", [])]
        broker.total_commission = float(state.get("total_commission", 0.0))
        broker.total_slippage = float(state.get("total_slippage", 0.0))
        return broker
