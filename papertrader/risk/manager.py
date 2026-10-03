"""The risk manager: the only gate between decisions and the broker.

It knows nothing about strategies. It sees proposed orders, the portfolio and
the latest marks, and it can do three things: approve an order, reject it
with reasons, or trip the kill switch.

Design rules:
  * Fail closed. Missing prices, NaN quantities, unknown symbols and missing
    marks all mean "reject", never "skip the check".
  * Reject, don't resize. Quietly shrinking an order hides the problem; a
    rejection with a reason shows up in the journal.
  * Orders that only shrink an existing position always get through (after
    sanity checks). You must always be able to get out, especially after
    the kill switch trips.
  * Orders in a batch are checked against *projected* positions, so several
    individually acceptable orders can't jointly breach a limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from papertrader.core import Order, PortfolioSnapshot
from papertrader.risk.killswitch import KillSwitch
from papertrader.risk.limits import RiskLimits
from papertrader.utils import is_valid_price


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    reducing: bool = False


@dataclass
class RiskEvent:
    timestamp: pd.Timestamp
    kind: str
    detail: str


def is_reducing(current: float, quantity: float) -> bool:
    """True if the order shrinks an existing position without flipping its sign."""
    try:
        current, quantity = float(current), float(quantity)
    except (TypeError, ValueError):
        return False
    if current == 0 or not math.isfinite(quantity):
        return False
    new = current + quantity
    return abs(new) <= abs(current) and (new == 0 or math.copysign(1, new) == math.copysign(1, current))


class RiskManager:
    def __init__(self, limits: RiskLimits, kill_switch: KillSwitch | None = None):
        self.limits = limits
        self.kill_switch = kill_switch if kill_switch is not None else KillSwitch()
        self.peak_equity: float | None = None
        self.last_close_equity: float | None = None
        self.current_day: pd.Timestamp | None = None
        self.orders_today = 0
        self.events: list[RiskEvent] = []

    # ------------------------------------------------------------------ lifecycle
    def start_day(self, day) -> None:
        day = pd.Timestamp(day)
        if self.current_day is None or day != self.current_day:
            self.current_day = day
            self.orders_today = 0

    def end_of_day(self, day, equity: float) -> list[RiskEvent]:
        """Check loss limits at the close. May trip the kill switch."""
        new: list[RiskEvent] = []
        L = self.limits
        if not math.isfinite(equity) or equity <= 0:
            new.append(self._trip(day, f"equity is {equity!r}"))
        else:
            if self.last_close_equity:
                change = equity / self.last_close_equity - 1
                if change <= -L.max_daily_loss_pct:
                    new.append(self._trip(day, f"daily loss {change:.2%} breached the {L.max_daily_loss_pct:.0%} limit"))
            self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)
            drawdown = 1 - equity / self.peak_equity
            if drawdown >= L.max_drawdown_pct:
                new.append(self._trip(day, f"drawdown {drawdown:.2%} breached the {L.max_drawdown_pct:.0%} limit"))
            self.last_close_equity = equity
        return [e for e in new if e is not None]

    def _trip(self, at, reason: str) -> RiskEvent | None:
        if self.kill_switch.trip(reason, at):
            event = RiskEvent(pd.Timestamp(at), "kill_switch_tripped", reason)
            self.events.append(event)
            return event
        return None

    # ------------------------------------------------------------------ orders
    def check_orders(self, orders: list[Order], snapshot: PortfolioSnapshot, now=None, data_as_of=None):
        """Check a batch. Returns [(order, RiskDecision)], exposure-reducing orders first."""
        positions = dict(snapshot.positions)
        ordered = sorted(orders, key=lambda o: 0 if is_reducing(positions.get(o.symbol, 0.0), o.quantity) else 1)
        missing = snapshot.missing_marks()
        equity = math.nan if missing else snapshot.equity
        results = []
        for order in ordered:
            decision = self._check_one(order, positions, snapshot.prices, equity, missing, now, data_as_of)
            if decision.approved:
                positions[order.symbol] = positions.get(order.symbol, 0.0) + order.quantity
                if not decision.reducing:
                    self.orders_today += 1
            results.append((order, decision))
        return results

    def _check_one(self, order, positions, prices, equity, missing_marks, now, data_as_of) -> RiskDecision:
        L = self.limits
        sym, ref = order.symbol, order.reference_price

        # Sanity checks apply to every order, reducing or not.
        try:
            q = float(order.quantity)
        except (TypeError, ValueError):
            return RiskDecision(False, [f"invalid quantity {order.quantity!r}"])
        if not math.isfinite(q) or q == 0:
            return RiskDecision(False, [f"invalid quantity {order.quantity!r}"])
        if not is_valid_price(ref):
            return RiskDecision(False, [f"invalid reference price {ref!r}"])
        px = prices.get(sym)
        if not is_valid_price(px):
            return RiskDecision(False, [f"no valid market price for {sym}"])

        current = positions.get(sym, 0.0)
        new = current + q
        if is_reducing(current, q):
            return RiskDecision(True, [], reducing=True)

        reasons = []
        if self.kill_switch.tripped:
            reasons.append(f"kill switch active ({self.kill_switch.reason()}): only exposure-reducing orders allowed")
        if L.symbol_whitelist is not None and sym not in L.symbol_whitelist:
            reasons.append(f"{sym} is not on the symbol whitelist")
        if self.orders_today >= L.max_orders_per_day:
            reasons.append(f"daily order limit reached ({L.max_orders_per_day})")
        deviation = abs(ref / px - 1)
        if deviation > L.max_price_deviation_pct:
            reasons.append(f"reference price {ref:.4g} is {deviation:.1%} away from market {px:.4g}")
        if L.max_data_age_days is not None and now is not None and data_as_of is not None:
            age = (pd.Timestamp(now).normalize() - pd.Timestamp(data_as_of).normalize()).days
            if age > L.max_data_age_days:
                reasons.append(f"market data is {age} days old (limit {L.max_data_age_days})")
        if missing_marks:
            reasons.append(f"can't value the portfolio, no marks for {missing_marks}")
        elif not math.isfinite(equity) or equity <= 0:
            reasons.append(f"equity is {equity!r}")
        else:
            notional = abs(q) * px
            if notional > L.max_order_notional:
                reasons.append(f"order notional {notional:,.0f} exceeds {L.max_order_notional:,.0f}")
            if new < 0 and not L.allow_short:
                reasons.append("short selling is disabled")
            position_pct = abs(new) * px / equity
            if position_pct > L.max_position_pct + 1e-9:
                reasons.append(f"position would be {position_pct:.1%} of equity (limit {L.max_position_pct:.0%})")
            gross = abs(new) * px
            for s, qty in positions.items():
                if s != sym and qty != 0:
                    mark = prices.get(s)
                    if not is_valid_price(mark):
                        reasons.append(f"no mark for held position {s}")
                        break
                    gross += abs(qty) * mark
            else:
                if gross / equity > L.max_gross_exposure_pct + 1e-9:
                    reasons.append(
                        f"gross exposure would be {gross / equity:.1%} of equity (limit {L.max_gross_exposure_pct:.0%})"
                    )
        return RiskDecision(not reasons, reasons, reducing=False)

    def flatten_orders(self, snapshot: PortfolioSnapshot, at) -> list[Order]:
        """Orders that close every open position (used after the kill switch trips)."""
        return [
            Order(sym, -qty, snapshot.prices[sym], pd.Timestamp(at), reason="kill switch: flatten")
            for sym, qty in sorted(snapshot.positions.items())
            if qty != 0 and is_valid_price(snapshot.prices.get(sym))
        ]

    # ------------------------------------------------------------------ persistence
    def state(self) -> dict:
        return {
            "peak_equity": self.peak_equity,
            "last_close_equity": self.last_close_equity,
            "current_day": None if self.current_day is None else str(self.current_day.date()),
            "orders_today": self.orders_today,
        }

    def load_state(self, state: dict | None) -> None:
        if not state:
            return
        self.peak_equity = state.get("peak_equity")
        self.last_close_equity = state.get("last_close_equity")
        day = state.get("current_day")
        self.current_day = None if day is None else pd.Timestamp(day)
        self.orders_today = int(state.get("orders_today", 0))
