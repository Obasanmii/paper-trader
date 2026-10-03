"""Turn target weights into proposed orders. Proposals only: risk decides."""
from __future__ import annotations

import math

import pandas as pd

from papertrader.core import Order, PortfolioSnapshot
from papertrader.utils import is_valid_price


def orders_from_targets(targets: dict[str, float], snapshot: PortfolioSnapshot, prices: dict[str, float], at, cfg) -> list[Order]:
    """Size each symbol to `weight * equity * (1 - cash_buffer)`, in whole shares unless fractional is allowed.

    Rebalancing is all-or-nothing: if any position is off target by more than
    `rebalance_band` (or must be closed), every position is brought
    back to target together. Rebalancing only the one that breached, while
    others sit above target inside their bands, can push gross exposure over
    100%. The risk layer caught exactly that bug in an earlier version of
    this function.
    """
    try:
        equity = snapshot.equity
    except KeyError:
        return []  # can't size anything without a full valuation (risk would refuse anyway)
    if not math.isfinite(equity) or equity <= 0:
        return []
    investable = equity * (1.0 - cfg.cash_buffer_pct)
    at = pd.Timestamp(at)
    candidates, triggered = [], False
    for sym in sorted(set(targets) | set(snapshot.positions)):
        px = prices.get(sym)
        if not is_valid_price(px):
            continue  # no price today, no trade
        w = targets.get(sym, 0.0)
        w = 0.0 if w is None or not math.isfinite(w) else float(w)
        current = snapshot.positions.get(sym, 0.0)
        target_qty = w * investable / px
        if not cfg.allow_fractional:
            target_qty = float(math.trunc(target_qty))
        delta = target_qty - current
        if delta == 0:
            continue
        current_w = current * px / equity
        closing = target_qty == 0 and current != 0
        if closing or abs(w - current_w) >= cfg.rebalance_band:
            triggered = True
        candidates.append((sym, delta, px, w, current_w, closing))
    if not triggered:
        return []
    return [
        Order(sym, delta, px, at, reason=f"target {w:.2%}, holding {current_w:.2%}")
        for sym, delta, px, w, current_w, closing in candidates
        if closing or abs(delta) * px >= cfg.min_trade_notional
    ]
