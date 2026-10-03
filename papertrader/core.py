"""Core types shared by every layer. Deliberately small."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class OrderStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    FILLED = "filled"
    PARTIAL = "partially_filled"
    CANCELLED = "cancelled"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Order:
    """A market order. `quantity` is signed: positive buys, negative sells."""

    symbol: str
    quantity: float
    reference_price: float  # the price the decision was based on (last close)
    created_at: pd.Timestamp
    reason: str = ""
    order_id: str = field(default_factory=_new_id)
    status: OrderStatus = OrderStatus.PROPOSED

    @property
    def side(self) -> str:
        return "buy" if self.quantity > 0 else "sell"

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.reference_price)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "reference_price": self.reference_price,
            "created_at": str(pd.Timestamp(self.created_at)),
            "reason": self.reason,
            "order_id": self.order_id,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Order":
        return cls(
            symbol=d["symbol"],
            quantity=float(d["quantity"]),
            reference_price=float(d["reference_price"]),
            created_at=pd.Timestamp(d["created_at"]),
            reason=d.get("reason", ""),
            order_id=d["order_id"],
            status=OrderStatus(d.get("status", "approved")),
        )


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    quantity: float  # signed, what actually traded (may be less than ordered)
    price: float  # includes slippage
    commission: float
    timestamp: pd.Timestamp

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.price)


@dataclass
class PortfolioSnapshot:
    """What the risk layer is allowed to see: holdings, cash and marks.

    Note what is *not* here: anything about strategies, signals or models.
    """

    cash: float
    positions: dict[str, float]
    prices: dict[str, float]

    def missing_marks(self) -> list[str]:
        return [s for s, q in self.positions.items() if q != 0 and not _valid_price(self.prices.get(s))]

    @property
    def net_exposure(self) -> float:
        return sum(q * self.prices[s] for s, q in self.positions.items() if q != 0)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(q * self.prices[s]) for s, q in self.positions.items() if q != 0)

    @property
    def equity(self) -> float:
        missing = self.missing_marks()
        if missing:
            # Fail loudly: valuing a position at zero would look like a crash
            # and could trip (or, worse, mask) risk limits.
            raise KeyError(f"no valid mark for held positions: {missing}")
        return self.cash + self.net_exposure


def _valid_price(p) -> bool:
    return p is not None and math.isfinite(p) and p > 0
