"""The broker interface.

v1 ships one implementation, SimulatedBroker. A real paper-trading API
adapter (Alpaca, IBKR, ...) would implement these same methods: `submit`
sends the order, and `process_open` would poll the API for fills instead of
simulating them. Everything upstream (strategy, risk, journal) stays the same.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from papertrader.core import Fill, Order, PortfolioSnapshot


class Broker(ABC):
    @abstractmethod
    def submit(self, order: Order) -> None:
        """Queue an order that the risk layer has approved."""

    @abstractmethod
    def process_open(self, at, open_prices: dict[str, float]) -> tuple[list[Fill], list[tuple[Order, str]]]:
        """Called at each session open. Returns (fills, [(cancelled order, why)])."""

    @abstractmethod
    def cancel_pending(self, predicate: Callable[[Order], bool], why: str) -> list[tuple[Order, str]]:
        """Cancel queued orders matching `predicate`."""

    @abstractmethod
    def snapshot(self, marks: dict[str, float]) -> PortfolioSnapshot:
        ...

    @property
    @abstractmethod
    def pending(self) -> list[Order]:
        ...
