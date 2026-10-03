"""An append-only SQLite log of everything the system decided and did.

Every day writes: target weights (+ the signal behind them), every proposed
order with the risk layer's verdict and reasons, every fill and cancel,
every risk event, and end-of-day equity. Nothing is updated in place, so
you can always answer "why did it do that?" after the fact.

    sqlite3 state/demo/journal.sqlite "select * from orders where event='rejected' limit 5"
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, mode TEXT, name TEXT, strategy TEXT, config TEXT, started_at TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    run_id TEXT, date TEXT, symbol TEXT, target_weight REAL, signal REAL
);
CREATE TABLE IF NOT EXISTS orders (
    run_id TEXT, order_id TEXT, date TEXT, event TEXT, symbol TEXT, quantity REAL,
    reference_price REAL, reason TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    run_id TEXT, order_id TEXT, date TEXT, symbol TEXT, quantity REAL, price REAL, commission REAL
);
CREATE TABLE IF NOT EXISTS risk_events (
    run_id TEXT, date TEXT, kind TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    run_id TEXT, date TEXT, cash REAL, equity REAL, gross_exposure REAL, net_exposure REAL, drawdown REAL
);
CREATE INDEX IF NOT EXISTS idx_orders_run ON orders(run_id, date);
CREATE INDEX IF NOT EXISTS idx_equity_run ON equity(run_id, date);
"""


def _day(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN -> NULL


class Journal:
    """Pass path=None for a no-op journal (fast parameter sweeps)."""

    def __init__(self, path: str | Path | None = None):
        self.path = path
        self.enabled = path is not None
        self.conn: sqlite3.Connection | None = None
        if self.enabled:
            if str(path) != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path))
            self.conn.executescript(SCHEMA)

    def _write(self, sql: str, rows) -> None:
        if self.enabled and rows:
            self.conn.executemany(sql, rows)

    def start_run(self, mode: str, name: str, strategy: str, config: dict | None = None) -> str:
        run_id = f"{pd.Timestamp.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self._write(
            "INSERT INTO runs VALUES (?,?,?,?,?,?)",
            [(run_id, mode, name, strategy, json.dumps(config or {}, default=str), str(pd.Timestamp.now()))],
        )
        return run_id

    def log_decisions(self, run_id, date, targets: dict, signals: dict | None = None) -> None:
        signals = signals or {}
        self._write(
            "INSERT INTO decisions VALUES (?,?,?,?,?)",
            [(run_id, _day(date), s, _num(w), _num(signals.get(s))) for s, w in targets.items()],
        )

    def log_order(self, run_id, order, date, event: str, reasons=()) -> None:
        self._write(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    run_id,
                    order.order_id,
                    _day(date),
                    event,
                    order.symbol,
                    _num(order.quantity),
                    _num(order.reference_price),
                    order.reason,
                    "; ".join(reasons),
                )
            ],
        )

    def log_fill(self, run_id, fill) -> None:
        self._write(
            "INSERT INTO fills VALUES (?,?,?,?,?,?,?)",
            [(run_id, fill.order_id, _day(fill.timestamp), fill.symbol, fill.quantity, fill.price, fill.commission)],
        )

    def log_risk_event(self, run_id, event) -> None:
        self._write("INSERT INTO risk_events VALUES (?,?,?,?)", [(run_id, _day(event.timestamp), event.kind, event.detail)])

    def log_equity(self, run_id, date, cash, equity, gross, net, drawdown) -> None:
        self._write(
            "INSERT INTO equity VALUES (?,?,?,?,?,?,?)",
            [(run_id, _day(date), _num(cash), _num(equity), _num(gross), _num(net), _num(drawdown))],
        )

    def commit(self) -> None:
        if self.enabled:
            self.conn.commit()

    def close(self) -> None:
        if self.enabled and self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None
            self.enabled = False

    def query(self, sql: str, params=()) -> pd.DataFrame:
        if not self.enabled:
            raise RuntimeError("journal is disabled")
        return pd.read_sql_query(sql, self.conn, params=params)
