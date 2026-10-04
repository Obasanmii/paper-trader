"""An append-only SQLite log of everything the system decided and did.

Every day writes: target weights (+ the signal behind them), every proposed
order with the risk layer's verdict and reasons, every fill and cancel,
every risk event, and end-of-day equity. Nothing is updated in place, so
you can always answer "why did it do that?" after the fact.

The one exception is `paper_state`: the paper account's current state (one
row per account), kept here rather than in a separate file so it commits in
the same transaction as the day's rows. A crash can then never leave a
journal that is a day ahead of the account, which on retry would replay the
day and log it twice.

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
CREATE TABLE IF NOT EXISTS paper_state (
    state_dir TEXT PRIMARY KEY, state TEXT, updated_at TEXT
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

    # ------------------------------------------------------------------ paper account state
    # The column is still called state_dir; it holds the account's key (see PaperRunner.state_key).
    def save_paper_state(self, key: str, state: dict) -> None:
        """Part of the open transaction: it lands with the day's rows or not at all."""
        if self.enabled:
            self.conn.execute(
                "INSERT OR REPLACE INTO paper_state VALUES (?,?,?)",
                (key, json.dumps(state, sort_keys=True, default=str), str(pd.Timestamp.now())),
            )

    def load_paper_state(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        row = self.conn.execute("SELECT state FROM paper_state WHERE state_dir = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def delete_paper_state(self, key: str) -> None:
        if self.enabled:
            self.conn.execute("DELETE FROM paper_state WHERE state_dir = ?", (key,))

    def paper_state_keys(self) -> list[str]:
        if not self.enabled:
            return []
        return [k for (k,) in self.conn.execute("SELECT state_dir FROM paper_state ORDER BY state_dir")]

    def has_runs(self, mode: str) -> bool:
        if not self.enabled:
            return False
        return self.conn.execute("SELECT 1 FROM runs WHERE mode = ? LIMIT 1", (mode,)).fetchone() is not None

    @staticmethod
    def read_paper_states(path: str | Path) -> dict[str, dict]:
        """Every account in the journal, {key: state}, read-only (status): never creates or
        upgrades the file. Empty if there is no journal, or no state in it yet; any other
        error is raised, not read as "no state"."""
        path = Path(path)
        if not path.exists():
            return {}
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT state_dir, state FROM paper_state").fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return {}  # a journal from before paper_state existed
        finally:
            conn.close()
        return {key: json.loads(state) for key, state in rows}

    # ------------------------------------------------------------------ transactions
    def begin(self) -> None:
        """Open the transaction now, holding SQLite's write lock (BEGIN IMMEDIATE) before
        anything is read. Two writers that both got this far can then never act on the same
        committed state: the second waits for the first to commit and reads its result, or
        fails. (By default sqlite3 only begins at the first write, after the reads.)"""
        if self.enabled:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc):
                    raise
                raise RuntimeError(f"{self.path} is locked: another run may be in progress ({exc})") from exc

    def commit(self) -> None:
        if self.enabled:
            self.conn.commit()

    def rollback(self) -> None:
        if self.enabled:
            self.conn.rollback()

    def close(self, commit: bool = True) -> None:
        """Backtests commit on close, even after an error (the rows up to it are what a
        post-mortem needs). The paper runner passes commit=False: a half-processed day
        must not be kept, or the retry would log it twice."""
        if self.enabled and self.conn is not None:
            if commit:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
            self.conn = None
            self.enabled = False

    def query(self, sql: str, params=()) -> pd.DataFrame:
        if not self.enabled:
            raise RuntimeError("journal is disabled")
        return pd.read_sql_query(sql, self.conn, params=params)
