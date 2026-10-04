"""A persistent record of every research trial, so the trial count can't be reset.

The deflated Sharpe ratio is only honest if N is every parameter set ever
tried on the same data, not just the latest grid. Without a memory, "sweep,
look at out-of-sample, tweak the grid, sweep again" looks exactly like a
clean first run. The ledger remembers both: every in-sample trial and every
out-of-sample window that was looked at.

"Same data" is matched, not hashed. A re-download that back-adjusts a
dividend or jitters prices in the 15th digit is the same market history,
and so is the same universe listed in another order, or the same bars read
from a CSV copy instead of the original source; an exact hash calls each of
those a new question and silently resets the count. Two runs are on matching
data when they test the same strategy on the same symbol set and each
symbol's in-sample returns line up (ReturnPanel.matches). A run then counts,
from every earlier run on matching data:

* every distinct parameter set tried, whatever its split date or warmup
  (that is the deflated Sharpe's N), and
* every out-of-sample window that overlaps its own (the "already evaluated"
  warning), so moving the split by a day or changing the warmup doesn't
  make a spent holdout look fresh.

For N, matching needs the two in-sample windows to share 90% of EACH one's
days, so a split moved by more than about a tenth of the in-sample period
starts a fresh trial count. For the warning it is enough that the shorter
window is 90% covered by the longer: a re-split into an out-of-sample period
already looked at, or a start date moved, is still the same market and the
holdout is still spent.

Nothing is updated or deleted, so the history can always be audited:

    sqlite3 state/research_ledger.sqlite "select run_id, strategy, split_date, oos_start, oos_end from runs"
"""
from __future__ import annotations

import contextlib
import datetime as dt
import functools
import hashlib
import json
import math
import sqlite3
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 2  # 1 keyed runs by an exact hash of the closes; it is not migrated
SCHEMA = [
    """CREATE TABLE panels (
        digest TEXT PRIMARY KEY, symbols TEXT, n_dates INTEGER, dates BLOB, returns BLOB
    )""",
    """CREATE TABLE runs (
        run_id TEXT PRIMARY KEY, recorded_at TEXT, strategy TEXT, source TEXT, symbols TEXT, split_date TEXT,
        warmup_days INTEGER, oos_start TEXT, oos_end TEXT, best_params TEXT, panel TEXT REFERENCES panels(digest)
    )""",
    "CREATE TABLE trials (run_id TEXT REFERENCES runs(run_id), params TEXT, sharpe REAL, sharpe_per_period REAL)",
    "CREATE INDEX idx_runs_data ON runs(strategy, source, symbols)",
    "CREATE INDEX idx_trials_run ON trials(run_id)",
]

MIN_CORRELATION = 0.99  # per symbol, daily and monthly in-sample log returns
MIN_COVERAGE = 0.90  # days both runs have, as a share of each run's in-sample days
MIN_SHARED_DAYS = 252  # same market (nested): this many shared days is enough, however long the windows
REVISED_SHARE = 0.005  # daily-match test ignores this share of most-disagreeing days (corrected ticks)
MIN_MONTHS = 6  # fewer calendar months than this: the daily test alone decides


def params_key(params: dict) -> str:
    """Canonical JSON, so the same parameter set always dedupes to one trial."""
    return json.dumps(params, sort_keys=True, default=str)


@dataclass(frozen=True, eq=False)
class ReturnPanel:
    """Daily log returns of each symbol, (dates x sorted symbols), NaN where a symbol has none.

    float32 is plenty to tell one market history from another and halves what the ledger stores.
    """

    symbols: tuple[str, ...]
    days: np.ndarray  # int64 days since 1970-01-01
    returns: np.ndarray  # float32, len(days) x len(symbols)

    @classmethod
    def from_data(cls, data) -> ReturnPanel:
        close = data.close.reindex(columns=sorted(data.symbols))  # column order is not part of the data
        r = np.log(close).diff().iloc[1:]
        days = r.index.to_numpy(dtype="datetime64[D]").astype(np.int64)
        return cls(tuple(close.columns), days, r.to_numpy(dtype=np.float32))

    @functools.cached_property
    def digest(self) -> str:
        h = hashlib.sha256(json.dumps(self.symbols).encode())
        h.update(self.days.astype("<i8").tobytes())
        h.update(self.returns.astype("<f4").tobytes())
        return h.hexdigest()

    def to_blobs(self) -> tuple[bytes, bytes]:
        return zlib.compress(self.days.astype("<i8").tobytes()), zlib.compress(self.returns.astype("<f4").tobytes())

    @classmethod
    def from_blobs(cls, symbols: str, n_dates: int, dates: bytes, returns: bytes) -> ReturnPanel:
        syms = tuple(json.loads(symbols))
        values = np.frombuffer(zlib.decompress(returns), dtype="<f4").reshape(n_dates, len(syms))
        return cls(syms, np.frombuffer(zlib.decompress(dates), dtype="<i8"), values)

    def matches(self, other: ReturnPanel, nested: bool = False) -> bool:
        """Same market history, allowing for revisions. Per symbol, the days both panels have a
        return must cover >= 90% of each panel's days (with nested=True, "same market": 90% of
        the shorter one's days, or a year of them, whichever is less, so a sliding window that
        moves start and split together is still recognised), and on those days the log returns
        must correlate >= 0.99, both day by day and summed by calendar month (once there are 6
        months).

        A back-adjusted dividend or rounding barely moves either correlation. A corrected bad
        tick can: one 10% outlier day pulls a daily correlation below 0.99, so both tests leave
        out the 0.5% of days where the two disagree most (at least two: the print and its
        reversal, which can fall in different months). A different drift hides in daily returns but
        not in monthly ones: the random-walk and trending demos share their daily shocks and
        correlate 0.996 day by day, but about 0.92 month by month. Different shocks (another
        seed) correlate near 0.
        """
        if self.symbols != other.symbols:
            return False
        if self.digest == other.digest:
            return True
        days, ia, ib = np.intersect1d(self.days, other.days, assume_unique=True, return_indices=True)
        compared = 0
        for k in range(len(self.symbols)):
            a, b = self.returns[:, k], other.returns[:, k]
            na, nb = int(np.isfinite(a).sum()), int(np.isfinite(b).sum())
            if na == nb == 0:
                continue  # no in-sample returns for this symbol in either run: nothing to compare
            x, y = a[ia].astype(float), b[ib].astype(float)
            both = np.isfinite(x) & np.isfinite(y)
            need = min(MIN_COVERAGE * min(na, nb), MIN_SHARED_DAYS) if nested else MIN_COVERAGE * max(na, nb)
            if both.sum() < need:
                return False
            keep = _unrevised(x[both], y[both])
            x, y, shared = x[both][keep], y[both][keep], days[both][keep]
            months, month = np.unique(shared.astype("datetime64[D]").astype("datetime64[M]"), return_inverse=True)
            mx, my = np.bincount(month, weights=x), np.bincount(month, weights=y)
            if not _correlated(x, y) or (months.size >= MIN_MONTHS and not _correlated(mx, my)):
                return False
            compared += 1
        return compared > 0


def _unrevised(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask leaving out the REVISED_SHARE of days where the two series disagree most (at least
    two: one corrected print moves two daily returns, into it and out of it), once there are
    enough days to spare, so a few corrected prints don't decide the match."""
    keep = np.ones(x.size, dtype=bool)
    drop = max(2, int(REVISED_SHARE * x.size)) if x.size >= 40 else 0
    if drop:
        keep[np.argsort(np.abs(x - y), kind="stable")[x.size - drop :]] = False
    return keep


def _correlated(x: np.ndarray, y: np.ndarray) -> bool:
    if x.size and np.allclose(x, y, rtol=0.0, atol=1e-6):
        return True  # equal up to float32 noise, including the no-variance case corrcoef can't score
    if x.size < 3:
        return False
    with np.errstate(invalid="ignore", divide="ignore"):
        return bool(np.corrcoef(x, y)[0, 1] >= MIN_CORRELATION)


@dataclass(frozen=True, eq=False)
class Study:
    """What one research run tested, on which data, and which out-of-sample window it looked at."""

    strategy: str
    source: str  # data.source (synthetic, csv or yfinance): recorded for the audit trail, not used to match
    panel: ReturnPanel  # in-sample returns, from the first date up to the split
    split_date: str
    warmup_days: int
    oos_start: str
    oos_end: str

    @property
    def symbols(self) -> str:
        return json.dumps(list(self.panel.symbols))


def study(strategy: str, source: str, data, split_date, warmup_days: int, oos_start, oos_end) -> Study:
    """Describe a research run. Only data up to the split is kept: the out-of-sample data is
    identified by its window, so appending days doesn't make the same history look new."""
    split = pd.Timestamp(split_date)

    def day(x) -> str:
        return pd.Timestamp(x).date().isoformat()

    panel = ReturnPanel.from_data(data.truncate(split))
    return Study(strategy, source, panel, day(split), int(warmup_days), day(oos_start), day(oos_end))


@dataclass(frozen=True)
class LedgerTrial:
    params: str  # canonical JSON (params_key) of the full parameter set the strategy ran with
    sharpe: float  # annualised
    sharpe_per_period: float  # what the deflated Sharpe ratio needs


@dataclass(frozen=True)
class Prior:
    """Earlier runs of a study's strategy on matching data, as the ledger held them at one moment."""

    run_ids: tuple[str, ...] = ()
    trials: dict[str, LedgerTrial] = field(default_factory=dict)  # distinct parameter set -> its latest result
    oos_overlaps: int = 0  # earlier runs whose out-of-sample window overlaps the study's


def _num(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None  # NaN -> NULL


def _nan(x) -> float:
    return math.nan if x is None else float(x)


class TrialLedger:
    """Append-only SQLite store of research runs, their trials and out-of-sample windows.

    Any database error becomes a RuntimeError rather than a skipped record:
    a ledger that silently stops recording is exactly how a re-sweep goes unnoticed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = None
        try:
            with self._loud():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Autocommit mode: every transaction below is an explicit BEGIN ... COMMIT.
                self.conn = sqlite3.connect(str(self.path), timeout=60, isolation_level=None)
                with self._transaction("IMMEDIATE"):
                    self._check_schema()
        except BaseException:
            self.close()
            raise

    def _check_schema(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version == 0 and not self.conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
            for statement in SCHEMA:
                self.conn.execute(statement)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        raise RuntimeError(
            f"research ledger {self.path}: not a version-{SCHEMA_VERSION} trial ledger (version {version}: an older "
            "papertrader's, or another database). Old ledgers are not migrated: rename this file, keeping it for the "
            "record, and run again to start a new ledger."
        )

    @contextlib.contextmanager
    def _loud(self):
        try:
            yield
        except (OSError, sqlite3.Error) as exc:
            raise RuntimeError(f"research ledger {self.path}: {exc}") from exc

    @contextlib.contextmanager
    def _transaction(self, mode: str = "DEFERRED"):
        self.conn.execute(f"BEGIN {mode}")
        try:
            yield
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:  # SQLite may already have rolled back (e.g. disk full)
                self.conn.execute("ROLLBACK")
            raise

    def prior(self, s: Study) -> Prior:
        """Earlier runs of `s.strategy` on data matching `s`: their trials and overlapping looks."""
        with self._loud(), self._transaction():  # one consistent snapshot
            return self._prior(s)

    def _prior(self, s: Study) -> Prior:
        # Not filtered by data.source: the same bars read from a CSV copy are the same market,
        # and the documented workflow (download once, then switch to csv) must keep the count.
        same = (s.strategy, s.symbols)
        runs = self.conn.execute(
            "SELECT run_id, oos_start, oos_end, panel FROM runs WHERE strategy = ? AND symbols = ? ORDER BY rowid",
            same,
        ).fetchall()
        matching: dict[str, tuple[bool, bool]] = {}  # panel digest -> (same data, same market)? Re-runs share a panel.
        run_ids, overlaps = [], 0
        for run_id, oos_start, oos_end, digest in runs:
            if digest not in matching:
                if digest == s.panel.digest:
                    matching[digest] = (True, True)
                else:
                    other = self._panel(digest)
                    matching[digest] = (s.panel.matches(other), s.panel.matches(other, nested=True))
            same_data, same_market = matching[digest]
            if same_data:
                run_ids.append(run_id)
            if same_market:
                overlaps += oos_start <= s.oos_end and oos_end >= s.oos_start  # ISO dates compare as text
        wanted = set(run_ids)
        rows = self.conn.execute(
            "SELECT t.run_id, t.params, t.sharpe, t.sharpe_per_period FROM trials t JOIN runs r ON t.run_id = r.run_id "
            "WHERE r.strategy = ? AND r.symbols = ? ORDER BY t.rowid",
            same,
        ).fetchall()
        trials = {p: LedgerTrial(p, _nan(sr), _nan(spp)) for run_id, p, sr, spp in rows if run_id in wanted}
        return Prior(tuple(run_ids), trials, overlaps)

    def _panel(self, digest: str) -> ReturnPanel:
        row = self.conn.execute("SELECT symbols, n_dates, dates, returns FROM panels WHERE digest = ?", (digest,)).fetchone()
        if row is None:
            raise sqlite3.DatabaseError(f"run refers to a missing return panel {digest}")
        try:
            return ReturnPanel.from_blobs(*row)
        except (ValueError, TypeError, zlib.error) as exc:
            raise sqlite3.DatabaseError(f"unreadable return panel {digest}: {exc}") from exc

    def record_run(self, s: Study, trials: list[LedgerTrial], best_params: dict) -> tuple[str, Prior]:
        """One transaction: this run's trials and its out-of-sample window, or neither.

        Returns the run id and the earlier runs as re-read under the same write lock. A run
        another process recorded after `prior` was read shows up here, so compare the two and
        recompute anything that used the first.
        """
        run_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        dates, returns = s.panel.to_blobs()  # compress before taking the lock
        with self._loud(), self._transaction("IMMEDIATE"):
            seen = self._prior(s)
            self.conn.execute(
                "INSERT OR IGNORE INTO panels VALUES (?,?,?,?,?)", (s.panel.digest, s.symbols, len(s.panel.days), dates, returns)
            )
            self.conn.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, at, s.strategy, s.source, s.symbols, s.split_date, s.warmup_days, s.oos_start, s.oos_end,
                 params_key(best_params), s.panel.digest),
            )
            self.conn.executemany(
                "INSERT INTO trials VALUES (?,?,?,?)",
                [(run_id, t.params, _num(t.sharpe), _num(t.sharpe_per_period)) for t in trials],
            )
        return run_id, seen

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> TrialLedger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
