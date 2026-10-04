"""Paper trading: run the TradingSession once per new trading day, keeping state on disk.

Run it after the close each day (cron, Task Scheduler, or by hand):

    python -m papertrader paper-step -c config/demo_paper.yaml

It is idempotent: running twice on the same day does nothing the second
time. If you miss days, it processes each missed day in order (up to
paper.max_catchup_days, or --force), each on the data as it stood that day:
the bars are downloaded once, then cut at the day and cleaned again, newest
bar held back if it can't be confirmed yet (see data/cleaning.py). Cleaning
the whole download once would let later bars decide earlier days, so a
catch-up would end somewhere an on-time run never could. A day whose bars
were all held back is skipped; the next day's data settles it.

When is a day over? With --as-of, that date counts as closed (simulation).
Without it, a run before paper.market_close + close_buffer_minutes on the
exchange's clock (paper.market_timezone) ignores today's bar, which a feed
serves while it is still forming.

Paper trading expects ADJUSTED data (the yfinance source, or CSVs with an
adj_close column). An adjusted feed rescales all earlier prices at every
split and dividend, while the account holds share counts and marks on the
old scale. So each run first compares the closes it stored for its last few
days with the new download. If a symbol's closes all moved by one factor f,
the account is re-anchored: shares and queued quantities / f, marks and
decision prices * f, journaled as a 'corporate_action' risk event. Equity is
unchanged (q/f * p*f), a 2:1 split is no longer a 50% "loss", and dividends
are reinvested, which is the total-return accounting adjusted data implies.
In a whole-share account (portfolio.allow_fractional: false) the new share
count is rounded toward zero and the fraction paid out as cash in lieu at
the new-scale mark. Only bars that actually traded are compared: a close
carried forward over a gap, or past a held-back bar, is the last good mark,
not the vendor's print for that day. A revision that is not one clean
factor trips the kill switch if the symbol is held. With an unadjusted feed
a split still looks like a crash.

The account also stores the data, strategy, portfolio, costs and risk config
it runs under. A run whose config differs is refused, listing each change,
until re-run with --accept-config-change, which journals the diff: a limit
can't be loosened without a trace.

State lives in paper.state_dir:
    journal.sqlite      the full audit log, plus the account itself (table paper_state):
                        cash, positions, queued orders, risk state, marks, recent closes,
                        config, last day processed
    state.json          a copy of the account state for people to read; never read back
                        except to migrate from older versions, where it was the state
    kill_switch.json    kill switch state (see risk/killswitch.py)
    KILL                optional sentinel file: if present, the kill switch is on
    .lock               locked while a run is in progress (see utils.run_lock)

A step's journal rows and the account state commit in one transaction at its
end, so a crash anywhere leaves both as they were and the retry does the work
once. That includes the kill switch: an automatic trip waits in memory and is
recorded in the account state; kill_switch.json is written right after the
commit. A crash before the commit leaves no trip behind (the retry derives it
again from the same data); one between the commit and the file write is
repaired by the next step, which re-applies the committed trip unless a
person has reset the switch since. A trip from outside (the `kill` command,
the KILL file) is journaled once, as 'kill_switch_observed', by the first
step that sees it, so the journal says why orders were cancelled.

The account's row in paper_state is keyed so that moving the state directory
keeps the account (see PaperRunner.state_key). A journal holding only other
accounts' rows is refused rather than started afresh or replayed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from papertrader.config import config_to_dict
from papertrader.data import load_raw, prepare_market_data
from papertrader.engine.session import DayResult, TradingSession
from papertrader.execution.paper_broker import SimulatedBroker
from papertrader.journal import Journal
from papertrader.risk.killswitch import KillSwitch
from papertrader.risk.manager import RiskEvent, RiskManager
from papertrader.strategies import build_strategy
from papertrader.utils import atomic_write_json, is_valid_price, read_json, run_lock

CHECKED_SECTIONS = ("data", "strategy", "portfolio", "costs", "risk")  # the settings that decide what it trades
HISTORY_DAYS = 3  # recent traded closes kept per symbol to spot a re-adjusted history
FACTOR_RTOL = 1e-6  # closes "moved by one factor" if their ratios agree this closely
REVISION_RTOL = 1e-4  # a held symbol revised by more than this, inconsistently, trips the kill switch
ACCOUNT_KEY = "account"  # the paper_state row of an account whose journal lives in its own state_dir


class PaperRunner:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.source = source  # optional DataSource override (tests)
        self.state_dir = Path(cfg.paper.state_dir)
        self.state_path = self.state_dir / "state.json"
        self.journal_path = Path(cfg.paper.journal_path) if cfg.paper.journal_path else self.state_dir / "journal.sqlite"
        self.step_events: list[RiskEvent] = []  # what the last step() found before its first day

    @property
    def state_key(self) -> str:
        """The account's row in paper_state. The default journal lives in state_dir and holds
        that one account, so the key is a constant: moving or re-mounting the directory keeps
        the account. A journal elsewhere (paper.journal_path) can be shared, so there the key
        is state_dir as configured."""
        return ACCOUNT_KEY if self.cfg.paper.journal_path is None else str(self.cfg.paper.state_dir)

    def _account_row(self, keys) -> str | None:
        """The paper_state row holding this account, or None if the journal holds no account at
        all. Fails closed if it holds only other keys: starting afresh, or migrating a stale
        state.json, would fork the account or replay committed days."""
        keys = sorted(keys)
        if self.state_key in keys:
            return self.state_key
        if not keys:
            return None
        if keys == [str(self.state_dir.resolve())]:
            return keys[0]  # an earlier version keyed it by the resolved path: adopted under state_key
        raise RuntimeError(
            f"{self.journal_path} has no paper account under {self.state_key!r}, only under {keys}. If one of those "
            f"is this account (moved, or keyed by an older version), rename it: UPDATE paper_state SET "
            f"state_dir = '{self.state_key}' WHERE state_dir = '<that key>'. If not, give this account its own journal."
        )

    def load_state(self) -> dict | None:
        """The account as last committed. Read-only."""
        rows = Journal.read_paper_states(self.journal_path)
        key = self._account_row(rows)
        if key is not None:
            return rows[key]
        return read_json(self.state_path) if self.state_path.exists() else None  # older layout: state.json was the state

    def kill_switch(self) -> KillSwitch:
        return KillSwitch(self.state_dir)

    def reset_kill_switch(self) -> dict:
        """What `reset-kill` does; returns the switch's status from before. Under the run lock,
        so a reset can't land inside a step. A trip committed with the account but not yet in
        kill_switch.json (see the module docstring) is written first: the reset then covers
        it, and its generation bump tells later steps a person has dealt with it."""
        with run_lock(self.state_dir / ".lock"):
            switch = self.kill_switch()
            switch.restore(_kill_state(self.load_state()).get("last_auto_trip"))
            before = switch.status()
            switch.reset(confirm=True)
            self._forget_auto_trip()
            return before

    def _forget_auto_trip(self) -> None:
        """Drop the committed record of the account's last automatic trip once a person has
        reset it, so nothing (a deleted or replaced kill_switch.json) can ever re-apply it."""
        journal = Journal(self.journal_path)
        try:
            journal.begin()
            row = self._account_row(journal.paper_state_keys())
            state = None if row is None else journal.load_paper_state(row)
            if state is None or not _kill_state(state).get("last_auto_trip"):
                return
            state["kill_switch"] = {**_kill_state(state), "last_auto_trip": None}
            journal.save_paper_state(row, state)
            journal.commit()
        finally:
            journal.close(commit=False)
        self._write_copy(state)

    def step(self, as_of=None, force: bool = False, accept_config_change: bool = False, now=None) -> list[DayResult]:
        """Process every trading day since the last run. Returns a list of DayResult.

        `now` is the wall clock for a live run (no as_of): default the real time;
        a naive value is read as market-local time. Tests inject it.
        """
        cfg = self.cfg
        self.step_events = []
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with run_lock(self.state_dir / ".lock"):
            cutoff, wall_clock = self._cutoff(as_of, now)
            raw = load_raw(cfg.data, self.source)  # once: every day below is cut from this download
            latest, _ = prepare_market_data(raw, cfg.data, as_of=cutoff, hold_unconfirmed=True)
            if len(latest) == 0:
                raise RuntimeError("no market data available")
            journal = Journal(self.journal_path)
            switch = KillSwitch(self.state_dir, deferred=True)  # automatic trips wait for the commit
            try:
                journal.begin()  # behind the lock, a second guard: two steps can't both act on one state
                results, state = self._step(journal, switch, raw, cutoff, latest, wall_clock, force, accept_config_change)
                if state is not None:
                    journal.save_paper_state(self.state_key, state)
                    journal.commit()  # the only commit: the days' rows and the account land together
                committed = state if state is not None else journal.load_paper_state(self.state_key)
            finally:
                journal.close(commit=False)  # after an error nothing of this run is kept, so a retry can't double it
            if state is not None:
                switch.flush()  # only after the commit: a rolled-back step's trip goes with it
            if committed is not None:
                self._write_copy(committed)
            return results

    def _write_copy(self, state: dict) -> None:
        """Make state.json the committed account. Also after a step with nothing to commit,
        which repairs a copy a crash between the commit and this write left behind."""
        try:
            current = read_json(self.state_path)
        except (OSError, ValueError):
            current = None
        if current != json.loads(json.dumps(state, sort_keys=True, default=str)):
            atomic_write_json(self.state_path, state)

    def _cutoff(self, as_of, now) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(newest bar date to use, wall clock). An explicit as_of is a simulated run after
        that day's close. Live, today's bar counts only once the market has closed and the
        feed has had close_buffer_minutes to publish the final print."""
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            return as_of, as_of
        p = self.cfg.paper
        tz = ZoneInfo(p.market_timezone)
        clock = pd.Timestamp.now(tz=tz) if now is None else pd.Timestamp(now)
        if clock.tzinfo is None:
            clock = clock.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
        local = clock.tz_convert(tz).tz_localize(None)  # the exchange's wall clock
        today = local.normalize()
        hours, minutes = (int(x) for x in p.market_close.split(":"))
        final_bar_at = today + pd.Timedelta(hours=hours, minutes=minutes + p.close_buffer_minutes)
        return (today if local >= final_bar_at else today - pd.Timedelta(days=1)), local

    def _step(self, journal, switch, raw, cutoff, latest, wall_clock, force, accept_config_change):
        """Everything inside the transaction. Returns (results, new state, or None if nothing changed)."""
        cfg = self.cfg
        row = self._account_row(journal.paper_state_keys())
        state = None if row is None else journal.load_paper_state(row)
        migrated = row is not None and row != self.state_key
        if migrated:
            journal.delete_paper_state(row)  # saved under state_key below
        elif state is None and self.state_path.exists():
            state, migrated = read_json(self.state_path), True  # older layout; committed into the journal below
        elif state is None and journal.has_runs("paper"):
            raise RuntimeError(
                f"{self.journal_path} has paper runs but no account state to carry on from: refusing to start "
                "another account in it. Restore the account, or use a new paper.state_dir."
            )
        kill = _kill_state(state)
        trip = kill.get("last_auto_trip")
        # Committed, but the last run died before writing the file (or the file was lost): re-apply
        # it. It was journaled when it happened. A trip a person has reset is never re-applied:
        # reset-kill drops the record, and a reset made any other way bumps the generation.
        restored = switch.restore(trip)
        forgotten = (
            not restored
            and isinstance(trip, dict)
            and isinstance(trip.get("generation"), int)
            and trip["generation"] < switch.generation
            and not switch.tripped
        )
        if forgotten:
            kill["last_auto_trip"] = None  # a person has reset it since: never re-apply it
        strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
        strategy_desc = strategy.describe()
        if state is not None and state.get("strategy") != strategy_desc:
            raise RuntimeError(
                f"state in {self.state_dir} belongs to strategy {state.get('strategy')!r}, "
                f"config now says {strategy_desc!r}. Use a new paper.state_dir for a new strategy."
            )
        config = _checked_config(cfg)
        changes = config_changes(state["config"], config) if state is not None and "config" in state else []
        if changes and not accept_config_change:
            raise RuntimeError(
                f"config changed since the account in {self.state_dir} last ran:\n  "
                + "\n  ".join(changes)
                + "\nRestore it, or re-run with --accept-config-change to adopt it (the change is journaled)."
            )

        risk = RiskManager(cfg.risk, switch)
        if state is None:
            run_id = journal.start_run("paper", cfg.name, strategy_desc, config_to_dict(cfg))
            todo = [latest.dates[-1]]  # start trading today; no replaying history
        else:
            run_id = state["run_id"]
            last = pd.Timestamp(state["last_processed"])
            todo = [d for d in latest.dates if d > last]
            if len(todo) > cfg.paper.max_catchup_days and not force:
                raise RuntimeError(
                    f"{len(todo)} trading days to catch up (limit {cfg.paper.max_catchup_days}); "
                    "check why runs were missed, then re-run with --force"
                )
        at = todo[0] if todo else latest.dates[-1]
        if todo:
            self._observe_kill_switch(journal, run_id, switch, kill, at)
        if state is None:
            broker = SimulatedBroker(cfg.portfolio.initial_cash, cfg.costs, allow_fractional=cfg.portfolio.allow_fractional)
            marks, history = {}, {}
        else:
            if changes:
                self._log(journal, run_id, RiskEvent(at, "config_changed", "; ".join(changes)))
            self._reanchor(state, latest, risk, journal, run_id, at)
            broker = SimulatedBroker.from_state(state["broker"], cfg.costs, allow_fractional=cfg.portfolio.allow_fractional)
            risk.load_state(state.get("risk"))
            marks = {s: float(p) for s, p in state.get("marks", {}).items()}
            history = state.get("close_history", {})

        results = []
        newest_raw = _newest_bar(raw, cutoff)
        for k, day in enumerate(todo):
            # Cutting at the newest raw bar keeps the same rows as cutting at `cutoff`, so the usual
            # on-time day is exactly the latest data; only catch-up days need a cut of their own.
            data = latest if day == newest_raw else prepare_market_data(raw, cfg.data, as_of=day, hold_unconfirmed=True)[0]
            if day not in data.dates:
                continue  # every symbol's bar held back: the next day's data settles it
            session = TradingSession(data, strategy.run(data), broker, risk, cfg.portfolio, journal, run_id)
            session.marks = marks
            # Catch-up days are processed as if run on the day; only the newest
            # day is checked against the wall clock for stale data.
            results.append(session.process_day(day, now=wall_clock if k == len(todo) - 1 else day))
            marks, history = session.marks, _recent_traded(data)
        if switch.pending is not None:
            kill["last_auto_trip"] = switch.pending  # lands with this commit; the file is written after it

        # Nothing processed, migrated, recorded or re-anchored: leave everything as it was.
        if not results and (state is None or not (migrated or forgotten or "config" not in state or self.step_events)):
            return [], None
        last_processed = results[-1].date if results else pd.Timestamp(state["last_processed"])
        return results, {
            "run_id": run_id,
            "strategy": strategy_desc,
            "config": config,
            "last_processed": str(last_processed.date()),
            "broker": broker.to_state(),
            "risk": risk.state(),
            "kill_switch": kill,
            "marks": marks,
            "close_history": history,
            "updated_at": str(pd.Timestamp.now()),
        }

    def _log(self, journal: Journal, run_id, event: RiskEvent) -> None:
        journal.log_risk_event(run_id, event)
        self.step_events.append(event)

    def _observe_kill_switch(self, journal: Journal, run_id, switch: KillSwitch, kill: dict, at) -> None:
        """Journal, once, a trip this account didn't make (the `kill` command, the KILL file):
        the step is about to cancel and flatten because of it, and the journal must say why.
        kill['observed'] is the trip last seen, so later steps don't log it again; the
        account's own trips (kill['last_auto_trip']) were journaled when they happened."""
        status = switch.status()
        seen = {"reason": status["reason"], "at": status["at"], "generation": switch.generation} if status["tripped"] else None
        if seen is not None and seen not in (kill.get("observed"), kill.get("last_auto_trip")):
            event = RiskEvent(pd.Timestamp(at), "kill_switch_observed", f"kill switch is on: {seen['reason']}")
            self._log(journal, run_id, event)
        kill["observed"] = seen

    def _reanchor(self, state: dict, latest, risk: RiskManager, journal: Journal, run_id, at) -> None:
        """Put the stored account on the scale of the new download (see the module docstring)."""
        stored = state.get("close_history") or {}
        dates = sorted(stored)
        if not dates:
            return  # older state: nothing to compare yet; the next processed day records it
        fresh = _closes(latest, dates)
        held = state["broker"]["positions"]
        rescaled: dict[str, float] = {}
        for sym in sorted({s for row in stored.values() for s in row}):
            # Each symbol keeps its own last few traded bars, so one that didn't trade on the
            # account's last days (a halt) is still compared. Its newest stored bar may have been
            # a preliminary print revised since for other reasons: use it only when it is alone.
            both = [d for d in dates if sym in stored[d] and sym in fresh.get(d, {})]
            pairs = [(d, fresh[d][sym] / stored[d][sym]) for d in (both[:-1] or both)]
            if not pairs:
                continue
            ratios = [r for _, r in pairs]
            f, spread = float(np.median(ratios)), max(ratios) / min(ratios) - 1.0
            if spread <= FACTOR_RTOL:
                if abs(f - 1.0) > FACTOR_RTOL:
                    self._log(journal, run_id, _rescale(state, sym, f, at, self.cfg.portfolio.allow_fractional))
                    rescaled[sym] = f
                continue
            moves = ", ".join(f"{d} x{r:.6g}" for d, r in pairs)
            detail = f"{sym}: stored closes changed by different factors ({moves}): history revised, not just rescaled"
            if spread > REVISION_RTOL and held.get(sym, 0.0) != 0:
                # Fail closed: a position that can't be re-anchored has marks nobody can vouch for.
                # Deferred like any automatic trip, so a retry after a crash derives and logs it again, once.
                reason = f"{detail}; can't re-anchor the held position"
                if risk.kill_switch.trip(reason, at):
                    self._log(journal, run_id, RiskEvent(pd.Timestamp(at), "kill_switch_tripped", reason))
                    continue
                detail = f"{reason} (the kill switch was already on)"
            self._log(journal, run_id, RiskEvent(pd.Timestamp(at), "history_revised", detail))
        # Compare against the new basis from now on, so one revision is reported once. A stored
        # bar the new download no longer shows as traded keeps its value, on the new scale.
        state["close_history"] = {
            d: {**{s: v * rescaled.get(s, 1.0) for s, v in stored[d].items()}, **fresh.get(d, {})} for d in dates
        }

    def status(self) -> dict:
        state = self.load_state()
        ks = self._kill_switch_status(state)
        if state is None:
            return {"initialised": False, "kill_switch": ks}
        b, marks = state["broker"], state.get("marks", {})
        positions = {
            s: {"quantity": q, "mark": marks.get(s), "value": q * marks[s] if is_valid_price(marks.get(s)) else None}
            for s, q in b["positions"].items()
        }
        # Never value a position without a mark at 0: that reads as a loss (see PortfolioSnapshot.equity).
        unpriced = sorted(s for s, p in positions.items() if p["value"] is None and p["quantity"] != 0)
        equity = None if unpriced else b["cash"] + sum(p["value"] or 0.0 for p in positions.values())
        return {
            "initialised": True,
            "strategy": state.get("strategy"),
            "last_processed": state["last_processed"],
            "cash": b["cash"],
            "equity": equity,
            "positions": positions,
            "unpriced": unpriced,
            "pending_orders": b.get("pending", []),
            "kill_switch": ks,
            "journal": str(self.journal_path),
        }

    def _kill_switch_status(self, state: dict | None) -> dict:
        """kill_switch.json, or the committed trip the next step will write to it (read-only)."""
        switch = self.kill_switch()
        trip = _kill_state(state).get("last_auto_trip")
        if switch.outstanding(trip):
            return {"tripped": True, "reason": trip.get("reason"), "at": trip.get("at")}
        return switch.status()


def _kill_state(state: dict | None) -> dict:
    """The account's own kill-switch record: {last_auto_trip, observed} (copied)."""
    return dict((state or {}).get("kill_switch") or {})


def _checked_config(cfg) -> dict:
    """The sections that decide what the account trades, as the stored state holds them (JSON)."""
    full = config_to_dict(cfg)
    return json.loads(json.dumps({k: full[k] for k in CHECKED_SECTIONS}, default=str))


def config_changes(old: dict, new: dict) -> list[str]:
    """Every dotted key whose value differs, as 'key: old -> new'."""
    a, b = _flatten(old), _flatten(new)
    unset = object()
    out = []
    for key in sorted(set(a) | set(b)):
        was, now = a.get(key, unset), b.get(key, unset)
        if was != now:
            was, now = ("<unset>" if v is unset else json.dumps(v) for v in (was, now))
            out.append(f"{key}: {was} -> {now}")
    return out


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _newest_bar(raw: dict[str, pd.DataFrame], cutoff) -> pd.Timestamp | None:
    """The newest raw bar date on or before cutoff: cutting there or at cutoff keeps the same rows."""
    newest = None
    for df in raw.values():
        dates = pd.DatetimeIndex(pd.to_datetime(df.index))
        dates = dates[dates <= cutoff]
        if len(dates):
            newest = dates.max() if newest is None else max(newest, dates.max())
    return newest


def _recent_traded(data, n: int = HISTORY_DAYS) -> dict[str, dict[str, float]]:
    """{date: {symbol: close}} holding each symbol's last n bars that actually traded (see
    _closes). Per symbol, not the account's last n dates: after a halt those dates hold only
    carried-forward marks, which can't show a later re-adjustment."""
    traded = data.close.where(data.open.notna() & (data.open > 0) & (data.close > 0))
    out: dict[str, dict[str, float]] = {}
    for sym in traded.columns:
        for d, v in traded[sym].dropna().iloc[-n:].items():
            if is_valid_price(v):
                out.setdefault(str(pd.Timestamp(d).date()), {})[sym] = float(v)
    return dict(sorted(out.items()))


def _closes(data, dates) -> dict[str, dict[str, float]]:
    """{date: {symbol: close}} on `dates` (those in the data), for bars that actually traded.
    A row with no open was forward-filled over a gap, or past a held-back bar: its close is
    the last good mark carried along, not the vendor's print for that day, and comparing it
    with a later download would read a halt or a late bar as a revised history."""
    out = {}
    for d in dates:
        day = pd.Timestamp(d)
        if day in data.dates:
            opens = data.open.loc[day]
            out[str(day.date())] = {
                s: float(v) for s, v in data.close.loc[day].items() if is_valid_price(v) and is_valid_price(opens[s])
            }
    return out


def _whole(q: float) -> float:
    """Whole shares, rounded toward zero. A factor is measured from prices, so a 2:1 split can
    come out as 0.4999999: an amount that close to a whole number is that number."""
    nearest = round(q)
    return float(nearest if abs(q - nearest) <= FACTOR_RTOL * max(abs(q), 1.0) else math.trunc(q))


def _rescale(state: dict, sym: str, f: float, at, fractional: bool = True) -> RiskEvent:
    """Re-anchor one symbol to a history rescaled by f. Value is unchanged (q/f * p*f),
    so the risk manager's peak and last close stay valid. In a whole-share account the
    position is rounded toward zero and the fraction paid out as cash at the new-scale
    mark (cash in lieu, no commission); queued quantities are rounded toward zero, and
    an order left with none is dropped."""
    broker = state["broker"]
    marks = state.get("marks", {})
    mark = float(marks[sym]) * f if sym in marks else None
    if sym in marks:
        marks[sym] = mark
    old = float(broker["positions"].get(sym, 0.0))
    new, paid = old / f, ""
    if old and not fractional:
        if is_valid_price(mark):
            whole = _whole(new)
            cash = (new - whole) * mark
            broker["cash"] = float(broker["cash"]) + cash
            if cash:
                paid = f" ({new - whole:.6g} share(s) paid out at {mark:.6g}: cash {cash:+.10g} in lieu)"
            new = whole
        else:
            paid = " (no mark to pay the fraction out at: left fractional)"
    if new:
        broker["positions"][sym] = new
    else:
        broker["positions"].pop(sym, None)
    queued = [o for o in broker.get("pending", []) if o["symbol"] == sym]
    for o in queued:
        q = float(o["quantity"]) / f
        o["quantity"] = q if fractional else _whole(q)
        o["reference_price"] = float(o["reference_price"]) * f
    dropped = [o for o in queued if o["quantity"] == 0]
    if dropped:
        broker["pending"] = [o for o in broker["pending"] if not any(o is d for d in dropped)]
    return RiskEvent(
        pd.Timestamp(at),
        "corporate_action",
        f"{sym}: price history rescaled by {f:.8g} (split or dividend in adjusted data): "
        f"position {old:.10g} -> {new:.10g}{paid}, mark and {len(queued)} queued order(s) re-anchored"
        + (f" ({len(dropped)} rounded to 0 shares and dropped)" if dropped else "")
        + ", equity unchanged",
    )
