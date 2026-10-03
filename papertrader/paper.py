"""Paper trading: run the TradingSession once per new trading day, keeping state on disk.

Run it after the close each day (cron, Task Scheduler, or by hand):

    python -m papertrader paper-step -c config/demo_paper.yaml

It is idempotent: running twice on the same day does nothing the second
time. If you miss days, it processes each missed day in order (up to
paper.max_catchup_days, or --force). That produces the same result as if it
had run on time, because strategies can't see past each decision date.

State lives in paper.state_dir:
    state.json          cash, positions, queued orders, risk state, last day processed
    kill_switch.json    kill switch state (see risk/killswitch.py)
    KILL                optional sentinel file: if present, the kill switch is on
    journal.sqlite      the full audit log
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from papertrader.config import config_to_dict
from papertrader.data import load_market_data
from papertrader.engine.session import TradingSession
from papertrader.execution.paper_broker import SimulatedBroker
from papertrader.journal import Journal
from papertrader.risk.killswitch import KillSwitch
from papertrader.risk.manager import RiskManager
from papertrader.strategies import build_strategy
from papertrader.utils import atomic_write_json, read_json, run_lock


class PaperRunner:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.source = source  # optional DataSource override (tests)
        self.state_dir = Path(cfg.paper.state_dir)
        self.state_path = self.state_dir / "state.json"
        self.journal_path = Path(cfg.paper.journal_path) if cfg.paper.journal_path else self.state_dir / "journal.sqlite"

    def load_state(self) -> dict | None:
        return read_json(self.state_path) if self.state_path.exists() else None

    def kill_switch(self) -> KillSwitch:
        return KillSwitch(self.state_dir)

    def step(self, as_of=None, force: bool = False) -> list:
        """Process every trading day since the last run. Returns a list of DayResult."""
        cfg = self.cfg
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with run_lock(self.state_dir / ".lock"):
            data, _ = load_market_data(cfg.data, as_of=as_of, source=self.source)
            if len(data) == 0:
                raise RuntimeError("no market data available")
            state = self.load_state()
            strategy_desc = build_strategy(cfg.strategy.name, cfg.strategy.params).describe()
            if state is not None and state.get("strategy") != strategy_desc:
                raise RuntimeError(
                    f"state in {self.state_dir} belongs to strategy {state.get('strategy')!r}, "
                    f"config now says {strategy_desc!r}. Use a new paper.state_dir for a new strategy."
                )

            risk = RiskManager(cfg.risk, self.kill_switch())
            journal = Journal(self.journal_path)
            try:
                if state is None:
                    broker = SimulatedBroker(cfg.portfolio.initial_cash, cfg.costs, allow_fractional=cfg.portfolio.allow_fractional)
                    run_id = journal.start_run("paper", cfg.name, strategy_desc, config_to_dict(cfg))
                    todo = [data.dates[-1]]  # start trading today; no replaying history
                else:
                    broker = SimulatedBroker.from_state(state["broker"], cfg.costs, allow_fractional=cfg.portfolio.allow_fractional)
                    risk.load_state(state.get("risk"))
                    run_id = state["run_id"]
                    last = pd.Timestamp(state["last_processed"])
                    todo = [d for d in data.dates if d > last]
                    if not todo:
                        return []
                    if len(todo) > cfg.paper.max_catchup_days and not force:
                        raise RuntimeError(
                            f"{len(todo)} trading days to catch up (limit {cfg.paper.max_catchup_days}); "
                            "check why runs were missed, then re-run with --force"
                        )

                strategy = build_strategy(cfg.strategy.name, cfg.strategy.params)
                session = TradingSession(data, strategy.run(data), broker, risk, cfg.portfolio, journal, run_id)
                if state is not None:
                    session.marks = {s: float(p) for s, p in state.get("marks", {}).items()}
                    session.peak_equity = state.get("peak_equity")

                wall_clock = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
                results = []
                for k, day in enumerate(todo):
                    # Catch-up days are processed as if run on the day; only the newest
                    # day is checked against the wall clock for stale data.
                    now = wall_clock if k == len(todo) - 1 else day
                    results.append(session.process_day(day, now=now))
                journal.commit()
                atomic_write_json(
                    self.state_path,
                    {
                        "run_id": run_id,
                        "strategy": strategy_desc,
                        "last_processed": str(todo[-1].date()),
                        "broker": broker.to_state(),
                        "risk": risk.state(),
                        "marks": session.marks,
                        "peak_equity": session.peak_equity,
                        "updated_at": str(pd.Timestamp.now()),
                    },
                )
                return results
            finally:
                journal.close()

    def status(self) -> dict:
        state = self.load_state()
        ks = self.kill_switch().status()
        if state is None:
            return {"initialised": False, "kill_switch": ks}
        b, marks = state["broker"], state.get("marks", {})
        positions = {
            s: {"quantity": q, "mark": marks.get(s), "value": q * marks[s] if s in marks else None}
            for s, q in b["positions"].items()
        }
        equity = b["cash"] + sum(p["value"] or 0.0 for p in positions.values())
        return {
            "initialised": True,
            "strategy": state.get("strategy"),
            "last_processed": state["last_processed"],
            "cash": b["cash"],
            "equity": equity,
            "positions": positions,
            "pending_orders": b.get("pending", []),
            "kill_switch": ks,
            "journal": str(self.journal_path),
        }
