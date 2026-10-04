"""A kill switch that stays tripped until a human resets it.

Three ways to trip it:
  * automatically, by the risk manager when a loss limit is breached;
  * from the CLI:            python -m papertrader kill -c config.yaml --reason "..."
  * with a sentinel file:    touch <state_dir>/KILL
    The sentinel works even if the Python process is wedged, and the switch
    can't be reset until someone deliberately deletes that file.

With no state directory (backtests) the switch lives in memory only.

The paper runner uses it `deferred`: an automatic trip is held in memory
(and honoured by `tripped`) until `flush()`, which the runner calls only once
the step that made it has committed. A step that fails is rolled back, and
its trip must go with it: a trip left on disk would be active from the first
open the retry replays, days before it happened. Manual and sentinel trips
are always immediate.

kill_switch.json also counts resets (`generation`), so a trip recorded
elsewhere (the paper account's committed state) can tell whether a person
has reset the switch since: see `outstanding`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from papertrader.utils import atomic_write_json, read_json

_CLEAR = {"tripped": False, "reason": None, "at": None, "generation": 0}


class KillSwitch:
    SENTINEL = "KILL"

    def __init__(self, state_dir: str | Path | None = None, deferred: bool = False):
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.deferred = deferred
        self._state = dict(_CLEAR)
        self._pending: dict | None = None  # a deferred trip, not on disk yet
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    @property
    def state_file(self) -> Path | None:
        return None if self.state_dir is None else self.state_dir / "kill_switch.json"

    @property
    def sentinel(self) -> Path | None:
        return None if self.state_dir is None else self.state_dir / self.SENTINEL

    def _load(self) -> None:
        """Re-read the file (another process may have tripped or reset it). A pending trip is kept apart."""
        if self.state_file is not None and self.state_file.exists():
            state = {**_CLEAR, **read_json(self.state_file)}
            gen = state["generation"]
            if isinstance(gen, bool) or not isinstance(gen, int) or gen < 0:
                # Fail closed: the generation decides whether a recorded trip still applies.
                raise ValueError(f"{self.state_file}: generation must be an integer >= 0, got {gen!r}")
            self._state = state

    def _save(self) -> None:
        if self.state_file is not None:
            atomic_write_json(self.state_file, self._state)

    def _sentinel_present(self) -> bool:
        return self.sentinel is not None and self.sentinel.exists()

    def _current(self) -> dict:
        """The trip in force: the file's, else a pending one."""
        return self._pending if self._pending is not None and not self._state["tripped"] else self._state

    @property
    def tripped(self) -> bool:
        if self._sentinel_present():
            return True
        self._load()  # another process (e.g. the CLI) may have tripped it
        return bool(self._state["tripped"]) or self._pending is not None

    @property
    def generation(self) -> int:
        """How many times a trip has been reset."""
        self._load()
        return self._state["generation"]

    @property
    def pending(self) -> dict | None:
        """The deferred trip flush() has yet to write, as {reason, at, generation}, or None."""
        return None if self._pending is None else {k: self._pending[k] for k in ("reason", "at", "generation")}

    def reason(self) -> str | None:
        if self._sentinel_present():
            return f"sentinel file present: {self.sentinel}"
        return self._current().get("reason")

    def trip(self, reason: str, at=None) -> bool:
        """Trip the switch. Returns True if this call tripped it, False if already tripped."""
        self._load()
        if self._state["tripped"] or self._pending is not None:
            return False
        trip = {"tripped": True, "reason": reason, "at": str(pd.Timestamp(at) if at is not None else pd.Timestamp.now())}
        if self.deferred:
            self._pending = {**trip, "generation": self._state["generation"]}
        else:
            self._state = {**self._state, **trip}
            self._save()
        return True

    def flush(self) -> bool:
        """Write a deferred trip to disk. True if it was written; if the switch was tripped
        on disk in the meantime (CLI, sentinel aside), that trip and its reason stand."""
        pending, self._pending = self._pending, None
        if pending is None:
            return False
        self._load()
        if self._state["tripped"]:
            return False
        self._state = {**self._state, **{k: pending[k] for k in ("tripped", "reason", "at")}}
        self._save()
        return True

    def outstanding(self, trip: dict | None) -> bool:
        """True if `trip` ({reason, at, generation}), recorded elsewhere, should be in force but
        isn't on disk: the file isn't tripped and nobody has reset the switch since (a reset
        bumps the generation). A generation that went backwards means the file was deleted or
        replaced, which is no reset: the trip still stands."""
        if not isinstance(trip, dict) or not isinstance(trip.get("generation"), int):
            return False
        self._load()
        return not self._state["tripped"] and trip["generation"] >= self._state["generation"]

    def restore(self, trip: dict | None) -> bool:
        """Write a trip that was recorded elsewhere but never reached the file, e.g. the paper
        runner died between its commit and flush(). Only if `outstanding`. True if written."""
        if not self.outstanding(trip):
            return False
        self._state = {**self._state, "tripped": True, "reason": trip.get("reason"), "at": trip.get("at")}
        self._save()
        return True

    def reset(self, confirm: bool = False) -> None:
        if not confirm:
            raise PermissionError("resetting the kill switch needs explicit confirmation")
        if self._sentinel_present():
            raise PermissionError(f"delete {self.sentinel} first; the sentinel file can only be removed by hand")
        self._load()
        # Only resetting a trip counts: one that cleared nothing mustn't cancel a trip still on its way to the file.
        generation = self._state["generation"] + (1 if self._state["tripped"] or self._pending is not None else 0)
        self._state = {**_CLEAR, "generation": generation}
        self._pending = None
        self._save()

    def status(self) -> dict:
        tripped = self.tripped
        return {"tripped": tripped, "reason": self.reason(), "at": self._current().get("at") if tripped else None}
