"""A kill switch that stays tripped until a human resets it.

Three ways to trip it:
  * automatically, by the risk manager when a loss limit is breached;
  * from the CLI:            python -m papertrader kill -c config.yaml --reason "..."
  * with a sentinel file:    touch <state_dir>/KILL
    The sentinel works even if the Python process is wedged, and the switch
    can't be reset until someone deliberately deletes that file.

With no state directory (backtests) the switch lives in memory only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from papertrader.utils import atomic_write_json, read_json

_CLEAR = {"tripped": False, "reason": None, "at": None}


class KillSwitch:
    SENTINEL = "KILL"

    def __init__(self, state_dir: str | Path | None = None):
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self._state = dict(_CLEAR)
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
        if self.state_file is not None and self.state_file.exists():
            self._state = {**_CLEAR, **read_json(self.state_file)}

    def _save(self) -> None:
        if self.state_file is not None:
            atomic_write_json(self.state_file, self._state)

    def _sentinel_present(self) -> bool:
        return self.sentinel is not None and self.sentinel.exists()

    @property
    def tripped(self) -> bool:
        if self._sentinel_present():
            return True
        self._load()  # another process (e.g. the CLI) may have tripped it
        return bool(self._state["tripped"])

    def reason(self) -> str | None:
        if self._sentinel_present():
            return f"sentinel file present: {self.sentinel}"
        return self._state.get("reason")

    def trip(self, reason: str, at=None) -> bool:
        """Trip the switch. Returns True if this call tripped it, False if already tripped."""
        self._load()
        if self._state["tripped"]:
            return False
        self._state = {"tripped": True, "reason": reason, "at": str(pd.Timestamp(at) if at is not None else pd.Timestamp.now())}
        self._save()
        return True

    def reset(self, confirm: bool = False) -> None:
        if not confirm:
            raise PermissionError("resetting the kill switch needs explicit confirmation")
        if self._sentinel_present():
            raise PermissionError(f"delete {self.sentinel} first; the sentinel file can only be removed by hand")
        self._state = dict(_CLEAR)
        self._save()

    def status(self) -> dict:
        return {"tripped": self.tripped, "reason": self.reason(), "at": self._state.get("at")}
