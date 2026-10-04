"""Small helpers with no domain knowledge."""
from __future__ import annotations

import json
import math
import os
import socket
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def is_valid_price(p) -> bool:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return False
    return math.isfinite(p) and p > 0


def atomic_write_json(path: str | Path, payload) -> None:
    """Write JSON so a crash mid-write can never leave a half-written state file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)


@contextmanager
def run_lock(path: str | Path):
    """Refuse to run twice at once against the same state directory.

    The lock is an OS advisory lock on `path` (flock on POSIX, msvcrt.locking on
    Windows), and the file is never deleted. The kernel releases the lock when its
    holder exits, however it dies (kill -9, power loss), so a crash can't leave a
    stale lock behind, and nothing has to guess from a recorded pid whether the
    holder is still alive (a guess that goes wrong across PID namespaces). The
    holder's pid, host and start time are written into the file only to say who
    holds it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)  # no O_TRUNC: the holder's details stay readable
    try:
        if not _try_lock(fd):
            raise RuntimeError(
                f"{path} is locked by {_holder(fd)}: another run may be in progress. "
                "Wait for it to finish; the lock is released when that process exits."
            )
        try:
            me = {"pid": os.getpid(), "host": socket.gethostname(), "started": datetime.now().astimezone().isoformat()}
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, json.dumps(me).encode())
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


# Windows locks are mandatory: lock a byte far past the holder's details, so they stay readable.
_WINDOWS_LOCK_OFFSET = 1 << 30


def _try_lock(fd: int) -> bool:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _holder(fd: int) -> str:
    """Who holds the lock, as its holder recorded it (for the error message only)."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        holder = json.loads(os.read(fd, 4096).decode())
        return f"process {holder['pid']} on {holder['host']} (started {holder['started']})"
    except (OSError, ValueError, TypeError, KeyError):
        return "another process"
