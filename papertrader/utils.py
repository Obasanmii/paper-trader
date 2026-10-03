"""Small helpers with no domain knowledge."""
from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import contextmanager
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
    """Refuse to run twice at once against the same state directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{path} exists: another run may be in progress. "
            "If you're sure nothing else is running, delete it and retry."
        ) from None
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
