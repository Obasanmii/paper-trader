"""run_lock: one run at a time, and a lock that dies with its holder, however it dies."""
import json
import os
import socket
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from papertrader import utils
from papertrader.utils import run_lock

HOST = socket.gethostname()
ROOT = Path(__file__).parents[1]


def hold_lock(lock):
    """Another process that takes the lock, says so, and holds it until its stdin closes."""
    code = textwrap.dedent(
        f"""
        import sys
        from papertrader.utils import run_lock
        with run_lock({str(lock)!r}):
            print("locked", flush=True)
            sys.stdin.read()
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    assert proc.stdout.readline().strip() == "locked"
    return proc


def no_pid_probing(monkeypatch):
    """Whatever the lock file says, nobody should be asking the OS about that pid."""
    def kill(pid, sig):  # pragma: no cover - failing is the point
        raise AssertionError(f"probed pid {pid}")

    monkeypatch.setattr(utils.os, "kill", kill)


def test_the_lock_says_who_holds_it_and_refuses_a_second_run(tmp_path):
    lock = tmp_path / ".lock"
    with run_lock(lock):
        holder = json.loads(lock.read_text())
        assert (holder["pid"], holder["host"]) == (os.getpid(), HOST) and holder["started"]
        with pytest.raises(RuntimeError, match=f"is locked by process {os.getpid()} on {HOST} .*another run may be in progress"):
            with run_lock(lock):
                pass  # pragma: no cover
    assert lock.exists()  # never deleted: deleting it is what let two runs both "hold" it
    with run_lock(lock):
        pass


def test_another_process_is_refused_while_the_first_holds_the_lock(tmp_path):
    lock = tmp_path / ".lock"
    holder = hold_lock(lock)
    try:
        with pytest.raises(RuntimeError, match=f"is locked by process {holder.pid} on {HOST} "):
            with run_lock(lock):
                pass  # pragma: no cover
    finally:
        holder.stdin.close()
        assert holder.wait(10) == 0
    with run_lock(lock):
        assert json.loads(lock.read_text())["pid"] == os.getpid()


def test_a_lock_left_by_a_killed_process_does_not_block(tmp_path, monkeypatch, capsys):
    """kill -9 (or a power cut) releases the lock with the process: no stale-lock cleanup needed."""
    lock = tmp_path / ".lock"
    holder = hold_lock(lock)
    holder.kill()
    holder.wait(10)
    holder.stdout.close()
    holder.stdin.close()
    assert json.loads(lock.read_text())["pid"] == holder.pid  # its details are still in the file
    no_pid_probing(monkeypatch)
    with run_lock(lock):
        assert json.loads(lock.read_text())["pid"] == os.getpid()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "content",
    [
        {"pid": 4242, "host": HOST, "started": "x"},  # a pid that may or may not be alive: irrelevant
        {"pid": os.getpid(), "host": HOST, "started": "x"},
        {"pid": 4242, "host": "other-machine"},
        "4242",  # the old pid-only format
        "",
        "{not json",
    ],
)
def test_what_an_unheld_lock_file_says_never_matters(tmp_path, monkeypatch, content):
    lock = tmp_path / ".lock"
    lock.write_text(content if isinstance(content, str) else json.dumps(content))
    no_pid_probing(monkeypatch)
    with run_lock(lock):
        assert json.loads(lock.read_text())["pid"] == os.getpid()


def test_a_live_holder_is_respected_whatever_pid_it_recorded(tmp_path, monkeypatch):
    """The old check read the recorded pid and probed it. A holder in another PID namespace
    (a container sharing the volume and hostname) has a pid that doesn't exist here, so its
    live lock was 'stale' and got taken over. The kernel's lock knows better."""
    lock = tmp_path / ".lock"
    holder = hold_lock(lock)
    try:
        lock.write_text(json.dumps({"pid": 2**22 + 12345, "host": HOST, "started": "x"}))  # no such pid here
        no_pid_probing(monkeypatch)
        with pytest.raises(RuntimeError, match="another run may be in progress"):
            with run_lock(lock):
                pass  # pragma: no cover
    finally:
        holder.stdin.close()
        holder.wait(10)


def test_windows_locks_a_byte_past_the_holders_details(tmp_path, monkeypatch):
    """msvcrt locks are mandatory, so the locked byte must not cover what the error message reads."""
    calls = []
    held = set()

    def locking(fd, mode, nbytes):
        at = os.lseek(fd, 0, os.SEEK_CUR)
        calls.append((mode, at, nbytes))
        if mode == fake.LK_NBLCK:
            if at in held:
                raise OSError("locked")
            held.add(at)
        else:
            held.discard(at)

    fake = types.SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    fd = os.open(tmp_path / ".lock", os.O_RDWR | os.O_CREAT)
    try:
        with monkeypatch.context() as m:
            m.setattr(utils.os, "name", "nt")
            assert utils._try_lock(fd) is True
            assert utils._try_lock(fd) is False
            utils._unlock(fd)
            assert utils._try_lock(fd) is True
    finally:
        os.close(fd)
    offset = utils._WINDOWS_LOCK_OFFSET
    assert offset > 4096 and calls == [(2, offset, 1), (2, offset, 1), (0, offset, 1), (2, offset, 1)]
