"""A dead run's lock does not block its paper forever (R2C-076).

The lock recorded the owning pid and host and never read them back, so a run
that died without releasing (killed mid-roll, crashed, machine rebooted) left
its paper unlaunchable until someone deleted the lock directory by hand. Found
2026-08-06: a forecasting roll was stopped deliberately, and the next launch of
the same paper died at startup on the lock its predecessor left behind.
"""

from __future__ import annotations

import json
import os
import socket

import pytest

from scripts.run_events import (
    RunDirectoryLock,
    RunDirectoryLockError,
    run_lock_blocks_launch,
    run_lock_owner_is_provably_gone,
)


def _pipeline(tmp_path):
    d = tmp_path / ".pipeline"
    d.mkdir(parents=True)
    return d


def _plant_lock(pipeline, *, pid, host=None, owner=True):
    lock = pipeline / "_lock"
    lock.mkdir(exist_ok=True)
    if owner:
        (lock / "owner.json").write_text(json.dumps({
            "schema_version": "1.0",
            "run_id": "some-paper",
            "pid": pid,
            "host": host if host is not None else socket.gethostname(),
            "created_at": "2026-08-06T21:22:05Z",
        }), encoding="utf-8")
    return lock


def _dead_pid() -> int:
    """A pid that is not running. Forked, reaped, and confirmed gone."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def test_a_dead_owner_on_this_host_is_reclaimed(tmp_path):
    pipeline = _pipeline(tmp_path)
    _plant_lock(pipeline, pid=_dead_pid())

    lock = RunDirectoryLock(pipeline, run_id="some-paper").acquire()

    assert lock.acquired
    owner = json.loads((pipeline / "_lock" / "owner.json").read_text())
    assert owner["pid"] == os.getpid()
    lock.release()


def test_a_live_owner_is_never_reclaimed(tmp_path):
    pipeline = _pipeline(tmp_path)
    _plant_lock(pipeline, pid=os.getpid())

    with pytest.raises(RunDirectoryLockError) as exc:
        RunDirectoryLock(pipeline, run_id="some-paper").acquire()
    assert str(os.getpid()) in str(exc.value)


def test_an_owner_on_another_host_is_never_reclaimed(tmp_path):
    # A pid on another machine says nothing about liveness here.
    pipeline = _pipeline(tmp_path)
    _plant_lock(pipeline, pid=_dead_pid(), host="some-other-machine")

    with pytest.raises(RunDirectoryLockError):
        RunDirectoryLock(pipeline, run_id="some-paper").acquire()


def test_a_lock_with_no_owner_record_is_not_reclaimed(tmp_path):
    # More likely a write that has not landed than an abandoned run. Racing it
    # would let two drivers into one run directory.
    pipeline = _pipeline(tmp_path)
    _plant_lock(pipeline, pid=0, owner=False)

    with pytest.raises(RunDirectoryLockError) as exc:
        RunDirectoryLock(pipeline, run_id="some-paper").acquire()
    assert "unreadable" in str(exc.value)


def test_an_unparseable_owner_record_is_not_reclaimed(tmp_path):
    pipeline = _pipeline(tmp_path)
    lock = _plant_lock(pipeline, pid=_dead_pid())
    (lock / "owner.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(RunDirectoryLockError):
        RunDirectoryLock(pipeline, run_id="some-paper").acquire()


def test_a_non_integer_pid_is_not_reclaimed(tmp_path):
    pipeline = _pipeline(tmp_path)
    _plant_lock(pipeline, pid="not-a-pid")

    with pytest.raises(RunDirectoryLockError):
        RunDirectoryLock(pipeline, run_id="some-paper").acquire()


def test_the_refusal_message_names_who_holds_it(tmp_path):
    # The old message said only that the directory was locked, which is the
    # least useful half of what the lock knows.
    pipeline = _pipeline(tmp_path)
    _plant_lock(pipeline, pid=os.getpid())

    with pytest.raises(RunDirectoryLockError) as exc:
        RunDirectoryLock(pipeline, run_id="some-paper").acquire()
    message = str(exc.value)
    assert socket.gethostname() in message
    assert "2026-08-06T21:22:05Z" in message


# --- the shared staleness rule -----------------------------------------------
#
# The verdict has two consumers: the lock's own acquire (which pairs it with the
# takeover) and the --fresh launch path (which asks whether archiving the run dir
# is safe). It lives in one read-only place because when it lived only inside
# acquire, --fresh kept a bare exists() check and the two disagreed.


@pytest.mark.parametrize("plant,gone,blocks", [
    pytest.param(lambda p: _plant_lock(p, pid=_dead_pid()), True, False,
                 id="dead-owner-on-this-host"),
    pytest.param(lambda p: _plant_lock(p, pid=os.getpid()), False, True,
                 id="live-owner"),
    pytest.param(lambda p: _plant_lock(p, pid=_dead_pid(), host="elsewhere"),
                 False, True, id="foreign-host"),
    pytest.param(lambda p: _plant_lock(p, pid=0, owner=False), False, True,
                 id="no-owner-record"),
    pytest.param(lambda p: _plant_lock(p, pid="not-a-pid"), False, True,
                 id="non-integer-pid"),
])
def test_the_staleness_rule_agrees_with_what_acquire_does(tmp_path, plant, gone, blocks):
    pipeline = _pipeline(tmp_path)
    plant(pipeline)

    assert run_lock_owner_is_provably_gone(pipeline) is gone
    assert run_lock_blocks_launch(pipeline) is blocks

    # And the acquire path reaches the same verdict, which is the point.
    if blocks:
        with pytest.raises(RunDirectoryLockError):
            RunDirectoryLock(pipeline, run_id="some-paper").acquire()
    else:
        RunDirectoryLock(pipeline, run_id="some-paper").acquire().release()


def test_no_lock_at_all_blocks_nothing(tmp_path):
    pipeline = _pipeline(tmp_path)
    assert run_lock_blocks_launch(pipeline) is False
    assert run_lock_owner_is_provably_gone(pipeline) is False


def test_the_rule_does_not_mutate_the_lock(tmp_path):
    """Read-only by construction: --fresh consults this BEFORE deciding whether
    to archive, so it must not disturb a lock it may end up leaving alone."""
    pipeline = _pipeline(tmp_path)
    lock = _plant_lock(pipeline, pid=_dead_pid())
    before = (lock / "owner.json").read_text(encoding="utf-8")

    assert run_lock_owner_is_provably_gone(pipeline) is True
    assert run_lock_blocks_launch(pipeline) is False

    assert lock.is_dir()
    assert (lock / "owner.json").read_text(encoding="utf-8") == before


def test_an_uncontended_lock_still_works(tmp_path):
    pipeline = _pipeline(tmp_path)
    lock = RunDirectoryLock(pipeline, run_id="some-paper").acquire()
    assert lock.acquired
    lock.release()
    assert not (pipeline / "_lock").exists()
