# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``os.fork`` guard installed by ``tests/conftest.py``.

A fork from a process holding a live OpenMP team deadlocks the child on libgomp's team barrier, so
the guard tears every loaded runtime's pool down first and refuses the fork only when that teardown
cannot be performed. Both halves matter: refusing a fork the teardown already made safe turned a
whole directory red at random, and allowing one it did not make safe hangs the suite with no verdict.
"""
import os
import signal

import numpy as np
import pytest

import dace
from dace.transformation.layout.isolation import pause_openmp_pools
from tests.conftest import UNGUARDED_FORK, guarded_fork, openmp_pool_may_outlive_fork, thread_count

N = dace.symbol("N", dtype=dace.int64)


@dace.program
def doubler(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0


@pytest.fixture(scope="module")
def live_team():
    """A compiled CPU kernel that has already run here, i.e. the state the guard exists to police."""
    compiled = doubler.compile()
    a = np.arange(1 << 14, dtype=np.float64)
    b = np.zeros_like(a)
    compiled(a=a, b=b, N=a.size)
    assert np.allclose(b, a * 2.0)
    return compiled, a, b


def fork_and_join(child_work, timeout=120):
    """Fork through the guard, run ``child_work`` in the child, and return the child's exit code.

    A deadlocked child would hang the suite, which is the very failure under test, so the wait
    carries its own deadline and reports the hang as a value rather than as a stuck run.
    """
    pid = os.fork()
    if pid == 0:  # child
        try:
            child_work()
            os._exit(0)
        except BaseException:  # noqa: BLE001 - never raise past a fork
            os._exit(17)

    def on_alarm(signum, frame):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(timeout)
    try:
        _, status = os.waitpid(pid, 0)
    except TimeoutError:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        return "deadlocked"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return os.WEXITSTATUS(status) if os.WIFEXITED(status) else f"signal {os.WTERMSIG(status)}"


def test_fork_after_a_cpu_kernel_is_allowed_and_the_child_runs_its_own_parallel_region(live_team):
    """The CI failure, inverted: a kernel has run, so a team was built, and the fork must still go
    through -- with the child able to enter a parallel region of its own, which is what proves the
    teardown really happened rather than the guard merely having looked the other way."""
    compiled, a, b = live_team

    def child_work():
        out = np.zeros_like(a)
        compiled(a=a, b=out, N=a.size)  # a parallel region in the child: deadlocks if the team was stranded
        assert np.allclose(out, a * 2.0)

    assert fork_and_join(child_work) == 0


def test_pause_reports_the_teardown_it_performed(live_team):
    """The signal the guard now decides on. libgomp is mapped and its pool has been built by the
    fixture, so a runtime that implements ``omp_pause_resource_all`` must report success."""
    assert pause_openmp_pools() is True
    assert openmp_pool_may_outlive_fork() is False


def test_fork_is_refused_when_the_pool_cannot_be_torn_down(live_team, monkeypatch):
    """A runtime predating OpenMP 5.0, or a fork from inside a parallel region: the teardown does not
    happen, the child would wait on a barrier forever, and the guard must refuse instead."""
    import dace.transformation.layout.isolation as isolation
    monkeypatch.setattr(isolation, "pause_openmp_pools", lambda mode=isolation.OMP_PAUSE_SOFT: False)
    monkeypatch.setattr("tests.conftest.thread_count", lambda: 8)  # a one-core host has no team either way
    assert openmp_pool_may_outlive_fork() is True
    with pytest.raises(RuntimeError, match="could not be torn down"):
        guarded_fork()


def test_a_thread_census_drop_is_not_read_as_a_stranded_team(live_team, monkeypatch):
    """The regression itself. Any thread exiting while the pools are being torn down -- pytest-timeout's
    per-test timer is the one that did it -- makes the process lose threads across that window for
    reasons that have nothing to do with OpenMP. Reading the drop as a stranded team refused forks
    that were already safe, so the fork has to survive a census that collapses under it."""
    compiled, a, _ = live_team
    counts = iter([64, 63, 1])  # collapses however many times the census is consulted
    monkeypatch.setattr("tests.conftest.thread_count", lambda: next(counts, 1))
    assert openmp_pool_may_outlive_fork() is False

    def child_work():
        out = np.zeros_like(a)
        compiled(a=a, b=out, N=a.size)
        assert np.allclose(out, a * 2.0)

    assert fork_and_join(child_work) == 0


def test_the_guard_is_installed_and_wraps_the_real_fork():
    """Non-vacuity: every fork in this suite goes through the guard, and the guard still forks."""
    assert os.fork is guarded_fork
    assert UNGUARDED_FORK is not guarded_fork
    assert thread_count() >= 1


if __name__ == "__main__":
    pytest.main([__file__])
