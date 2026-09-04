# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A rank that skipped mpi4py's automatic ``MPI_Init`` must still find MPI up after ``import dace``.

``from mpi4py import MPI`` initializes MPI as a side effect, and ``MPI4PY_RC_INITIALIZE=0`` turns
that off so an unlaunched parse cannot bootstrap a singleton MPI job. The switch is process-global
and one-way: nothing initializes MPI afterwards, so under a launcher the first communicator call
aborts the whole job with ``MPI_Comm_rank() was called before MPI_INIT`` and no Python traceback.
Fifty-odd test modules export it at import time, so collecting the suite poisoned every later MPI
test in the same interpreter -- which is how the MPI CI job died with an empty report.
"""
import os
import shutil
import subprocess
import sys
import types

import pytest

from dace.frontend.python.preprocessing import ensure_mpi_initialized
from dace.sdfg.sdfg import MPI_RANK_VARS

#: Run under a launcher, this brings MPI up and reports a rank and the thread level MPI granted;
#: without the fix the rank call aborts, and with a bare ``MPI_Init`` the level comes back SINGLE.
PROBE = ('import dace\n'
         'from mpi4py import MPI\n'
         'print("rank", MPI.COMM_WORLD.Get_rank(), "level", MPI.Query_thread(), MPI.THREAD_FUNNELED)')


class FakeMPI(types.ModuleType):
    """An ``mpi4py.MPI`` that starts down and records the bring-up."""

    #: The levels, ordered as MPI orders them.
    THREAD_SINGLE, THREAD_FUNNELED, THREAD_SERIALIZED, THREAD_MULTIPLE = 0, 1, 2, 3

    def __init__(self):
        super().__init__('mpi4py.MPI')
        self.initialized = False
        self.init_calls = 0
        self.requested_levels = []

    def Is_initialized(self):
        return self.initialized

    def Is_finalized(self):
        return False

    def Init(self):
        raise AssertionError('bare MPI_Init promises MPI_THREAD_SINGLE; use Init_thread')

    def Init_thread(self, required):
        self.initialized = True
        self.init_calls += 1
        self.requested_levels.append(required)
        return required

    def Finalize(self):
        self.initialized = False


@pytest.fixture
def fake_mpi(monkeypatch):
    """Swap in an uninitialized MPI and unset every launcher variable the environment exports."""
    mpi = FakeMPI()
    package = types.ModuleType('mpi4py')
    package.MPI = mpi
    monkeypatch.setitem(sys.modules, 'mpi4py', package)
    monkeypatch.setitem(sys.modules, 'mpi4py.MPI', mpi)
    for var in MPI_RANK_VARS:
        monkeypatch.delenv(var, raising=False)
    return mpi


@pytest.mark.parametrize('rank_var', MPI_RANK_VARS)
def test_launched_rank_gets_mpi_initialized(fake_mpi, monkeypatch, rank_var):
    """Every launcher's rank variable means "this process is a rank of a job": bring MPI up."""
    monkeypatch.setenv(rank_var, '0')

    ensure_mpi_initialized()

    assert fake_mpi.init_calls == 1


def test_bring_up_asks_for_thread_multiple(fake_mpi, monkeypatch):
    """Bare ``MPI_Init`` promises MPI_THREAD_SINGLE -- one thread in the process -- and no DaCe
    process keeps that promise: its maps are OpenMP regions and its BLAS spawns a pool. Open MPI
    believes it and drops the locking around its shared-memory transport, so ScaLAPACK's panel
    broadcasts come back corrupted and PBLAS returns a different wrong answer on every call.
    mpi4py's own bring-up asks for MPI_THREAD_MULTIPLE; a rank that took this path instead must be
    indistinguishable from one that did not."""
    monkeypatch.setenv(MPI_RANK_VARS[0], '0')

    ensure_mpi_initialized()

    assert fake_mpi.requested_levels == [fake_mpi.THREAD_MULTIPLE]


def test_already_initialized_rank_is_left_alone(fake_mpi, monkeypatch):
    """MPI_Init is not idempotent -- a second call aborts the job."""
    monkeypatch.setenv(MPI_RANK_VARS[0], '0')
    fake_mpi.initialized = True

    ensure_mpi_initialized()

    assert fake_mpi.init_calls == 0


def test_unlaunched_process_is_left_down(fake_mpi):
    """No launcher: the switch guards against exactly the singleton bring-up that stalls, so obey it."""
    ensure_mpi_initialized()

    assert fake_mpi.init_calls == 0


def test_slurm_task_alone_is_not_a_rank(fake_mpi, monkeypatch):
    """``srun`` sets SLURM_PROCID for steps that run no MPI at all -- enough to name a build folder,
    not enough to call MPI_Init on."""
    monkeypatch.setenv('SLURM_PROCID', '0')

    ensure_mpi_initialized()

    assert fake_mpi.init_calls == 0


def test_unreachable_mpi_is_not_an_import_error(monkeypatch):
    """``import dace`` runs this: a machine with the wheel but no libmpi must still import dace."""

    class UnusableMPI4Py(types.ModuleType):

        def __getattr__(self, name):
            raise RuntimeError('cannot load MPI library')

    monkeypatch.setitem(sys.modules, 'mpi4py', UnusableMPI4Py('mpi4py'))
    monkeypatch.delitem(sys.modules, 'mpi4py.MPI', raising=False)
    monkeypatch.setenv(MPI_RANK_VARS[0], '0')

    ensure_mpi_initialized()  # must not raise


def launcher_free_env() -> dict:
    """The ambient environment minus the rank of the launcher that may be running this test."""
    return {key: value for key, value in os.environ.items() if key not in MPI_RANK_VARS}


@pytest.mark.skipif(shutil.which('mpirun') is None, reason='needs an MPI launcher')
def test_poisoned_rank_can_still_talk_to_its_communicator():
    """The regression end to end: a real rank with the switch set must reach COMM_WORLD."""
    env_probe = subprocess.run([sys.executable, '-c', 'from mpi4py import MPI'], capture_output=True)
    if env_probe.returncode != 0:
        pytest.skip('mpi4py cannot reach an MPI runtime here')

    result = subprocess.run(['mpirun', '-n', '2', sys.executable, '-c', PROBE],
                            capture_output=True,
                            text=True,
                            timeout=300,
                            env={
                                **launcher_free_env(), 'MPI4PY_RC_INITIALIZE': '0'
                            })

    assert result.returncode == 0, result.stderr
    assert result.stdout.count('rank') == 2, result.stdout

    # ... and at a thread level that admits the other threads the process has. Below FUNNELED an
    # MPI is entitled to run unlocked, which is what turned a pgemm into nondeterministic garbage.
    for line in result.stdout.split('\n'):
        if not line.startswith('rank '):
            continue
        _, _, _, level, funneled = line.split()
        assert int(level) >= int(funneled), f'MPI came up below MPI_THREAD_FUNNELED: {line}'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
