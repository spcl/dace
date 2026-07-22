# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CMake build subprocess must be forked with SIGCHLD deliverable. MPI/Slurm launchers
(``srun``, ``mpirun``) start their tasks with SIGCHLD *blocked* in the signal mask, which
every child inherits; CMake/KWSys reaps the helper processes it spawns during configure
(``uname``, compiler-id / ABI test binaries) via SIGCHLD, so with SIGCHLD blocked it spins
forever in ``select()`` leaving ``<defunct>`` children -- the daint configure hang.
``_build_subprocess_sigmask`` unblocks SIGCHLD only for the duration of the fork and
restores the caller's mask afterwards."""
import signal

import pytest

from dace.codegen import compiler

_UNSUPPORTED = not hasattr(signal, 'pthread_sigmask') or not hasattr(signal, 'SIGCHLD')


@pytest.mark.skipif(_UNSUPPORTED, reason='pthread_sigmask/SIGCHLD unavailable on this platform')
def test_sigmask_unblocks_sigchld_when_blocked_and_restores():
    original = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        # Reproduce the launcher condition: SIGCHLD blocked on entry.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
        assert signal.SIGCHLD in signal.pthread_sigmask(signal.SIG_BLOCK, [])

        with compiler._build_subprocess_sigmask():
            # Inside the context (i.e. across the fork) SIGCHLD must be deliverable.
            assert signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, [])

        # The caller's mask must be restored exactly after the context exits.
        assert signal.SIGCHLD in signal.pthread_sigmask(signal.SIG_BLOCK, [])
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, original)


@pytest.mark.skipif(_UNSUPPORTED, reason='pthread_sigmask/SIGCHLD unavailable on this platform')
def test_sigmask_is_noop_when_sigchld_already_deliverable():
    original = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
        assert signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, [])

        with compiler._build_subprocess_sigmask():
            assert signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, [])

        # Still deliverable, and no spurious change to the rest of the mask.
        assert signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, [])
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, original)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
