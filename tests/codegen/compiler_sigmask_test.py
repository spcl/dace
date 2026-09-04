# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests the SIGCHLD unblocking that keeps CMake from hanging under MPI/Slurm launchers."""

import os
import signal

import pytest

from dace.codegen.compiler import build_subprocess_sigmask

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='pthread_sigmask/SIGCHLD are POSIX-only')


def test_sigchld_unblocked_inside_and_restored_after():
    """Simulates a launcher: SIGCHLD blocked on entry must be deliverable inside, blocked again after."""
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
    try:
        with build_subprocess_sigmask():
            assert signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, [])
        assert signal.SIGCHLD in signal.pthread_sigmask(signal.SIG_BLOCK, [])
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})


def test_unblocked_mask_is_left_alone():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    with build_subprocess_sigmask():
        assert signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, [])
    assert signal.SIGCHLD not in signal.pthread_sigmask(signal.SIG_BLOCK, [])


if __name__ == '__main__':
    test_sigchld_unblocked_inside_and_restored_after()
    test_unblocked_mask_is_left_alone()
