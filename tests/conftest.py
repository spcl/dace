# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
pytest configuration file.
"""
import os

# Disable hwloc's GL/X11 topology backend before anything imports ``mpi4py``.
# ``from mpi4py import MPI`` auto-calls ``MPI_Init``; OpenMPI 5.x asks hwloc to
# detect topology, and hwloc's GL backend connects to every local X11 display
# (including stale/orphaned Xwayland sockets) to enumerate NVIDIA GPUs via the
# NV-CONTROL extension. A dead display makes that connect block forever, hanging
# pytest at collection/teardown. Disabling only the GL probe leaves CPU/memory
# binding intact. ``setdefault`` respects an explicit override.
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

import pytest


@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # If running MPI tests and a failure has been detected, terminate the process to notify MPI to stop the other ranks
    if config.option.markexpr == 'mpi':
        if exitstatus in (pytest.ExitCode.TESTS_FAILED, pytest.ExitCode.INTERNAL_ERROR, pytest.ExitCode.INTERRUPTED):
            os._exit(1)


def pytest_generate_tests(metafunc):
    """
    This method sets up the parametrizations for the custom fixtures
    """
    if "use_cpp_dispatcher" in metafunc.fixturenames:
        metafunc.parametrize("use_cpp_dispatcher", [
            pytest.param(True, id="use_cpp_dispatcher"),
            pytest.param(False, id="no_use_cpp_dispatcher"),
        ])
