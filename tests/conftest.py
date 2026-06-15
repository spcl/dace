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

# Disable UCX's FUSE-based virtual filesystem before anything imports
# ``mpi4py``. OpenMPI 5.x selects UCX as its transport, and UCX's VFS feature
# mounts a per-process FUSE filesystem backed by an ``inotify`` watch to export
# runtime introspection state. Under test fan-out (xdist workers and/or
# per-test process isolation) each process opens its own watch, and the
# default ``fs.inotify.max_user_instances`` (128 on this host) is quickly
# exhausted -- UCX then aborts at init with
# ``vfs_fuse.c ... inotify_add_watch(...) No space left on device``, crashing
# otherwise-passing tests during MPI bring-up/teardown. The VFS is a debugging
# convenience with no bearing on transport correctness, so disabling it is
# safe and removes the inotify pressure entirely. ``setdefault`` respects an
# explicit override.
os.environ.setdefault("UCX_VFS_ENABLE", "n")

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
