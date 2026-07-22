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

# Steer OpenMPI off UCX entirely before ``mpi4py`` is imported (dace's frontend
# does ``from mpi4py import MPI`` lazily inside ``to_sdfg``, which auto-calls
# ``MPI_Init``). On hosts where UCX bring-up stalls -- e.g. a wedged RDMA device
# or the VFS/inotify pressure above -- ``MPI_Init`` can block forever. The
# ``ob1`` point-to-point messaging layer over the ``self`` (loopback) and
# ``vader`` (shared-memory) BTLs is sufficient for the single-node test runs and
# avoids the UCX path completely. ``setdefault`` keeps any externally-provided
# MPI configuration.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")

import pytest

#: Seed applied before every test, so a numerical comparison that only fails on some inputs fails
#: on every run instead of intermittently. Tests that want their own draw still call
#: ``np.random.seed`` / ``default_rng(seed)`` themselves; this only fixes the starting state.
GLOBAL_RANDOM_SEED = 0


@pytest.fixture(autouse=True)
def seeded_global_rng():
    """Reseed NumPy's legacy global RNG before each test."""
    import numpy as np
    np.random.seed(GLOBAL_RANDOM_SEED)


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


def _active_cuda_impl():
    # Imported lazily so pytest collection works even if the dace package can't be imported.
    from dace.config import Config
    return Config.get('compiler', 'cuda', 'implementation')


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked old_gpu_codegen_only / new_gpu_codegen_only based on the
    current ``compiler.cuda.implementation`` config value."""
    try:
        impl = _active_cuda_impl()
    except Exception:
        return  # If dace config is unavailable, don't interfere with collection.

    skip_old = pytest.mark.skip(reason="Requires legacy CUDA codegen (compiler.cuda.implementation=legacy)")
    skip_new = pytest.mark.skip(reason="Requires experimental CUDA codegen (compiler.cuda.implementation=experimental)")

    for item in items:
        if 'old_gpu_codegen_only' in item.keywords and impl != 'legacy':
            item.add_marker(skip_old)
        if 'new_gpu_codegen_only' in item.keywords and impl != 'experimental':
            item.add_marker(skip_new)
