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

import warnings

import pytest

#: ``os.fork`` as it was before the guard below replaced it.
UNGUARDED_FORK = os.fork


def thread_count() -> int:
    """Number of threads in this process, or ``0`` where the kernel does not report it."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return 0  # not Linux: no thread census, so the guard abstains


def openmp_team_was_live() -> bool:
    """True iff an OpenMP thread team was up -- and tear it down while finding out.

    No OpenMP runtime reports team liveness, so this asks the only way that works: pause every
    loaded runtime and see whether the process lost threads. Pausing is semantically transparent --
    the next parallel region rebuilds the team -- which is what makes the probe safe to run on every
    fork, and what keeps it quiet for a caller that (correctly) paused the pools itself first.

    Cheap checks first: without ``libgomp`` mapped there is no team and no reason to import dace.
    A runtime predating ``omp_pause_resource_all`` (OpenMP 5.0) cannot be torn down, so the probe
    abstains there rather than guessing.
    """
    before = thread_count()
    if before <= 1:
        return False  # single-threaded: fork carries everything it has
    try:
        with open("/proc/self/maps") as maps:
            if "libgomp" not in maps.read():
                return False  # no OpenMP runtime loaded -> no team to strand
    except OSError:
        return False
    from dace.transformation.layout.isolation import pause_openmp_pools
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a pause that could not run shows up as "no drop" below
        pause_openmp_pools()
    return thread_count() < before


def guarded_fork() -> int:
    """``os.fork`` that refuses to strand a live OpenMP team, turning a silent hang into an error.

    ``fork`` keeps only the calling thread, but the child inherits libgomp's heap-resident team
    barrier still recording the whole team -- so the child's first parallel region waits on a futex
    for threads that do not exist in it. 0 CPU, forever, and the parent blocks in ``waitpid`` behind
    it. A whole test directory then has no verdict, which is a worse failure than any assertion.

    This does not ban ``fork``. It fires only when a team is actually up, so pytest-xdist (which
    spawns its workers through ``subprocess``, never through ``os.fork``), a spawn-based child, a
    suite pinned to ``OMP_NUM_THREADS=1``, and a caller that paused the pools itself all pass
    straight through.
    """
    if openmp_team_was_live():
        raise RuntimeError("os.fork() from a process holding a live OpenMP thread team deadlocks the child: the "
                           "child inherits libgomp's team barrier recording threads that fork did not carry over, "
                           "so its first parallel region waits on them forever. The team is up because a compiled "
                           "DaCe CPU kernel has already run in this process. Run the kernel in a spawned child via "
                           "tests.helpers.isolation instead, or pin OMP_NUM_THREADS=1 before dace is imported so no "
                           "team is ever built.")
    return UNGUARDED_FORK()


os.fork = guarded_fork

#: Seed applied before every test, so a numerical comparison that only fails on some inputs fails
#: on every run instead of intermittently. Tests that want their own draw still call
#: ``np.random.seed`` / ``default_rng(seed)`` themselves; this only fixes the starting state.
GLOBAL_RANDOM_SEED = 0


@pytest.fixture(scope='session', autouse=True)
def xdist_build_folder():
    """Give each xdist worker its own build directory, so same-named SDFGs do not race.

    Prefers the CONFIG: ``Config.get`` outranks it with the env var, which would defeat every
    ``set_temporary('default_build_folder')``. An already-exported env var is moved too.
    """
    worker = os.environ.get('PYTEST_XDIST_WORKER')
    if not worker:
        return
    from dace.config import Config
    target = os.path.join(Config.get('default_build_folder'), worker)
    # An exported env var outranks the config, so it has to move too or isolation is silently lost.
    if 'DACE_default_build_folder' in os.environ:
        os.environ['DACE_default_build_folder'] = target
    Config.set('default_build_folder', value=target)


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


#: Directories owned by a dedicated workflow. Marked by path so the two suites stay selectable
#: without touching hundreds of files; general CI deselects them.
_SUITE_DIRS = (
    (os.path.join('tests', 'passes', 'vectorization'), 'vectorization'),
    (os.path.join('tests', 'passes', 'canonicalize'), 'canonicalization'),
    (os.path.join('tests', 'canonicalize'), 'canonicalization'),
)


def pytest_collection_modifyitems(config, items):
    """Mark the vectorization / canonicalization suites by path, then auto-skip tests marked
    old_gpu_codegen_only / new_gpu_codegen_only per ``compiler.cuda.implementation``."""
    for item in items:
        path = str(getattr(item, 'fspath', ''))
        for prefix, mark in _SUITE_DIRS:
            if prefix in path:
                item.add_marker(getattr(pytest.mark, mark))
                break

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
