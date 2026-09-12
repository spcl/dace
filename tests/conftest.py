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

# Run parallel kernels on MORE THAN TWO threads unless the caller pinned a count. A race in a
# generated kernel -- a dropped WCR, a parallelised reducing axis -- is invisible at one thread and
# often still invisible at two, so a suite that leaves the thread count to the machine reports a
# miscompile as a pass on any host that happens to run it serially. Four is the floor that made
# polybench ``symm``'s dropped reduction reproducible; a smaller machine gets what it has, and a
# test that needs a deterministic reduction order pins the variable itself.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(4, os.cpu_count() or 1))))

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


#: Substrings identifying an OpenMP runtime in ``/proc/self/maps``; keep in step with
#: ``dace.transformation.layout.isolation.OMP_RUNTIME_SONAMES``.
OMP_RUNTIME_MARKERS = ("libgomp", "libomp", "libiomp", "libnvomp")


def openmp_pool_may_outlive_fork() -> bool:
    """True iff an OpenMP thread pool could still be standing after asking every loaded runtime to
    tear its pool down -- i.e. iff the coming fork can still strand a team.

    Team liveness is NOT observable: no OpenMP runtime reports it. Counting the process's threads
    across the teardown looks like it answers the question and does not -- it cannot tell an
    OpenMP worker exiting from any other thread that happened to exit in the same window, and that
    window is wide because tearing a team down joins every worker in it. pytest-timeout's per-test
    timer thread lands in exactly that window, which is how this fired on forks that ``run_isolated``
    had already made safe. What IS observable is whether the teardown ran, and that is the property
    fork safety actually rests on: a runtime that reports success has no pool left to strand.

    Cheap checks first: a single-threaded process has nothing to strand, and without an OpenMP
    runtime mapped there is no pool and no reason to import dace.
    """
    if thread_count() <= 1:
        return False  # single-threaded: fork carries everything it has
    try:
        with open("/proc/self/maps") as maps:
            mapped = maps.read()
    except OSError:
        return False  # not Linux: no census, so the guard abstains
    if not any(marker in mapped for marker in OMP_RUNTIME_MARKERS):
        return False  # no OpenMP runtime loaded -> no pool to strand
    from dace.transformation.layout.isolation import pause_openmp_pools
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the return value carries the same verdict as the warning
        return not pause_openmp_pools()


def guarded_fork() -> int:
    """``os.fork`` that tears OpenMP pools down first and refuses the fork when it cannot, turning a
    silent hang into an error.

    ``fork`` keeps only the calling thread, but the child inherits libgomp's heap-resident team
    barrier still recording the whole team -- so the child's first parallel region waits on a futex
    for threads that do not exist in it. 0 CPU, forever, and the parent blocks in ``waitpid`` behind
    it. A whole test directory then has no verdict, which is a worse failure than any assertion.

    This does not ban ``fork``. Pausing the pools is semantically transparent -- the next parallel
    region rebuilds the team -- so the common case is made safe rather than rejected, and the error
    is reserved for a runtime that could not be paused at all (pre-OpenMP-5.0, or a fork from inside
    a parallel region), where the deadlock is real and no teardown can prevent it.
    """
    if openmp_pool_may_outlive_fork():
        raise RuntimeError("os.fork() from a process whose OpenMP thread pool could not be torn down deadlocks the "
                           "child: the child inherits libgomp's team barrier recording threads that fork did not "
                           "carry over, so its first parallel region waits on them forever. omp_pause_resource_all "
                           "refused or is missing -- a pre-OpenMP-5.0 runtime, or a fork from inside a parallel "
                           "region. Run the kernel in a spawned child via tests.helpers.isolation instead, or pin "
                           "OMP_NUM_THREADS=1 before dace is imported so no team is ever built.")
    return UNGUARDED_FORK()


os.fork = guarded_fork

#: Seed applied before every test, so a numerical comparison that only fails on some inputs fails
#: on every run instead of intermittently. Tests that want their own draw still call
#: ``np.random.seed`` / ``default_rng(seed)`` themselves; this only fixes the starting state.
GLOBAL_RANDOM_SEED = 0


@pytest.fixture(scope='session', autouse=True)
def xdist_build_folder():
    """Give each xdist worker its own build directory, so same-named SDFGs do not race.

    DaCe keys a build on the SDFG NAME and many tests reuse generic ones ("testing", "tester"), so two
    workers compiling same-named SDFGs race on one build entry and load a half-written .so. Sets the
    CONFIG rather than exporting ``DACE_default_build_folder``: ``Config.get`` returns an env var
    before it consults the config, so exporting it would defeat every
    ``set_temporary('default_build_folder')`` for the whole session. An already-exported env var is
    moved too, or the config write it outranks buys no isolation. Serial runs are untouched.
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


#: Directories owned by a dedicated workflow. Marked by path so the suites stay selectable
#: without touching hundreds of files; general CI deselects them.
_SUITE_DIRS = (
    (os.path.join('tests', 'passes', 'vectorization'), 'vectorization'),
    (os.path.join('tests', 'passes', 'canonicalize'), 'canonicalization'),
    (os.path.join('tests', 'canonicalize'), 'canonicalization'),
    # 892 combinatorial sweeps over layout permutations. Marked so the general selection skips them
    # and the dedicated `layout` job is the only thing that runs them.
    (os.path.join('tests', 'transformations', 'layout'), 'layout'),
    # Needs a real MPI launcher (the heterogeneous runner runs these under ``mpirun -n 2``), so mark
    # the directory rather than rely on each file remembering the marker -- three did not.
    (os.path.join('tests', 'library', 'mpi'), 'mpi'),
    # Before the plain-corpus entry: first match wins, and CloudSC is a whole-application
    # integration run (offload, pipeline, coalesce, checkpoint-resume), not a kernel corpus.
    (os.path.join('tests', 'corpus', 'cloudsc'), 'integration'),
    (os.path.join('tests', 'corpus'), 'corpus'),
    (os.path.join('tests', 'npbench'), 'corpus'),
)

#: Library-node and loop-to-libnode lift tests (loop2sym / loop2reduce / loop2scan / lift-einsum
#: and the library nodes this branch adds). One home: the "LibNodes + LoopToLibNodeLifts" CI job.
#: The file list beats the suite-directory marking below, so a lift test under a suite directory
#: still lands here. MPI library tests are NOT here -- ``tests/library/mpi`` is directory-marked
#: ``mpi`` and runs under the heterogeneous runner's ``mpirun``.
_LIBNODE_FILES = (
    'allany_node_test.py',
    'arg_reduce_test.py',
    'argminmax_test.py',
    'blas_environment_test.py',
    'broadcast_test.py',
    'copy_node_test.py',
    'count_node_test.py',
    'cshift_test.py',
    'fft_axis_test.py',
    'fft_fftw3_test.py',
    'fft_interpolate_test.py',
    'fft_pure_ndim_test.py',
    'fill_node_test.py',
    'fortran_io_test.py',
    'gemm_runtime_coeff_test.py',
    'integer_sort_test.py',
    'lapacke_link_test.py',
    'lift_einsum_matmul_test.py',
    'loop_to_reduce_test.py',
    'loop_to_scan_test.py',
    'loop_to_symm_test.py',
    'loop_to_symmetrize_test.py',
    'matmul_batched_test.py',
    'matmul_broadcast_test.py',
    'matmul_test.py',
    'matmul_trans_test.py',
    'matmul_unit_dim_squeeze_test.py',
    'merge_node_test.py',
    'norm2_test.py',
    'preexpanded_libnode_stream_test.py',
    'scan_strided_test.py',
    'scan_test.py',
    'strengthen2_loop_to_scan_test.py',
    'symm_test.py',
    'syrk_test.py',
    'tensordot_tblis_test.py',
    'test_reduce_view_gpu.py',
)

#: Corpus files living outside those directories. A corpus case compiles and RUNS a C++ program per
#: kernel, so wall time tracks the kernel count -- general CI carried ~880 of them, which is why it
#: takes as long as it does. Matched on the file name, since these sit among ordinary unit tests.
_CORPUS_FILES = (
    'tsvc_integration_test.py',
    'parallelize_peeling_tsvc_test.py',
    'native_corpus_test.py',
    'test_corpus_compile.py',
    'test_corpus_equivalence.py',
)


def pytest_collection_modifyitems(config, items):
    """Mark the vectorization / canonicalization suites by path, then auto-skip tests marked
    old_gpu_codegen_only / new_gpu_codegen_only per ``compiler.cuda.implementation``."""
    for item in items:
        path = str(getattr(item, 'fspath', ''))
        name = os.path.basename(path)
        # File list beats directory: a loop-to-libnode lift test living under a suite directory
        # (loop_to_symm* under tests/passes/canonicalize) belongs to the libnode runner, not the
        # suite its directory would mark it into.
        if name in _LIBNODE_FILES or os.path.join('tests', 'library', 'tileops') in path:
            item.add_marker(pytest.mark.loop2x_libnodes)
            continue
        for prefix, mark in _SUITE_DIRS:
            if prefix in path:
                item.add_marker(getattr(pytest.mark, mark))
                break
        else:
            if name in _CORPUS_FILES:
                item.add_marker(pytest.mark.corpus)

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
