"""Pytest fixtures + CLI flags for ``tests/ab_perf``.

These tests are EXPENSIVE (they compile multiple SDFG variants and time
them under realistic problem sizes). They are skipped by default; pass
``--ab-perf`` to opt in.

CLI options:

* ``--ab-perf`` -- enable the A/B performance tests (default: skipped).
* ``--ab-iters N`` -- number of timed iterations per variant (default 5).
* ``--ab-warmup K`` -- warmup iterations before timing (default 1).
* ``--no-gpu`` -- skip GPU variants even if a CUDA device + cupy are
  available (default: GPU variants run when feasible).
* ``--ab-klev N`` -- vertical levels for cloudsc-shape kernels (default 96).
* ``--ab-klon N`` -- horizontal columns (default 20480; must be a multiple
  of 32 to mirror cloudsc).
* ``--ab-threads N`` -- OpenMP threads for every timed variant (default 4).
"""
import os

import pytest

#: Threads every timed variant runs on unless ``--ab-threads`` says otherwise. These A/Bs compare a
#: PARALLEL lowering against a sequential one -- ``test_break_parallelization_ab`` states its wall
#: time tracks ``k / threads`` -- so at one thread the comparison measures nothing and reports a
#: working parallelisation as no gain. Four is the same floor ``tests/conftest.py`` picked for
#: making races visible. The count is PINNED rather than inherited: a perf number that depends on
#: whatever the caller exported is not comparable across two runs, let alone two machines.
DEFAULT_AB_THREADS = 4


def pytest_addoption(parser):
    parser.addoption('--ab-perf',
                     action='store_true',
                     default=False,
                     help='Enable A/B performance tests (skipped by default).')
    parser.addoption('--ab-iters',
                     action='store',
                     type=int,
                     default=5,
                     help='Number of timed iterations per variant (default 5).')
    parser.addoption('--ab-warmup',
                     action='store',
                     type=int,
                     default=1,
                     help='Warmup iterations before timing (default 1).')
    parser.addoption('--no-gpu',
                     action='store_true',
                     default=False,
                     help='Skip GPU variants even if CUDA + cupy are available.')
    parser.addoption('--ab-klev',
                     action='store',
                     type=int,
                     default=96,
                     help='Vertical levels for cloudsc-shape kernels (default 96).')
    parser.addoption('--ab-klon',
                     action='store',
                     type=int,
                     default=20480,
                     help='Horizontal columns (default 20480, must be a multiple of 32).')
    parser.addoption('--ab-threads',
                     action='store',
                     type=int,
                     default=DEFAULT_AB_THREADS,
                     help=f'OpenMP threads per timed variant (default {DEFAULT_AB_THREADS}).')


def pytest_configure(config):
    """Pin the thread count before any kernel is built or run.

    This OVERRIDES an inherited value on purpose. ``tests/conftest.py`` can only ``setdefault``,
    so a caller exporting ``OMP_NUM_THREADS=1`` -- the DaCe MPI-prefix convention -- silently
    serialises every A/B here. The BLAS counts move together so a library call cannot spill into
    a different width than the kernel it is being compared against.
    """
    threads = str(config.getoption('--ab-threads'))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = threads


def pytest_collection_modifyitems(config, items):
    """Skip every ab_perf test unless ``--ab-perf`` is on."""
    if config.getoption('--ab-perf'):
        return
    skip_marker = pytest.mark.skip(reason='ab_perf tests are off by default; pass --ab-perf to enable.')
    for item in items:
        if 'ab_perf' in str(item.fspath):
            item.add_marker(skip_marker)


@pytest.fixture
def ab_iters(request):
    return request.config.getoption('--ab-iters')


@pytest.fixture
def ab_warmup(request):
    return request.config.getoption('--ab-warmup')


@pytest.fixture
def ab_klev(request):
    return request.config.getoption('--ab-klev')


@pytest.fixture
def ab_klon(request):
    return request.config.getoption('--ab-klon')


@pytest.fixture
def ab_threads(request):
    return request.config.getoption('--ab-threads')


@pytest.fixture
def ab_gpu_enabled(request):
    """``True`` iff the user did not pass ``--no-gpu`` AND a CUDA device +
    ``cupy`` are importable. Tests that wrap a GPU variant should consult
    this fixture and ``pytest.skip(...)`` the GPU half when ``False``."""
    if request.config.getoption('--no-gpu'):
        return False
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False
