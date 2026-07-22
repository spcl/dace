# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe must find and link LAPACKE for the LAPACK-using kernels (e.g. npbench cholesky2 and
contour_integral).

DaCe's LAPACK / linalg library nodes (Cholesky -> Potrf, Solve -> Getrf+Getrs) reuse the BLAS
``OpenBLAS`` environment and emit LAPACKE calls (``LAPACKE_dpotrf``, ``LAPACKE_zgetrf`` /
``LAPACKE_zgetrs``). spack/conda ship LAPACKE inside the single ``libopenblas`` (headers
``lapacke.h`` under the install's own include dir, and the ``LAPACKE_*`` symbols in the
library) -- so once the OpenBLAS environment resolves that library + include dir, LAPACKE
links with no separate reference-LAPACK/LAPACKE package. This guards that path.
"""
import ctypes.util
import os

import numpy as np
import pytest

import dace
from dace.libraries.blas.environments import OpenBLAS


def _openmp_runtime_loadable():
    """DaCe's CMake build always links OpenMP, so any DaCe-compiled .so needs libomp/libgomp
    at load time regardless of BLAS/LAPACK (see the slurm scripts / openmp_rpath_flags)."""
    for name in ('libomp.so', 'libgomp.so.1', 'libgomp.so'):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


_SKIP_BUILD = pytest.mark.skipif(
    not OpenBLAS.is_installed() or not _openmp_runtime_loadable(),
    reason='needs an installed OpenBLAS (LAPACKE provider) and a loadable OpenMP runtime')


def test_lapacke_header_and_symbols_available():
    """When OpenBLAS is present, the LAPACKE header and symbols must be reachable -- otherwise
    a Cholesky/Solve kernel cannot compile (needs lapacke.h) or link (needs LAPACKE_*)."""
    if not OpenBLAS.is_installed():
        pytest.skip('OpenBLAS (LAPACKE provider) not installed')

    # lapacke.h must be reachable: it's in the env's declared headers, and in single-lib
    # (spack/conda) mode it lives off the default include path, so an include dir must resolve.
    assert 'lapacke.h' in OpenBLAS.headers
    if OpenBLAS._mode() == 'single':
        incs = OpenBLAS.cmake_includes()
        assert any(os.path.isfile(os.path.join(d, 'lapacke.h')) for d in incs), \
            f'lapacke.h not found under resolved include dirs {incs}'

    # The linked library must actually export the LAPACKE entry points the kernels use.
    libs = OpenBLAS.cmake_libraries()
    assert libs, 'OpenBLAS reports installed but exposes no library to link'
    abs_libs = [p for p in libs if os.path.isabs(p) and os.path.isfile(p)]
    if abs_libs:  # single-lib mode gives a concrete path; verify the symbols are really there
        loaded = ctypes.CDLL(abs_libs[0])
        for sym in ('LAPACKE_dpotrf', 'LAPACKE_zgetrf', 'LAPACKE_zgetrs'):
            assert hasattr(loaded, sym), f'{sym} missing from {abs_libs[0]} (LAPACKE not in this build)'


@_SKIP_BUILD
def test_cholesky_compiles_and_links_lapacke():
    """cholesky2 path: np.linalg.cholesky -> Potrf -> LAPACKE_?potrf, end to end."""
    @dace.program
    def chol(A: dace.float64[48, 48]):
        return np.linalg.cholesky(A)

    rng = np.random.default_rng(0)
    M = rng.random((48, 48))
    A = M @ M.T + 48 * np.eye(48)  # symmetric positive definite
    assert np.allclose(chol(A.copy()), np.linalg.cholesky(A), atol=1e-8)


@_SKIP_BUILD
def test_complex_solve_compiles_and_links_lapacke():
    """contour_integral path: complex np.linalg.solve -> Getrf+Getrs -> LAPACKE_z{getrf,getrs}."""
    @dace.program
    def solve(A: dace.complex128[32, 32], B: dace.complex128[32, 32]):
        return np.linalg.solve(A, B)

    rng = np.random.default_rng(1)
    A = (rng.random((32, 32)) + 1j * rng.random((32, 32))) + 32 * np.eye(32)
    B = rng.random((32, 32)) + 1j * rng.random((32, 32))
    assert np.allclose(solve(A.copy(), B.copy()), np.linalg.solve(A, B), atol=1e-8)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
