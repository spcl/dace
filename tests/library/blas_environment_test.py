# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe must find a BLAS implementation on a machine that has one installed.

Regression this guards: spack and conda ship OpenBLAS as a single ``libopenblas`` (no
reference-BLAS symlinks ``libblas``/``libcblas``/``liblapacke``, and headers under the
install's own ``include`` dir). The OpenBLAS environment used to look only for the
reference-BLAS libraries via ``ctypes.util.find_library``, so ``is_installed()`` was
``False`` and no BLAS lane was available even after ``spack load openblas``.
"""
import ctypes.util
import os

import numpy as np
import pytest

import dace
from dace.libraries.blas import Gemm
from dace.libraries.blas import environments as blas_envs
from dace.libraries.blas.environments import OpenBLAS

#: Every BLAS-providing environment DaCe ships.
ALL_BLAS_ENVIRONMENTS = [OpenBLAS, blas_envs.IntelMKL, blas_envs.cuBLAS, blas_envs.rocBLAS]


def _system_has_a_blas():
    """True if any BLAS shared library is loadable here (spack/conda OpenBLAS, reference
    BLAS, or MKL) -- i.e. DaCe *ought* to be able to find one."""
    return any(ctypes.util.find_library(n) for n in ('openblas', 'blas', 'cblas', 'mkl_rt', 'mkl_rt.1'))


def _openmp_runtime_loadable():
    """True if the OpenMP runtime can actually be dlopen'd. DaCe's CMake build always links
    OpenMP, so *any* DaCe-compiled .so needs it at load time; if the toolchain's libomp/
    libgomp dir isn't on LD_LIBRARY_PATH (nor rpath'd), loading fails independently of BLAS.
    Gating the end-to-end test on this keeps a BLAS check from failing for an OpenMP-runtime
    reason (see the slurm scripts / native_harness.openmp_rpath_flags that set this up)."""
    for name in ('libomp.so', 'libgomp.so.1', 'libgomp.so'):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def test_some_blas_environment_detected_when_a_blas_is_present():
    """If the machine has a BLAS at all, at least one DaCe BLAS environment must detect it."""
    if not _system_has_a_blas():
        pytest.skip('no BLAS library installed on this machine')
    detected = {env.__name__: env.is_installed() for env in ALL_BLAS_ENVIRONMENTS}
    assert any(detected.values()), (
        f'a BLAS library is loadable but no DaCe BLAS environment detected it: {detected}. '
        'DaCe cannot build any BLAS library node in this configuration.')


def test_openblas_single_library_is_detected():
    """The spack/conda single-``libopenblas`` layout must be detected (the reported bug)."""
    if not ctypes.util.find_library('openblas'):
        pytest.skip('libopenblas not loadable (run `spack load openblas` / not installed)')

    assert OpenBLAS.is_installed(), 'libopenblas is loadable but OpenBLAS.is_installed() is False'

    libs = OpenBLAS.cmake_libraries()
    assert libs, 'OpenBLAS.is_installed() is True but cmake_libraries() is empty'

    if OpenBLAS._mode() == 'single':
        # A single libopenblas -> headers live off the default include path, so an include
        # dir with cblas.h must be provided, and we must NOT require find_package(BLAS)
        # (CMake can't satisfy it for an off-path spack/conda install -> configure would fail).
        includes = OpenBLAS.cmake_includes()
        assert includes, 'single libopenblas but no include dir was resolved for cblas.h'
        assert any(os.path.isfile(os.path.join(inc, 'cblas.h')) for inc in includes), \
            f'resolved include dirs lack cblas.h: {includes}'
        assert not OpenBLAS.cmake_packages(), \
            'single libopenblas must link directly, not require find_package(BLAS)'


@pytest.mark.skipif(not OpenBLAS.is_installed(), reason='OpenBLAS not installed on this machine')
@pytest.mark.skipif(not _openmp_runtime_loadable(),
                    reason='OpenMP runtime (libomp/libgomp) not loadable -- DaCe-compiled .so '
                    'cannot be loaded; set up LD_LIBRARY_PATH/rpath for the toolchain first')
def test_gemm_compiles_and_runs_through_openblas():
    """End-to-end: a GEMM library node compiles and runs via the OpenBLAS implementation
    (finding + including + linking libopenblas) and matches numpy."""
    prev = Gemm.default_implementation
    Gemm.default_implementation = 'OpenBLAS'
    try:

        @dace.program
        def simple_gemm(A: dace.float64[32, 24], B: dace.float64[24, 16]):
            return A @ B

        A = np.random.rand(32, 24)
        B = np.random.rand(24, 16)
        result = simple_gemm(A, B)
        assert np.allclose(result, A @ B)
    finally:
        Gemm.default_implementation = prev


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
