# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Native (no-CMake) build mode: library-node support.

The native builder resolves the linker/compiler flags for a program from the environments its
library nodes pull in, without running CMake ``find_package``. These tests pin that resolution for
every library environment DaCe ships: each one must either resolve to concrete flags or raise a
clear :class:`CompilerConfigurationError` telling the user to fall back to cmake -- never crash with
an unexpected error. A handful of gated end-to-end tests then actually build + run a library-node
program under ``compiler.build_mode = native``.
"""
import ctypes.util
import os

import numpy as np
import pytest

import dace
from dace.config import set_temporary
from dace.codegen import exceptions as cgx
from dace.codegen import native_compiler as nc
from dace.libraries.blas import environments as blas_envs
from dace.libraries.mpi import environments as mpi_envs
from dace.libraries.standard import environments as std_envs

#: Every library environment exercised here. Each ships as a ``@dace.library.environment`` class
#: carrying the ``cmake_*`` attributes the native resolver consumes.
LIBRARY_ENVIRONMENTS = [
    blas_envs.cuBLAS,
    blas_envs.OpenBLAS,
    blas_envs.IntelMKL,
    blas_envs.rocBLAS,
    mpi_envs.MPI,
    std_envs.CUDA,
]


def _make_env(name='FakeEnv', **overrides):
    """Minimal stand-in for a ``@dace.library.environment`` class, for negative-path tests."""
    attrs = dict(cmake_packages=[],
                 cmake_variables={},
                 cmake_includes=[],
                 cmake_libraries=[],
                 cmake_compile_flags=[],
                 cmake_link_flags=[],
                 cmake_files=[],
                 headers=[],
                 _dace_file_path=os.path.abspath(__file__))
    attrs.update(overrides)
    return type(name, (), attrs)


# ---------------------------------------------------------------------------
# Flag classification / helpers
# ---------------------------------------------------------------------------


def test_classify_library():
    spec = nc._LinkSpec()
    nc._classify_library(spec, 'cublas')  # bare soname -> -l
    nc._classify_library(spec, '/opt/mkl/lib/libmkl_rt.so')  # absolute path -> verbatim
    nc._classify_library(spec, '-lfoo')  # already a flag -> verbatim
    nc._classify_library(spec, '-L/x')
    nc._classify_library(spec, '   ')  # empty -> ignored
    assert spec.libs == ['cublas']
    assert '/opt/mkl/lib/libmkl_rt.so' in spec.link_flags
    assert '-lfoo' in spec.link_flags and '-L/x' in spec.link_flags


def test_is_deferred():
    assert nc._is_deferred('${MPI_CXX_LIBRARIES}')
    assert nc._is_deferred('-I${MPI_CXX_HEADER_DIR}')
    assert not nc._is_deferred('-lcublas')
    assert not nc._is_deferred('/usr/lib/libfoo.so')


def test_cuda_arch_flags_auto():
    """The local GPU is always targeted via ``-arch=native``, with no extra archs for auto/empty."""
    for value in ('auto', 'native', ''):
        with set_temporary('compiler', 'cuda', 'cuda_arch', value=value):
            assert nc._cuda_arch_flags(None) == ['-arch=native']


def test_cuda_arch_flags_explicit_additional():
    """Explicit architectures are appended as additional ``-gencode`` targets on top of native."""
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='80,90'):
        assert nc._cuda_arch_flags({80, 90}) == [
            '-arch=native', '-gencode', 'arch=compute_80,code=sm_80', '-gencode', 'arch=compute_90,code=sm_90'
        ]


def test_cuda_arch_flags_skips_unsupported():
    """An architecture the toolkit dropped (e.g. sm_60 on CUDA 13) is skipped, not fatal -- so a
    stale cuda_arch default never breaks the build; the local GPU is still covered by native."""
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='60'):
        assert nc._cuda_arch_flags({80, 89}) == ['-arch=native']


# ---------------------------------------------------------------------------
# Per-environment resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('env', LIBRARY_ENVIRONMENTS, ids=[e.__name__ for e in LIBRARY_ENVIRONMENTS])
def test_every_library_environment_resolves_or_errors_clearly(env):
    """Each shipped library environment must resolve to flags, or raise a clear
    CompilerConfigurationError -- never an unrelated exception."""
    spec = nc._LinkSpec()
    try:
        nc._resolve_environment(env, spec, dict(os.environ))
    except cgx.CompilerConfigurationError:
        pass  # acceptable: e.g. MPI wrapper or ROCm not installed on this machine
    # Resolution must not have crashed with anything else, and any produced fragments are strings.
    for bucket in (spec.includes, spec.libs, spec.libdirs, spec.link_flags, spec.compile_flags):
        assert all(isinstance(x, str) for x in bucket)


def test_resolve_cublas_bare_name():
    """cuBLAS declares ``cmake_libraries = ['cublas']`` -- native must turn it into ``-lcublas``
    (the ``-L`` for the toolkit is added separately when a CUDA target is present)."""
    spec = nc._LinkSpec()
    nc._resolve_environment(blas_envs.cuBLAS, spec, dict(os.environ))
    assert 'cublas' in spec.libs


def test_resolve_drops_deferred_cmake_vars():
    """An environment whose flags reference ${CMAKE_VARS} (only CMake could expand them) must not
    leak the unexpanded token onto the link line."""
    env = _make_env(cmake_libraries=['${MPI_CXX_LIBRARIES}'], cmake_compile_flags=['-I${MPI_CXX_HEADER_DIR}'])
    spec = nc._LinkSpec()
    nc._resolve_environment(env, spec, dict(os.environ))
    assert not any('${' in x for x in spec.libs + spec.link_flags + spec.compile_flags + spec.includes)


def test_resolve_cmake_files_rejected():
    """An environment shipping a ``.cmake`` script cannot be honored without CMake."""
    env = _make_env(name='NeedsCMake', cmake_files=['find_thing.cmake'])
    with pytest.raises(cgx.CompilerConfigurationError, match='CMake files'):
        nc._resolve_environment(env, nc._LinkSpec(), dict(os.environ))


def test_resolve_unknown_package_rejected():
    """An environment needing an unimplemented ``find_package`` must raise, not silently drop it."""
    env = _make_env(name='NeedsFindPackage', cmake_packages=['SomeExoticPackage'])
    with pytest.raises(cgx.CompilerConfigurationError, match='find_package'):
        nc._resolve_environment(env, nc._LinkSpec(), dict(os.environ))


def test_resolve_mpi_missing_wrapper_errors_clearly():
    """MPI resolves through the wrapper compiler; a missing wrapper must give an actionable error."""
    with set_temporary('compiler', 'mpi', 'executable', value='/nonexistent/mpicxx-xyz'):
        with pytest.raises(cgx.CompilerConfigurationError, match='MPI'):
            nc._resolve_environment(mpi_envs.MPI, nc._LinkSpec(), dict(os.environ))


# ---------------------------------------------------------------------------
# End-to-end builds (gated on toolchain / hardware)
# ---------------------------------------------------------------------------


def _blas_loadable():
    return any(ctypes.util.find_library(n) for n in ('openblas', 'blas', 'cblas', 'mkl_rt', 'mkl_rt.1'))


@pytest.mark.skipif(os.name != 'posix', reason='native build mode is Linux-only')
def test_native_build_plain_cpu(tmp_path):
    """A plain (no-libnode) program builds + runs under native mode, producing the library and its
    loader stub where the loader expects them."""

    @dace.program
    def axpy(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        c[:] = 2.0 * a + b

    a = np.random.rand(32)
    b = np.random.rand(32)
    c = np.zeros(32)
    sdfg = axpy.to_sdfg()
    sdfg.build_folder = str(tmp_path / 'cache')
    with set_temporary('compiler', 'build_mode', value='native'):
        csdfg = sdfg.compile()
    lib = str(csdfg._lib._library_filename)
    stub = os.path.join(os.path.dirname(lib), 'libdacestub_' + os.path.basename(lib)[3:])
    assert os.path.isfile(lib) and os.path.isfile(stub)
    csdfg(a=a, b=b, c=c)
    assert np.allclose(c, 2.0 * a + b)


@pytest.mark.skipif(os.name != 'posix' or not _blas_loadable(), reason='no BLAS available')
def test_native_build_blas_matmul(tmp_path):
    """A BLAS library node (matmul) builds + runs correctly under native mode."""
    n = 48

    @dace.program
    def mm(x: dace.float64[n, n], y: dace.float64[n, n], z: dace.float64[n, n]):
        z[:] = x @ y

    x = np.random.rand(n, n)
    y = np.random.rand(n, n)
    z = np.zeros((n, n))
    sdfg = mm.to_sdfg()
    sdfg.build_folder = str(tmp_path / 'cache')
    with set_temporary('compiler', 'build_mode', value='native'):
        try:
            csdfg = sdfg.compile()
        except cgx.CompilerConfigurationError as ex:
            pytest.skip(f'native BLAS resolution unavailable here: {ex}')
    csdfg(x=x, y=y, z=z)
    assert np.allclose(z, x @ y)


@pytest.mark.gpu
def test_native_build_cublas_matmul(tmp_path):
    """A cuBLAS library node builds through the native ``.cu -> .a -> .so`` path and runs."""
    import dace.libraries.blas as blas
    from dace.transformation.interstate import GPUTransformSDFG
    n = 64
    old = blas.default_implementation
    blas.default_implementation = 'cuBLAS'
    try:

        @dace.program
        def mmg(x: dace.float64[n, n], y: dace.float64[n, n], z: dace.float64[n, n]):
            z[:] = x @ y

        sdfg = mmg.to_sdfg()
        sdfg.apply_transformations(GPUTransformSDFG)
        sdfg.build_folder = str(tmp_path / 'cache')
        x = np.random.rand(n, n)
        y = np.random.rand(n, n)
        z = np.zeros((n, n))
        with set_temporary('compiler', 'build_mode', value='native'):
            csdfg = sdfg.compile()
        csdfg(x=x, y=y, z=z)
        assert np.allclose(z, x @ y, rtol=1e-4)
    finally:
        blas.default_implementation = old


if __name__ == '__main__':
    test_classify_library()
    test_is_deferred()
    test_cuda_arch_flags_auto()
    test_cuda_arch_flags_explicit_additional()
    test_cuda_arch_flags_skips_unsupported()
    for e in LIBRARY_ENVIRONMENTS:
        test_every_library_environment_resolves_or_errors_clearly(e)
    test_resolve_cublas_bare_name()
    test_resolve_drops_deferred_cmake_vars()
    test_resolve_cmake_files_rejected()
    test_resolve_unknown_package_rejected()
    test_resolve_mpi_missing_wrapper_errors_clearly()
    print('resolver unit tests passed')
