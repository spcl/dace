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
import glob
import os
import time

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


def test_classify_library_filename_becomes_stem():
    """A library FILENAME must be reduced to its ``-l`` stem, exactly as CMake does.

    ``ctypes.util.find_library`` (used by the OpenBLAS environment) returns names like
    ``libblas.so.3``, and the reference ScaLAPACK environments hardcode ``libscalapack-mpich.so``.
    Emitting those verbatim yields ``-llibblas.so.3``, which ld cannot resolve.
    """
    spec = nc._LinkSpec()
    for lib in ('libblas.so.3', 'libopenblas.so.0', 'liblapacke.so.3', 'libscalapack-mpich.so', 'libfoo.a'):
        nc._classify_library(spec, lib)
    assert spec.libs == ['blas', 'openblas', 'lapacke', 'scalapack-mpich', 'foo']


def test_resolve_splits_multi_token_flag_strings():
    """An environment may return several flags in ONE string (IntelMKLScaLAPACKMPICH returns its
    whole link line that way). CMake lets the shell tokenize it; native must tokenize too, or the
    string arrives as a single unusable argv element and the -l libraries never reach ld."""
    env = _make_env(cmake_link_flags=['-L /opt/mkl/lib -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_core -ldl'],
                    cmake_compile_flags=['-m64 -DMKL_ILP64'])
    spec = nc._LinkSpec()
    nc._resolve_environment(env, spec)
    assert spec.link_flags == ['-L', '/opt/mkl/lib', '-lmkl_scalapack_lp64', '-Wl,--no-as-needed', '-lmkl_core', '-ldl']
    assert spec.compile_flags == ['-m64', '-DMKL_ILP64']


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


def test_cuda_build_type_flags_are_nvcc_safe():
    """nvcc must never be handed the host's ``-Os``: it rejects it outright with
    ``nvcc fatal : 's': expected a number``, so MinSizeRel maps to -O1 for CUDA. That is why the CUDA
    build-type table is separate from the host one -- do not merge them back together."""
    assert set(nc._CUDA_BUILD_TYPE_FLAGS) == set(nc._BUILD_TYPE_FLAGS), 'every build_type needs CUDA flags'
    for build_type, flags in nc._CUDA_BUILD_TYPE_FLAGS.items():
        assert '-Os' not in flags, f'{build_type} would pass -Os to nvcc, which is fatal'
    assert nc._CUDA_BUILD_TYPE_FLAGS['MinSizeRel'] == ['-O1', '-DNDEBUG']


def test_cuda_arch_flags_normalizes_prefixed_tokens():
    """'sm_90'/'compute_80' are canonical nvcc spellings; interpolating them raw would emit the
    unbuildable 'arch=compute_sm_90,code=sm_sm_90', so the prefix must be stripped first."""
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='sm_90'):
        assert nc._cuda_arch_flags({90}) == ['-arch=native', '-gencode', 'arch=compute_90,code=sm_90']
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='compute_80'):
        assert nc._cuda_arch_flags({80}) == ['-arch=native', '-gencode', 'arch=compute_80,code=sm_80']


def test_cuda_arch_flags_ignores_native_token_in_list():
    """A 'native' entry mixed into the list is already covered by -arch=native; emitting it as a
    -gencode would produce 'arch=compute_native'."""
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='native,80'):
        assert nc._cuda_arch_flags({80}) == ['-arch=native', '-gencode', 'arch=compute_80,code=sm_80']


def test_cuda_arch_flags_skips_unparseable():
    """An entry that is not an architecture at all is warned about and skipped, never interpolated."""
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='garbage'):
        with pytest.warns(UserWarning, match='unparseable'):
            assert nc._cuda_arch_flags({80}) == ['-arch=native']


def test_cuda_arch_flags_feature_suffix():
    """An architecture token with a feature suffix (e.g. '90a') is emitted verbatim; only its numeric
    part is matched against the supported set, so int() never chokes on the letter."""
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='90a'):
        assert nc._cuda_arch_flags({90}) == ['-arch=native', '-gencode', 'arch=compute_90a,code=sm_90a']


def test_cuda_arch_flags_no_native_uses_explicit():
    """With no local GPU (allow_native=False) the configured architectures are the only targets."""
    with set_temporary('compiler', 'cuda', 'cuda_arch', value='80,90'):
        assert nc._cuda_arch_flags({80, 90}, allow_native=False) == [
            '-gencode', 'arch=compute_80,code=sm_80', '-gencode', 'arch=compute_90,code=sm_90'
        ]


def test_cuda_arch_flags_no_native_no_arch_errors():
    """No local GPU and no configured arch is unbuildable -- raise clearly instead of emitting an
    empty (and thus broken) flag list."""
    for value in ('', 'auto', 'native'):
        with set_temporary('compiler', 'cuda', 'cuda_arch', value=value):
            with pytest.raises(cgx.CompilerConfigurationError, match='cuda_arch'):
                nc._cuda_arch_flags(None, allow_native=False)


# ---------------------------------------------------------------------------
# Per-environment resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('env', LIBRARY_ENVIRONMENTS, ids=[e.__name__ for e in LIBRARY_ENVIRONMENTS])
def test_every_library_environment_resolves_or_errors_clearly(env):
    """Each shipped library environment must resolve to flags, or raise a clear
    CompilerConfigurationError -- never an unrelated exception."""
    spec = nc._LinkSpec()
    try:
        nc._resolve_environment(env, spec)
    except cgx.CompilerConfigurationError:
        return  # acceptable: e.g. MPI wrapper or ROCm not installed on this machine
    # Resolution must not have crashed with anything else, and any produced fragments are strings.
    # This says little on its own (it is vacuously true for an empty spec); the tests below pin the
    # actual resolved content per shape -- bare name, library filename, and multi-token flag string.
    for bucket in (spec.includes, spec.libs, spec.libdirs, spec.link_flags, spec.compile_flags):
        assert all(isinstance(x, str) for x in bucket)


def test_resolve_cublas_bare_name():
    """cuBLAS declares ``cmake_libraries = ['cublas']`` -- native must turn it into ``-lcublas``
    (the ``-L`` for the toolkit is added separately when a CUDA target is present)."""
    spec = nc._LinkSpec()
    nc._resolve_environment(blas_envs.cuBLAS, spec)
    assert 'cublas' in spec.libs


def test_resolve_drops_deferred_cmake_vars():
    """An environment whose flags reference ${CMAKE_VARS} (only CMake could expand them) must not
    leak the unexpanded token onto the link line."""
    env = _make_env(cmake_libraries=['${MPI_CXX_LIBRARIES}'], cmake_compile_flags=['-I${MPI_CXX_HEADER_DIR}'])
    spec = nc._LinkSpec()
    nc._resolve_environment(env, spec)
    assert not any('${' in x for x in spec.libs + spec.link_flags + spec.compile_flags + spec.includes)


def test_resolve_cmake_files_rejected():
    """An environment shipping a ``.cmake`` script cannot be honored without CMake."""
    env = _make_env(name='NeedsCMake', cmake_files=['find_thing.cmake'])
    with pytest.raises(cgx.CompilerConfigurationError, match='CMake files'):
        nc._resolve_environment(env, nc._LinkSpec())


def test_resolve_unknown_package_rejected():
    """An environment needing an unimplemented ``find_package`` must raise, not silently drop it."""
    env = _make_env(name='NeedsFindPackage', cmake_packages=['SomeExoticPackage'])
    with pytest.raises(cgx.CompilerConfigurationError, match='find_package'):
        nc._resolve_environment(env, nc._LinkSpec())


def test_resolve_mpi_missing_wrapper_errors_clearly():
    """MPI resolves through the wrapper compiler; a missing wrapper must give an actionable error."""
    with set_temporary('compiler', 'mpi', 'executable', value='/nonexistent/mpicxx-xyz'):
        with pytest.raises(cgx.CompilerConfigurationError, match='MPI'):
            nc._resolve_environment(mpi_envs.MPI, nc._LinkSpec())


# ---------------------------------------------------------------------------
# End-to-end builds (gated on toolchain / hardware)
# ---------------------------------------------------------------------------


def _blas_loadable():
    return any(ctypes.util.find_library(n) for n in ('openblas', 'blas', 'cblas', 'mkl_rt', 'mkl_rt.1'))


def test_unknown_build_mode_raises(tmp_path):
    """A typo'd compiler.build_mode must raise a clear error, not silently fall back to cmake."""

    @dace.program
    def inc(a: dace.float64[8]):
        a[:] = a + 1.0

    sdfg = inc.to_sdfg()
    sdfg.build_folder = str(tmp_path / 'cache')
    with set_temporary('compiler', 'build_mode', value='cmakee'):
        with pytest.raises(cgx.CompilerConfigurationError, match='build_mode'):
            sdfg.compile()


@pytest.mark.skipif(os.name != 'posix', reason='native build mode is Linux-only')
def test_native_rejects_hip_backend(tmp_path):
    """Native mode is CUDA-only: a HIP/ROCm backend must raise a clear error, not crash downstream
    on a half-wired code path."""
    from dace.transformation.interstate import GPUTransformSDFG

    @dace.program
    def inc(a: dace.float64[16]):
        a[:] = a + 1.0

    sdfg = inc.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG)
    sdfg.build_folder = str(tmp_path / 'cache')
    with set_temporary('compiler', 'cuda', 'backend', value='hip'):
        with set_temporary('compiler', 'build_mode', value='native'):
            with pytest.raises(cgx.CompilerConfigurationError, match='HIP'):
                sdfg.compile()


@pytest.mark.skipif(os.name != 'posix', reason='native build mode is Linux-only')
def test_cmake_build_still_works(tmp_path):
    """The default cmake backend (extracted into its own function) still builds + runs -- guards the
    refactor that split configure_and_compile into native/cmake dispatch."""

    @dace.program
    def axpy_cmake(a: dace.float64[32], b: dace.float64[32], c: dace.float64[32]):
        c[:] = 2.0 * a + b

    a = np.random.rand(32)
    b = np.random.rand(32)
    c = np.zeros(32)
    sdfg = axpy_cmake.to_sdfg()
    sdfg.build_folder = str(tmp_path / 'cache')
    with set_temporary('compiler', 'build_mode', value='cmake'):
        csdfg = sdfg.compile()
    lib = str(csdfg._lib._library_filename)
    stub = os.path.join(os.path.dirname(lib), 'libdacestub_' + os.path.basename(lib)[3:])
    assert os.path.isfile(lib) and os.path.isfile(stub)
    csdfg(a=a, b=b, c=c)
    assert np.allclose(c, 2.0 * a + b)


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
    """A real BLAS library node (matmul) builds + runs correctly under native mode.

    ``blas.default_implementation`` MUST be forced: with the stock config the matmul expands to
    ExpandGemmPure (plain generated loops, no environment at all), so the test would pass without
    ever resolving or linking a BLAS library -- i.e. providing zero coverage of the very link path
    it exists to protect.
    """
    import dace.libraries.blas as blas
    n = 48
    previous = blas.default_implementation
    blas.default_implementation = 'OpenBLAS'
    try:

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
    finally:
        blas.default_implementation = previous


@pytest.mark.skipif(os.name != 'posix', reason='native build mode is Linux-only')
def test_native_incremental_rebuild_and_invalidation(tmp_path):
    """The staleness machinery must reuse everything on an identical rebuild and rebuild when an
    input actually changes -- neither half was covered, since every other test builds exactly once.

    ``configure_and_compile`` is driven directly on one program folder: calling ``sdfg.compile()``
    twice would rename the SDFG (it is still loaded) and build a different program instead.
    """
    from dace.codegen import codegen, compiler

    @dace.program
    def incr(a: dace.float64[16]):
        a[:] = a + 1.0

    sdfg = incr.to_sdfg()
    program_folder = compiler.generate_program_folder(sdfg, codegen.generate_code(sdfg), str(tmp_path / 'prog'))
    build = os.path.join(program_folder, 'build')

    def build_products():
        """Objects + the program library. The loader stub is excluded on purpose: it is built from a
        fixed dacestub.cpp with fixed flags, so it is correctly unaffected by source/flag changes."""
        artifacts = glob.glob(os.path.join(build, '*.o')) + glob.glob(os.path.join(build, 'lib*.so'))
        return [p for p in artifacts if 'dacestub' not in os.path.basename(p)]

    with set_temporary('compiler', 'build_mode', value='native'):
        compiler.configure_and_compile(program_folder)
        products = build_products()
        assert products, 'native build produced no objects/library'
        stamps = {p: os.path.getmtime(p) for p in products}

        # (a) identical inputs -> the fast path fires: nothing is recompiled or relinked.
        compiler.configure_and_compile(program_folder)
        for product, stamp in stamps.items():
            assert os.path.getmtime(product) == stamp, f'{os.path.basename(product)} rebuilt on a no-op build'

        # (b) a changed header must invalidate the objects (touch the generated include, not a repo
        # file, so the test never mutates the source tree).
        generated_header = glob.glob(os.path.join(program_folder, 'include', '*.h'))
        assert generated_header, 'expected a generated include header'
        os.utime(generated_header[0], (time.time() + 1, time.time() + 1))
        compiler.configure_and_compile(program_folder)
        objects = [p for p in stamps if p.endswith('.o')]
        for obj in objects:
            assert os.path.getmtime(obj) > stamps[obj], f'{os.path.basename(obj)} not rebuilt after a header change'

    # (c) changed flags must invalidate too: the .cmd sidecar records the exact command.
    after_header = {p: os.path.getmtime(p) for p in products}
    with set_temporary('compiler', 'build_mode', value='native'):
        with set_temporary('compiler', 'build_type', value='Debug'):
            compiler.configure_and_compile(program_folder)
    for product, stamp in after_header.items():
        assert os.path.getmtime(product) > stamp, f'{os.path.basename(product)} not rebuilt after a flag change'


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
    test_cuda_arch_flags_feature_suffix()
    test_cuda_arch_flags_no_native_uses_explicit()
    test_cuda_arch_flags_no_native_no_arch_errors()
    for e in LIBRARY_ENVIRONMENTS:
        test_every_library_environment_resolves_or_errors_clearly(e)
    test_resolve_cublas_bare_name()
    test_resolve_drops_deferred_cmake_vars()
    test_resolve_cmake_files_rejected()
    test_resolve_unknown_package_rejected()
    test_resolve_mpi_missing_wrapper_errors_clearly()
    print('resolver unit tests passed')
