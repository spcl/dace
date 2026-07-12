# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util
import os

import dace.library


def _resolve_lib_dir(soname_or_path):
    """Directory that actually contains a library, given whatever
    ``ctypes.util.find_library`` returned -- which on Linux is often a bare soname
    (``libopenblas.so.0``) rather than an absolute path. Resolved against
    ``LD_LIBRARY_PATH`` in that case (e.g. a ``spack load``/``module load`` install)."""
    if not soname_or_path:
        return None
    if os.path.isabs(soname_or_path) and os.path.exists(soname_or_path):
        return os.path.dirname(os.path.realpath(soname_or_path))
    for d in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep):
        cand = os.path.join(d, soname_or_path) if d else ''
        if cand and os.path.exists(cand):
            return os.path.dirname(os.path.realpath(cand))
    return None


def _single_libopenblas():
    """``(lib_path, include_dir)`` for a single-library OpenBLAS install, or ``(None, None)``.

    spack and conda ship OpenBLAS as one ``libopenblas`` that provides BLAS, CBLAS, LAPACK
    and LAPACKE together -- with no reference-BLAS symlinks (``libblas``/``libcblas``/
    ``liblapacke``) and with the headers (``cblas.h``/``lapacke.h``) under the install's own
    ``include`` dir, off the default compiler search path. This locates that library and its
    header dir so both can be passed to the build directly."""
    for var in ('OPENBLAS_ROOT', 'OPENBLAS_HOME', 'OpenBLAS_HOME'):
        root = os.environ.get(var)
        if root and os.path.isdir(root):
            for libdir in ('lib', 'lib64'):
                cand = os.path.join(root, libdir, 'libopenblas.so')
                if os.path.exists(cand):
                    inc = os.path.join(root, 'include')
                    return cand, (inc if os.path.isfile(os.path.join(inc, 'cblas.h')) else None)
    found = ctypes.util.find_library('openblas')
    libdir = _resolve_lib_dir(found)
    if libdir:
        cand = os.path.join(libdir, 'libopenblas.so')
        lib = cand if os.path.exists(cand) else found  # else let the linker resolve the soname
        inc = os.path.join(os.path.dirname(libdir), 'include')
        return lib, (inc if os.path.isfile(os.path.join(inc, 'cblas.h')) else None)
    return None, None


def _reference_blas_libs():
    """Reference/update-alternatives BLAS shared libs (``liblapacke``/``libcblas``/
    ``libblas``) that are on the loader path, or ``[]``."""
    return [p for p in (ctypes.util.find_library(l) for l in ('lapacke', 'cblas', 'blas')) if p]


@dace.library.environment
class OpenBLAS:

    # Works both with a reference/update-alternatives BLAS (liblapacke/libcblas/libblas all
    # pointing at OpenBLAS) and with a single-library OpenBLAS as shipped by spack/conda
    # (libopenblas only, headers under its own include dir).

    cmake_minimum_version = "3.6"
    cmake_compile_flags = []
    cmake_files = []

    headers = ["cblas.h", "lapacke.h", "../include/dace_blas.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def _mode():
        """``'reference'`` if reference-BLAS libs are on the loader path (let CMake's
        FindBLAS locate them); else ``'single'`` if a lone ``libopenblas`` is (link it
        directly); else ``None`` (not installed)."""
        if _reference_blas_libs():
            return 'reference'
        lib, _ = _single_libopenblas()
        return 'single' if lib else None

    @staticmethod
    def cmake_packages():
        # A single libopenblas is passed as a full library path below, so requiring
        # find_package(BLAS REQUIRED) -- which CMake can't satisfy for an off-default-path
        # spack/conda install -- would only make configure fail. Only the reference path
        # relies on FindBLAS/FindLAPACK.
        return ["LAPACK", "BLAS"] if OpenBLAS._mode() == 'reference' else []

    @staticmethod
    def cmake_variables():
        return {"BLA_VENDOR": "OpenBLAS"} if OpenBLAS._mode() == 'reference' else {}

    @staticmethod
    def cmake_link_flags():
        # These vars are only defined once find_package(LAPACK/BLAS) has run (reference path).
        return ["${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}"] if OpenBLAS._mode() == 'reference' else []

    @staticmethod
    def cmake_includes():
        _, inc = _single_libopenblas()
        return [inc] if inc else []

    @staticmethod
    def cmake_libraries():
        ref = _reference_blas_libs()
        if ref:
            return ref
        lib, _ = _single_libopenblas()
        return [lib] if lib else []

    @staticmethod
    def is_installed():
        return len(OpenBLAS.cmake_libraries()) > 0
