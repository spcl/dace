# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FFTW3 environment for the :mod:`dace.libraries.fft` library nodes.

Links ``libfftw3`` (double precision) and ``libfftw3f`` (single precision). A distro install
(``apt-get install libfftw3-dev``) keeps ``fftw3.h`` on the default include path and the libraries in
the ``ldconfig`` cache. A spack or module install keeps both under its own prefix, found through
:data:`FFTW_ENV_VARS`, the ``CPATH`` / ``LIBRARY_PATH`` / ``LD_LIBRARY_PATH`` search paths, or spack.
"""

import contextlib
import ctypes.util
import functools
import glob
import os

import dace.library
from dace.libraries.blas.environments.openblas import spack_install_prefix

#: Environment variables that may name an FFTW install prefix (``include`` + ``lib`` under it).
FFTW_ENV_VARS = ("FFTW_ROOT", "FFTW_DIR", "FFTW3_DIR", "FFTW_HOME")

#: The two precisions the expansions call into, ``fftw_*`` and ``fftwf_*``.
FFTW_LIBRARIES = ("fftw3", "fftw3f")
#: The companion libraries that add the threaded planner, OpenMP's first so it shares the program's team.
THREADED_SUFFIXES = ("_omp", "_threads")

#: Where a distro install keeps ``fftw3.h``, on the compiler's default include path.
SYSTEM_INCLUDE_DIRS = ("/usr/include", "/usr/local/include")


def prefix_libraries(prefix: str) -> list[str]:
    """Full paths of :data:`FFTW_LIBRARIES` under ``prefix``; ``[]`` without its header or its double library."""
    if not os.path.isfile(os.path.join(prefix, "include", "fftw3.h")):
        return []

    def shared(lib: str) -> str | None:
        hits = sorted(
            glob.glob(os.path.join(prefix, "lib", f"lib{lib}.so*"))
            + glob.glob(os.path.join(prefix, "lib64", f"lib{lib}.so*"))
        )
        return hits[0] if hits else None

    found = []
    for lib in FFTW_LIBRARIES:
        base = shared(lib)
        if base is None:
            if lib == "fftw3":
                return []
            continue
        # The threaded planner lives in a companion library, linked ahead of the one it extends.
        threaded = next(filter(None, (shared(f"{lib}{suffix}") for suffix in THREADED_SUFFIXES)), None)
        found += [threaded, base] if threaded else [base]
    return found


#: Set while :func:`single_threaded_planner` is active.
_SINGLE_THREADED_PLANNER = False


@contextlib.contextmanager
def single_threaded_planner():
    """Expand FFT nodes for the single-threaded planner whatever this machine's install carries.

    For a unit built on another machine (DaCe's CPF): the threaded planner lives in a companion library
    (``libfftw3_omp`` or ``libfftw3_threads``) an install may or may not have, so code that calls it links only
    where the rendering machine's choice happens to be available. ``libfftw3`` and ``libfftw3f`` alone are what
    every FFTW3 install provides.
    """
    global _SINGLE_THREADED_PLANNER
    previous, _SINGLE_THREADED_PLANNER = _SINGLE_THREADED_PLANNER, True
    try:
        yield
    finally:
        _SINGLE_THREADED_PLANNER = previous


def threaded() -> bool:
    """Whether the FFTW3 the build uses carries its threaded planner (``fftw_plan_with_nthreads``), outside a
    :func:`single_threaded_planner`."""
    if _SINGLE_THREADED_PLANNER:
        return False
    return any(
        any(f"fftw3{suffix}" in os.path.basename(lib) for suffix in THREADED_SUFFIXES) for lib in fftw_install()[1]
    )


def search_path_prefixes() -> list[str]:
    """Candidate install prefixes: the explicit env vars, then the parents of every search-path entry."""
    candidates = [os.environ[var] for var in FFTW_ENV_VARS if os.environ.get(var)]
    for var in ("CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"):
        candidates += [
            os.path.dirname(entry.rstrip("/")) for entry in os.environ.get(var, "").split(os.pathsep) if entry
        ]
    return candidates


@functools.lru_cache(maxsize=1, typed=True)
def fftw_install() -> tuple[str | None, tuple[str, ...]]:
    """``(include_dir, libraries)`` of the FFTW3 the build uses, ``(None, ())`` when none is usable.

    An install prefix carrying both ``include/fftw3.h`` and ``libfftw3`` wins, so the header and the
    library come from one build. A distro install is the fallback: its header sits on the default
    include path (no include dir to pass) and the loader resolves the libraries. A library without
    its header is not an install -- the build could not compile against it.
    """
    for prefix in search_path_prefixes():
        libraries = prefix_libraries(prefix)
        if libraries:
            return os.path.join(prefix, "include"), tuple(libraries)
    if any(os.path.isfile(os.path.join(d, "fftw3.h")) for d in SYSTEM_INCLUDE_DIRS):
        libraries = tuple(
            path
            for lib in FFTW_LIBRARIES
            for path in (
                next(filter(None, (ctypes.util.find_library(lib + suffix) for suffix in THREADED_SUFFIXES)), None),
                ctypes.util.find_library(lib),
            )
            if path
        )
        if libraries:
            return None, libraries
    prefix = spack_install_prefix("fftw")
    libraries = prefix_libraries(prefix) if prefix else []
    if libraries:
        return os.path.join(prefix, "include"), tuple(libraries)
    return None, ()


@dace.library.environment
class FFTW3:
    """CMake + link wiring for the FFTW3 backend of the FFT lib node."""

    cmake_minimum_version = "3.6"
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    # Header forwarded into the codegen unit; complex types come from
    # ``<complex.h>`` and FFTW's own ``fftw_complex`` typedef.
    headers = ["fftw3.h", "omp.h"]
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: list[str] = []

    @staticmethod
    def cmake_includes():
        include, _ = fftw_install()
        return [include] if include else []

    @staticmethod
    def cmake_libraries():
        """Both precisions, ``libfftw3`` and ``libfftw3f``, as far as the install provides them.

        The codegen prefix-dispatches between them on the operand dtype.
        """
        return list(fftw_install()[1])

    @staticmethod
    def is_installed():
        return len(FFTW3.cmake_libraries()) > 0
