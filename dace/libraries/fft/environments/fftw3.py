# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FFTW3 environment: ``libfftw3`` (double) and ``libfftw3f`` (single), with their threaded planners when installed.

A distro install (``apt-get install libfftw3-dev``) keeps ``fftw3.h`` on the default include path and the libraries
in the ``ldconfig`` cache. A spack or module install keeps both under its own prefix, found through
:data:`FFTW_ENV_VARS` or the ``CPATH`` / ``LIBRARY_PATH`` / ``LD_LIBRARY_PATH`` search paths.
"""

import ctypes.util
import functools
import glob
import os

import dace.library

#: Environment variables that may name an FFTW install prefix (``include`` + ``lib`` under it).
FFTW_ENV_VARS = ("FFTW_ROOT", "FFTW_DIR", "FFTW3_DIR", "FFTW_HOME")

#: The two precisions the expansions call into, ``fftw_*`` and ``fftwf_*``.
FFTW_LIBRARIES = ("fftw3", "fftw3f")
#: The companion libraries that add the threaded planner, OpenMP's first so it shares the program's team.
THREADED_SUFFIXES = ("_omp", "_threads")

#: Where a distro install keeps ``fftw3.h``, on the compiler's default include path.
SYSTEM_INCLUDE_DIRS = ("/usr/include", "/usr/local/include")


def prefix_libraries(prefix: str) -> list[str]:
    """Full paths of :data:`FFTW_LIBRARIES` under ``prefix``, each threaded planner ahead of the library it extends;
    ``[]`` without the header or the double-precision library."""
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
        threaded_lib = next(filter(None, (shared(f"{lib}{suffix}") for suffix in THREADED_SUFFIXES)), None)
        found += [threaded_lib, base] if threaded_lib else [base]
    return found


def search_path_prefixes() -> list[str]:
    """Candidate install prefixes: the explicit env vars, then the parents of every search-path entry."""
    candidates = [os.environ[var] for var in FFTW_ENV_VARS if os.environ.get(var)]
    for var in ("CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"):
        candidates += [
            os.path.dirname(entry.rstrip("/")) for entry in os.environ.get(var, "").split(os.pathsep) if entry
        ]
    return candidates


@functools.lru_cache(maxsize=1)
def fftw_install() -> tuple[str | None, tuple[str, ...]]:
    """``(include_dir, libraries)`` of the FFTW3 the build uses, ``(None, ())`` when none is usable.

    A prefix carrying both ``include/fftw3.h`` and ``libfftw3`` wins, so the header and the library come from one
    build. A distro install is the fallback: its header is on the default include path and the loader resolves the
    libraries.
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
    return None, ()


def threaded(double: bool) -> bool:
    """Whether the build links the threaded planner of FFTW's double (``fftw3_omp`` / ``fftw3_threads``) or single
    (``fftw3f_omp`` / ``fftw3f_threads``) precision."""
    base = "fftw3" if double else "fftw3f"
    return any(
        os.path.basename(lib).startswith(f"lib{base}{suffix}.")
        for suffix in THREADED_SUFFIXES
        for lib in fftw_install()[1]
    )


@dace.library.environment
class FFTW3:
    cmake_minimum_version = "3.6"
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

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
        """Both precisions as far as the install provides them; the codegen picks one by the operand type."""
        return list(fftw_install()[1])

    @staticmethod
    def is_installed():
        return any(os.path.basename(lib).startswith("libfftw3.") for lib in fftw_install()[1])
