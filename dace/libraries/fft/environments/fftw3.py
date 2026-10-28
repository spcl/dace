# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FFTW3 environment: ``libfftw3`` (double) and ``libfftw3f`` (single), each with its OpenMP planner when installed.

The libraries come from ``$FFTW_ROOT`` (``include`` and ``lib`` / ``lib64`` under it) when set, else from the loader,
as for OpenBLAS.
"""

import ctypes.util
import functools
import glob
import os

import dace.library

#: Each precision's OpenMP planner links ahead of its base library, so it shares the program's team.
LIBRARIES = ("fftw3_omp", "fftw3", "fftw3f_omp", "fftw3f")


@functools.lru_cache(maxsize=None, typed=True)
def installed(root: str | None) -> tuple[str, ...]:
    """The :data:`LIBRARIES` found under ``root``, or by the loader for ``None``."""

    def find(lib: str) -> str | None:
        if root is None:
            return ctypes.util.find_library(lib)
        hits = sorted(glob.glob(os.path.join(root, "lib", f"lib{lib}.so*")))
        hits += sorted(glob.glob(os.path.join(root, "lib64", f"lib{lib}.so*")))
        return hits[0] if hits else None

    return tuple(path for path in map(find, LIBRARIES) if path)


def libraries() -> tuple[str, ...]:
    return installed(os.environ.get("FFTW_ROOT") or None)


def threaded(double: bool) -> bool:
    """Whether the build links FFTW's OpenMP planner of double (``fftw3_omp``) or single (``fftw3f_omp``) precision."""
    name = "libfftw3_omp." if double else "libfftw3f_omp."
    return any(os.path.basename(lib).startswith(name) for lib in libraries())


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
        root = os.environ.get("FFTW_ROOT")
        return [os.path.join(root, "include")] if root else []

    @staticmethod
    def cmake_libraries():
        """Both precisions as far as the install provides them; the codegen picks one by the operand type."""
        return list(libraries())

    @staticmethod
    def is_installed():
        return any(os.path.basename(lib).startswith("libfftw3.") for lib in libraries())
