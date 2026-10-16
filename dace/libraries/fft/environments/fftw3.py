# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FFTW3 environment: ``libfftw3`` (double) and ``libfftw3f`` (single), with their OpenMP planners when installed."""

import contextlib
import ctypes.util

import dace.library

#: Set while :func:`single_threaded_planner` is active.
_SINGLE_THREADED_PLANNER = False


@contextlib.contextmanager
def single_threaded_planner():
    """Expand FFT nodes for the single-threaded planner whatever this machine's install carries.

    For a unit built on another machine (DaCe's CPF): the threaded planner lives in a companion library
    (``libfftw3_omp``) an install may or may not have, so code that calls it links only where the rendering
    machine's choice happens to be available. ``libfftw3`` and ``libfftw3f`` alone are what every FFTW3 install
    provides.
    """
    global _SINGLE_THREADED_PLANNER
    previous, _SINGLE_THREADED_PLANNER = _SINGLE_THREADED_PLANNER, True
    try:
        yield
    finally:
        _SINGLE_THREADED_PLANNER = previous


def threaded() -> bool:
    """Whether FFTW's OpenMP planner (``fftw_plan_with_nthreads``) is linked, outside a
    :func:`single_threaded_planner`."""
    if _SINGLE_THREADED_PLANNER:
        return False
    return ctypes.util.find_library("fftw3_omp") is not None


@dace.library.environment
class FFTW3:
    cmake_minimum_version = "3.6"
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["fftw3.h", "omp.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_libraries():
        # The OpenMP planner extends the library it precedes
        paths = []
        for lib in ("fftw3_omp", "fftw3", "fftw3f_omp", "fftw3f"):
            path = ctypes.util.find_library(lib)
            if path:
                paths.append(path)
        return paths

    @staticmethod
    def is_installed():
        return ctypes.util.find_library("fftw3") is not None
