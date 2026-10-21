# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FFTW3 environment: ``libfftw3`` (double) and ``libfftw3f`` (single), with their OpenMP planners when installed."""

import ctypes.util

import dace.library


def threaded(double: bool) -> bool:
    """Whether the OpenMP planner of FFTW's double (``fftw3_omp``) or single (``fftw3f_omp``) precision is linked."""
    return ctypes.util.find_library("fftw3_omp" if double else "fftw3f_omp") is not None


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
