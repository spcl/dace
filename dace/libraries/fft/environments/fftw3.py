# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""FFTW3 environment for the :mod:`dace.libraries.fft` library nodes.

Links against the system-installed ``libfftw3`` (double precision) +
``libfftw3f`` (single precision). On Debian / Ubuntu install with::

    sudo apt-get install libfftw3-dev
"""
import ctypes.util

import dace.library


@dace.library.environment
class FFTW3:
    """CMake + link wiring for the FFTW3 backend of the FFT lib node."""

    cmake_minimum_version = "3.6"
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    # Header forwarded into the codegen unit; complex types come from
    # ``<complex.h>`` and FFTW's own ``fftw_complex`` typedef.
    headers = ["fftw3.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_libraries():
        """Resolve the FFTW3 shared libraries via :mod:`ctypes.util`.

        Returns both the double-precision (:c:expr:`fftw3`) and single-precision
        (:c:expr:`fftw3f`) variants when both are installed -- the codegen
        prefix-dispatches between them based on the operand dtype.
        """
        paths = []
        for lib in ("fftw3", "fftw3f"):
            path = ctypes.util.find_library(lib)
            if path:
                paths.append(path)
        return paths

    @staticmethod
    def is_installed():
        return len(FFTW3.cmake_libraries()) > 0
