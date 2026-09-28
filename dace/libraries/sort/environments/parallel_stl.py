# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment for the C++17 parallel algorithms (``std::execution::par_unseq``).

libstdc++ runs them on TBB when TBB is installed, and then the program must link it; without TBB it
falls back to a serial backend that needs no library.
"""
import ctypes.util

import dace.library


@dace.library.environment
class ParallelSTL:
    """Links ``tbb`` when the host has it, for libstdc++'s parallel algorithm backend."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ['tbb'] if ctypes.util.find_library('tbb') else []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ['algorithm', 'execution']}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
