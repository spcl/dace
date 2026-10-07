# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment for the C++17 parallel algorithms (``std::execution::par_unseq``).

libstdc++ runs them on TBB when TBB is installed, and then the program must link it; without TBB it
falls back to a serial backend that needs no library.
"""

import ctypes.util

import dace.library
from typing import Dict, List


@dace.library.environment
class ParallelSTL:
    """Links ``tbb`` when the host has it, for libstdc++'s parallel algorithm backend."""

    cmake_minimum_version = None
    cmake_packages: List[str] = []
    cmake_variables: Dict[str, str] = {}
    cmake_includes: List[str] = []
    cmake_libraries = ["tbb"] if ctypes.util.find_library("tbb") else []
    cmake_compile_flags: List[str] = []
    cmake_link_flags: List[str] = []
    cmake_files: List[str] = []

    headers = {"frame": ["algorithm", "execution"]}
    state_fields: List[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: List[str] = []
