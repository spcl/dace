# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment exposing the C++ standard headers used by CPU-side libnode expansions."""
import dace.library


@dace.library.environment
class CPU:
    """Minimal library environment that pulls in ``<cstring>`` for plain CPU expansions."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["cstring"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
