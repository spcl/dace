# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment exposing the C++ standard headers used by CPU-side libnode expansions."""
import dace.library


@dace.library.environment
class CPU:
    """Standard headers the plain CPU expansions call into: ``<cstring>`` for ``memcpy``/``memset``
    and ``<algorithm>`` for ``std::fill_n``. Both are named here rather than left to arrive
    transitively through another header, which is a libstdc++ accident that libc++ does not repeat.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["cstring", "algorithm"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
