# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment exposing the C++17 ``<numeric>`` family for the CPU scan.

Used by :class:`~dace.libraries.standard.nodes.scan.Scan`'s ``CPU`` and ``pure``
expansions. Frame-scope headers are emitted at file scope (not inside the tasklet
body) so ``std::inclusive_scan`` / ``std::exclusive_scan`` / ``std::partial_sum``
declarations are visible to the generated tasklet.
"""
import dace.library


@dace.library.environment
class ScanCPU:
    """C++17 ``<numeric>`` + ``<functional>`` + ``<algorithm>`` headers."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ['numeric', 'functional', 'algorithm']}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
