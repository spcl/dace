# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment exposing the C++ standard headers used by CPU-side libnode expansions."""
import dace.library


@dace.library.environment
class CPU:
    """C++ standard-library headers used across the ``standard`` library's CPU expansions.

    ``cstring`` covers ``memset``/``memcpy`` (``MemsetLibraryNode`` and the parallel
    chunk-map expansions); ``numeric`` covers ``std::inclusive_scan`` /
    ``std::exclusive_scan`` / ``std::partial_sum`` (``Scan``); ``functional`` covers
    ``std::plus<>{}`` / ``std::multiplies<>{}`` etc.; ``algorithm`` covers ``std::min`` /
    ``std::max`` / ``std::copy``. All are zero-dependency, standard, and small; pulling
    them in unconditionally keeps a single CPU env across the library's tasklets instead
    of fragmenting per-node. (The parallel copy/memset OpenMP loop comes from a
    ``CPU_Multicore`` map schedule, so no ``omp.h`` is needed.)
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ['cstring', 'numeric', 'functional', 'algorithm', 'dace/scan.hpp']}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
