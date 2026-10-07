# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment exposing the bundled ``ska_sort.hpp`` header.

``ska_sort`` is a Boost-licensed single-header MSD radix sort by Malte Skarupke,
vendored at :file:`dace/runtime/include/dace/ska_sort.hpp` (see that file's header
for full attribution and license). The environment exists solely so CPU expansions
of :class:`~dace.libraries.sort.nodes.integer_sort.IntegerSort` can declare an
``#include <dace/ska_sort.hpp>`` and have the codegen pull the runtime header in.
No extra CMake packages, libraries, or flags are needed -- it is header-only.
"""

import dace.library
from typing import Dict, List


@dace.library.environment
class SkaSort:
    """Pulls in the bundled ``dace/ska_sort.hpp`` header. No external deps."""

    cmake_minimum_version = None
    cmake_packages: List[str] = []
    cmake_variables: Dict[str, str] = {}
    cmake_includes: List[str] = []
    cmake_libraries: List[str] = []
    cmake_compile_flags: List[str] = []
    cmake_link_flags: List[str] = []
    cmake_files: List[str] = []

    headers = {"frame": ["dace/ska_sort.hpp"]}
    state_fields: List[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: List[str] = []
