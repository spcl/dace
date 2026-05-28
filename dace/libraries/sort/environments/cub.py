# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment exposing the CUDA CUB headers for ``DeviceRadixSort``.

``cub::DeviceRadixSort`` is part of the CUDA toolkit (under ``cub/cub.cuh``), so
no extra CMake package is needed beyond the existing CUDA setup. The environment
declares the include and inherits the standard CUDA environment for the runtime.
"""
import dace.library
from dace.libraries.standard.environments.cuda import CUDA


@dace.library.environment
class CUB:
    """Pulls in CUB headers for the CUDA expansion of :class:`IntegerSort`."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ['cub/cub.cuh']}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = [CUDA]
