# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
from dace.libraries.standard.environments.cuda import CUDA


@dace.library.environment
class MemoryEngine:
    """Memory engine (TMA) helpers, needed both by the host (descriptor encoding) and the device code."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["dace/cuda/memoryengine.cuh"], 'cuda': ["dace/cuda/memoryengine.cuh"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = [CUDA]
