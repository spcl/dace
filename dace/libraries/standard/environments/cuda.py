# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The GPU runtime environment, under whichever backend ``compiler.cuda.backend`` selects.

Registered as ``CUDA`` because that is the implementation key its dependents publish; what it
declares follows the backend, the way the ``dace/cuda/*`` headers do.
"""
import dace.library
from dace.codegen import common


@dace.library.environment
class CUDA:

    cmake_minimum_version = None

    @staticmethod
    def cmake_packages():
        # A HIP build resolves its runtime through ``enable_language(HIP)``; asking for the CUDA
        # package as well aborts the configure on a ROCm-only machine ("Specify
        # CUDA_TOOLKIT_ROOT_DIR"), which took every CUB-backed libnode down with it.
        return [] if common.get_gpu_backend() == 'hip' else ['CUDA']

    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    # ``cudacommon.cuh`` rather than ``cuda_runtime.h``: it is the one place the two runtimes are
    # reconciled, so it names the right backend header and is parseable by a host compiler.
    headers = {'frame': ['dace/cuda/cudacommon.cuh']}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
