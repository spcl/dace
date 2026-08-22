# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment for the strided ``Scan`` GPU expansion.

Provides ONLY the ``extern "C"`` declarations of the
``dace_cuda_strided_inclusive_<op>_<dtype>`` wrappers and the nvcc-compiled
``.cu`` translation unit that defines them. Deliberately does NOT pull in
``cub/cub.cuh`` -- that header's CCCL 3+ template bodies (CUDA Toolkit 13+)
use nvcc-only constructs that fail to compile under g++ when included in
the SDFG's host ``.cpp`` translation unit. Splitting the strided GPU path
into its own environment keeps host-side libnode tasklets cub-free.
"""
import os

import dace.library
from dace.libraries.standard.environments.cuda import CUDA

_DACE_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_STRIDED_SCAN_CU = os.path.join(_DACE_REPO_ROOT, 'runtime', 'include', 'dace', 'cuda', 'scan_strided.cu')


@dace.library.environment
class ScanStrided:
    """``extern "C"`` host-callable wrappers for the strided inclusive scan.

    The wrappers live in :file:`dace/runtime/include/dace/cuda/scan_strided.cu`
    (compiled by nvcc thanks to the ``auxiliary_sources`` extension to
    ``library.environment``); the declarations included by the host ``.cpp``
    are in :file:`dace/runtime/include/dace/cuda/scan_strided_decls.h`.

    Pulls in the ``CUDA`` env to ensure ``find_package(CUDAToolkit)`` runs
    first (the wrappers' signatures reference ``cudaStream_t``).
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ['dace/cuda/scan_strided_decls.h']}
    auxiliary_sources = [_STRIDED_SCAN_CU]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = [CUDA]
