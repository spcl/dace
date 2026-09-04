# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment for the AMD hipTensor backend."""
import ctypes.util
import os
import pathlib
import shutil

import dace.library


@dace.library.environment
class hipTensor:
    """Build/link configuration and per-node setup code for hipTensor-backed library nodes."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_libraries = ["hiptensor"]
    # The HIP headers require a platform macro when the compiler is not hipcc.
    cmake_compile_flags = ["-D__HIP_PLATFORM_AMD__"]
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["dace/dace_hiptensor.h"], 'cuda': ["dace/dace_hiptensor.h"]}
    state_fields = ["dace::linalg::HipTensorHandle hiptensor_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    #: dtype -> (tensor data type, compute descriptor, C scalar type for alpha/beta). Same shape as
    #: :attr:`~dace.libraries.linalg.environments.cutensor.cuTensor.TYPE_MAP`, but SHORTER, and the
    #: difference is measured rather than assumed. Two things bite here:
    #:
    #: * the tensor data type is hipTensor's OWN enum -- ``HIPTENSOR_R_32F``, not the HIP-wide
    #:   ``HIP_R_32F``, which the descriptor call rejects outright;
    #: * hipTensor ACCEPTS a plan for every dtype and then refuses it at execution. Measured on
    #:   ROCm 7.2.3 / gfx942, calling ``hiptensorPermute`` for real: fp32 succeeds at ranks 2, 3
    #:   and 4 and fp16 at rank 3, while fp64, complex64 and complex128 return
    #:   ``HIPTENSOR_STATUS_NOT_SUPPORTED`` at every rank tried.
    #:
    #: So the unsupported types are ABSENT here rather than listed and left to fail: the shared
    #: expansion body falls back to the pure expansion for a dtype this map does not carry, which on
    #: GPU-resident data is still a GPU map. Listing them instead aborts the process, because the
    #: status is raised as a C++ exception from inside the tasklet.
    TYPE_MAP = {
        dace.float16: ('HIPTENSOR_R_16F', 'HIPTENSOR_COMPUTE_DESC_16F', '__half'),
        dace.float32: ('HIPTENSOR_R_32F', 'HIPTENSOR_COMPUTE_DESC_32F', 'float'),
    }

    @staticmethod
    def cmake_includes():
        """The ROCm include directory, so the HOST compiler can find the HIP headers too.

        Same reason as :meth:`~dace.libraries.blas.environments.rocblas.rocBLAS.cmake_includes`: a
        tasklet reaching this environment through the ``frame`` entry is compiled by g++, which gets
        no HIP include path of its own. Resolved here rather than as a CMake variable, so no ROCm
        path is written down.
        """
        root = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
        if not root:
            driver = shutil.which("hipcc")
            root = str(pathlib.Path(driver).resolve().parent.parent) if driver else ""
        include = pathlib.Path(root) / "include" if root else None
        return [str(include)] if include and include.is_dir() else []

    @staticmethod
    def handle_setup_code(node):
        return dace.library.reject_gpu_location(node) + """\
hiptensorHandle_t &__dace_hiptensor_handle = __state->hiptensor_handle.Get();\n"""

    @staticmethod
    def is_installed():
        return ctypes.util.find_library('hiptensor') is not None
