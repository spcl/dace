# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util
import os
import pathlib
import shutil

import dace.library


@dace.library.environment
class rocBLAS:

    cmake_minimum_version = None
    cmake_packages = [""]
    cmake_variables = {}
    cmake_libraries = ["rocblas"]
    # The HIP headers require a platform macro when the compiler is not hipcc.
    cmake_compile_flags = ["-D__HIP_PLATFORM_AMD__"]
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_rocblas.h"], 'cuda': ["../include/dace_rocblas.h"]}
    state_fields = ["dace::blas::RocblasHandle rocblas_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_includes():
        """The ROCm include directory, so the HOST compiler can find the HIP headers too.

        A libnode whose rocBLAS call lands in the SDFG's host ``.cpp`` -- every graph reaching this
        environment through the ``frame`` header entry does -- is compiled by g++, which knows
        nothing of HIP. hipcc gets the include path implicitly and g++ does not, so ``<hip/...>``
        does not resolve and the build stops on the first one.

        Resolved HERE rather than as a CMake variable: ``${HIP_PATH}`` is expanded before the HIP
        block that defines it, so it lands on the compile line as a bare ``-I/include`` (measured).
        Asked of the environment and then of ``hipcc`` on PATH, so no ROCm path is written down.
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
rocblas_handle &__dace_rocblas_handle = __state->rocblas_handle.Get();
dace::blas::CheckRocblasError(rocblas_set_stream(__dace_rocblas_handle, __dace_current_stream));\n"""

    @staticmethod
    def _find_library():
        # *nix-based search
        blas_path = ctypes.util.find_library('rocblas')
        if blas_path:
            return [blas_path]

        return []

    @staticmethod
    def is_installed():
        return len(rocBLAS._find_library()) > 0
