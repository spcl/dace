# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import ctypes.util


@dace.library.environment
class cuBLAS:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cublas"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_cublas.h"], 'cuda': ["../include/dace_cublas.h"]}
    state_fields = ["dace::blas::CublasHandle cublas_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        location = node.location
        if not location or "gpu" not in node.location:
            location = -1  # -1 means current device
        else:
            try:
                location = int(location["gpu"])
            except ValueError:
                raise ValueError("Invalid GPU identifier: {}".format(location))

        code = """\
const int __dace_cuda_device = {location};
cublasHandle_t &__dace_cublas_handle = __state->cublas_handle.Get(__dace_cuda_device);
cublasSetStream(__dace_cublas_handle, __dace_current_stream);\n"""

        return code.format(location=location)

    @staticmethod
    def _find_library():
        # *nix-based search
        blas_path = ctypes.util.find_library('cublas')
        if blas_path:
            return [blas_path]

        # Windows-based search
        versions = (10, 11, 12)
        for version in versions:
            blas_path = ctypes.util.find_library(f'cublas64_{version}')
            if blas_path:
                return [blas_path]
        return []

    @staticmethod
    def is_installed():
        return len(cuBLAS._find_library()) > 0
