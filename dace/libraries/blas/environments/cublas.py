# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


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

    headers = ["../include/dace_cublas.h"]
    state_fields = ["dace::blas::CublasHandle cublas_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        code = """\
cublasHandle_t &__dace_cublas_handle = __state->cublas_handle.Get(__dace_cuda_device);
cublasSetStream(__dace_cublas_handle, __dace_current_stream);\n"""
        return code
