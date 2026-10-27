# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
from dace.libraries import hip_dialect
from dace.libraries.blas.environments.cublas import cuBLAS


@dace.library.environment
class hipBLAS:
    """cuBLAS in the HIP dialect: the same header and handle, linked against hipBLAS."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["hipblas"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = cuBLAS.headers
    state_fields = ["dace::blas::CublasHandle hipblas_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        return hip_dialect.to_hip(cuBLAS.handle_setup_code(node), hip_dialect.BLAS)

    @staticmethod
    def is_installed():
        return hip_dialect.find_rocm_library("hipblas") is not None
