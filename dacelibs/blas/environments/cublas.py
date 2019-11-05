import dace.library

@dace.library.environment
class cuBLAS:

    cmake_minimum_version = None
    cmake_packages = ["CUDA"]
    cmake_variables = {}
    cmake_includes = ["${CUDA_INCLUDE_DIRS}"]
    cmake_libraries = ["mmul_2", "cublas", "curand"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["../include/dace_cublas.h"]
    init_code = ""
    finalize_code = ""
