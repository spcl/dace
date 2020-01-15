import dace.library

@dace.library.environment
class IntelMKL:

    cmake_minimum_version = None
    cmake_packages = ["BLAS"]
    cmake_variables = {"BLA_VENDOR": "Intel10_64lp"}
    cmake_includes = []  # For some reason, FindBLAS does not find includes
    cmake_libraries = ["mkl_rt"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["mkl.h", "../include/dace_blas.h"]
    init_code = ""
    finalize_code = ""
