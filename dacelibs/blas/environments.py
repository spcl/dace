import dace.library

@dace.library.environment
class OpenBLAS:

    cmake_minimum_version = "3.6"
    cmake_packages = ["BLAS"]
    cmake_variables = {"BLA_VENDOR": "OpenBLAS"}
    cmake_includes = []  # For some reason, FindBLAS does not find includes
    cmake_libraries = ["${BLAS_LIBRARIES}"]
    cmake_compile_flags = []
    cmake_link_flags = ["${BLAS_LINKER_FLAGS}"]
    cmake_files = []

    headers = ["cblas.h"]
    init_code = ""
    finalize_code = ""

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

    headers = ["cblas.h"]
    init_code = ""
    finalize_code = ""
