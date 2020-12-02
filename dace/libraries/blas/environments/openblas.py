# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
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

    headers = ["cblas.h", "../include/dace_blas.h"]
    init_code = ""
    finalize_code = ""
