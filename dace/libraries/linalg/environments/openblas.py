# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class OpenBLAS:

    cmake_minimum_version = "3.6"
    cmake_packages = ["LAPACK", "BLAS"]
    cmake_variables = {"BLA_VENDOR": "OpenBLAS"}
    cmake_includes = []  
    cmake_libraries = ["${LAPACK_LIBRARIES}", "${BLAS_LIBRARIES}"]
    cmake_compile_flags = []
    cmake_link_flags = ["${LAPACK_LINKER_FLAGS}", "${BLAS_LINKER_FLAGS}"]
    cmake_files = []

    headers = ["cblas.h", "lapacke.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
