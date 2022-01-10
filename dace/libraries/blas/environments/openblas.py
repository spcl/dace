# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import ctypes.util


@dace.library.environment
class OpenBLAS:

    # NOTE: This works with OpenBLAS on Linux when liblapack and libblas are
    # pointing to libopenblas through update-alternatives.

    cmake_minimum_version = "3.6"
    cmake_packages = ["LAPACK", "BLAS"]
    cmake_variables = {"BLA_VENDOR": "OpenBLAS"}
    cmake_includes = []  # For some reason, FindBLAS does not find includes
    cmake_compile_flags = []
    cmake_link_flags = ["${LAPACK_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}"]
    cmake_files = []

    headers = ["cblas.h", "lapacke.h", "../include/dace_blas.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_libraries():
        lapacke_path = ctypes.util.find_library('lapacke')
        blas_path = ctypes.util.find_library('blas')
        if lapacke_path and blas_path:
            return [lapacke_path, blas_path]
        return []

    @staticmethod
    def is_installed():
        return len(OpenBLAS.cmake_libraries()) > 0
