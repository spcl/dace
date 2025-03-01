# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util

import dace.library


@dace.library.environment
class BLAS:
    """
    General CPU-based BLAS library environment definition.
    """

    cmake_minimum_version = "3.6"
    cmake_packages = ["BLAS"]
    cmake_variables = {}
    cmake_includes = []
    cmake_compile_flags = []
    cmake_link_flags = ["${BLAS_LINKER_FLAGS}"]
    cmake_files = []

    headers = ["cblas.h", "../include/dace_blas.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_libraries():
        blas_path = ctypes.util.find_library('blas')
        if blas_path:
            return [blas_path]
        return []

    @staticmethod
    def is_installed():
        return len(BLAS.cmake_libraries()) > 0
