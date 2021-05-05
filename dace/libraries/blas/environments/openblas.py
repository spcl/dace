# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import ctypes.util

@dace.library.environment
class OpenBLAS:

    cmake_minimum_version = "3.6"
    cmake_packages = ["LAPACK", "BLAS"]
    cmake_variables = {"BLA_VENDOR": "OpenBLAS"}
    cmake_includes = []  # For some reason, FindBLAS does not find includes
    cmake_compile_flags = []
    # cmake_link_flags = ["${BLAS_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}"]
    cmake_link_flags = ["-llapacke -llapack -lblas -lgfortran"]
    cmake_files = []

    headers = ["cblas.h", "lapacke.h", "../include/dace_blas.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_libraries():
        path = ctypes.util.find_library('openblas')
        print(path)
        if path:
            return ['liblapacke.so', 'liblapack.so', 'libblas.so']
            # return ["${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}"]
            # return ["${BLAS_openblas_LIBRARY}", "${LAPACK_openblas_LIBRARY}"]

        return []

    @staticmethod
    def is_installed():
        return len(OpenBLAS.cmake_libraries()) > 0
