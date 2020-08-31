# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace.library

def _find_mkl_include():
    if 'MKLROOT' in os.environ:
        return [os.path.join(os.environ['MKLROOT'], 'include')]
    else:
        return []

@dace.library.environment
class IntelMKL:

    cmake_minimum_version = None
    cmake_packages = ["BLAS"]
    cmake_variables = {"BLA_VENDOR": "Intel10_64lp"}
    cmake_includes = _find_mkl_include()
    cmake_libraries = ["mkl_rt"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["mkl.h", "../include/dace_blas.h"]
    init_code = ""
    finalize_code = ""
