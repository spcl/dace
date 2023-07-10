# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import ctypes.util


@dace.library.environment
class cuTENSOR:

    cmake_minimum_version = None
    cmake_packages = ["CUDA"]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cutensor"]
    cmake_compile_flags = ["-L/users/jbazinsk/libcutensor-linux-x86_64-1.7.0.1-archive/lib/11"]
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_cutensor.h"], 'cuda': ["../include/dace_cutensor.h"]}
    state_fields = ["dace::blas::CutensorHandle cutensor_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        location = node.location
        if not location or "gpu" not in node.location:
            location = 0
        else:
            try:
                location = int(location["gpu"])
            except ValueError:
                raise ValueError("Invalid GPU identifier: {}".format(location))

        code = """\
const int __dace_cuda_device = {location};
cutensorHandle_t* __dace_cutensor_handle = __state->cutensor_handle.Get(__dace_cuda_device);\n"""

        return code.format(location=location)

    @staticmethod
    def _find_library():
        # *nix-based search
        blas_path = ctypes.util.find_library('cutensor')
        if blas_path:
            return [blas_path]

        # Windows-based search
        versions = (10, 11, 12)
        for version in versions:
            blas_path = ctypes.util.find_library(f'cutensor64_{version}')
            if blas_path:
                return [blas_path]
        return []
