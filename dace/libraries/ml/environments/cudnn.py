# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util

import dace.library


@dace.library.environment
class cuDNN:
    """Environment for the NVIDIA cuDNN deep neural network primitive library."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cudnn"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {"frame": ["../include/dace_cudnn.h"], "cuda": ["../include/dace_cudnn.h"]}
    state_fields = ["dace::ml::CudnnHandle cudnn_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        return (
            dace.library.reject_gpu_location(node)
            + """\
cudnnHandle_t &__dace_cudnn_handle = __state->cudnn_handle.Get();
dace::ml::CheckCudnnError(cudnnSetStream(__dace_cudnn_handle, __dace_current_stream));\n"""
        )

    @staticmethod
    def _find_library():
        path = ctypes.util.find_library("cudnn")
        if path:
            return [path]
        return []

    @staticmethod
    def is_installed():
        return len(cuDNN._find_library()) > 0
