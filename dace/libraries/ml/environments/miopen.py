# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util

import dace.library


@dace.library.environment
class MIOpen:
    """Environment for the AMD MIOpen deep neural network primitive library."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["MIOpen"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {"frame": ["../include/dace_miopen.h"], "cuda": ["../include/dace_miopen.h"]}
    state_fields = ["dace::ml::MiopenHandle miopen_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        return (
            dace.library.reject_gpu_location(node)
            + """\
miopenHandle_t &__dace_miopen_handle = __state->miopen_handle.Get();
dace::ml::CheckMiopenError(miopenSetStream(__dace_miopen_handle, __dace_current_stream));\n"""
        )

    @staticmethod
    def _find_library():
        path = ctypes.util.find_library("MIOpen")
        if path:
            return [path]
        return []

    @staticmethod
    def is_installed():
        return len(MIOpen._find_library()) > 0
