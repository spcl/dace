# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment for hipTensor: cuTENSOR in the HIP dialect."""

import dace.library
from dace.libraries import hip_dialect
from dace.libraries.linalg.environments.cutensor import cuTensor


@dace.library.environment
class hipTensor:
    """cuTENSOR in the HIP dialect: the same header and handle, linked against hipTensor."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["hiptensor"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = cuTensor.headers
    state_fields = ["dace::linalg::CuTensorHandle hiptensor_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []
    TYPE_MAP = cuTensor.TYPE_MAP

    @staticmethod
    def handle_setup_code(node):
        return hip_dialect.to_hip(cuTensor.handle_setup_code(node), hip_dialect.TENSOR)
