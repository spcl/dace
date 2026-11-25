# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
from dace.codegen import common


def _cmake_packages() -> list[str]:
    """HIP builds find their toolchain in the runtime CMake file; ``FindCUDA`` would fail without a CUDA toolkit."""
    return [] if common.get_gpu_backend() == "hip" else ["CUDA"]


@dace.library.environment
class CUDA:
    cmake_minimum_version = None
    cmake_packages = _cmake_packages
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = []
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
