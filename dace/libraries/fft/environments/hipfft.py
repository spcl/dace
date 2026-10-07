# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""hipFFT environment for the ROCm backend of the :mod:`dace.libraries.fft` library nodes."""

import ctypes.util

import dace.library
from dace.libraries.blas.environments.rocblas import rocBLAS
from typing import Dict, List


@dace.library.environment
class hipFFT:
    cmake_minimum_version = None
    cmake_packages: List[str] = []
    cmake_variables: Dict[str, str] = {}
    cmake_libraries = ["hipfft"]
    # The HIP headers require a platform macro when the compiler is not hipcc.
    cmake_compile_flags = ["-D__HIP_PLATFORM_AMD__"]
    cmake_link_flags: List[str] = []
    cmake_files: List[str] = []

    headers = {"frame": ["hipfft/hipfft.h", "hipfft/hipfftXt.h"], "cuda": ["hipfft/hipfft.h", "hipfft/hipfftXt.h"]}
    state_fields: List[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: List[str] = []

    @staticmethod
    def cmake_includes():
        """The ROCm include directory, for the host compiler that builds the plan-and-execute call."""
        return rocBLAS.cmake_includes()

    @staticmethod
    def is_installed():
        return ctypes.util.find_library("hipfft") is not None
