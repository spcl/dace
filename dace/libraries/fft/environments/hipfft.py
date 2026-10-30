# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""hipFFT environment for the ROCm backend of the :mod:`dace.libraries.fft` library nodes."""

import dace.library
from dace.libraries import hip_dialect
from dace.libraries.fft.environments.plans import FFTPlans


@dace.library.environment
class hipFFT:
    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_libraries = ["hipfft"]
    # The HIP headers require a platform macro when the compiler is not hipcc.
    cmake_compile_flags = ["-D__HIP_PLATFORM_AMD__"]
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers = {"frame": ["hipfft/hipfft.h", "hipfft/hipfftXt.h"], "cuda": ["hipfft/hipfft.h", "hipfft/hipfftXt.h"]}
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    # The plan cache the renamed cuFFT code keeps its plans in
    dependencies = [FFTPlans]

    cmake_includes: list[str] = []

    @staticmethod
    def is_installed():
        return hip_dialect.find_rocm_library("hipfft") is not None
