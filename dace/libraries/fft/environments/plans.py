# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The plan cache type of the vendor FFT environments, whose program state holds one cache per plan type."""

import dace.library


@dace.library.environment
class FFTPlans:
    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_includes: list[str] = []
    cmake_libraries: list[str] = []
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers = {"frame": ["../include/dace_fft.h"], "cuda": ["../include/dace_fft.h"]}
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: list[str] = []
