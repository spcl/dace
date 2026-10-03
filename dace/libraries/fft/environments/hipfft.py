# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class hipFFT:

    cmake_minimum_version = None
    cmake_packages = [""]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["hipfft"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["hipfft/hipfft.h", "hipfft/hipfftXt.h"], 'cuda': ["hipfft/hipfft.h", "hipfft/hipfftXt.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
