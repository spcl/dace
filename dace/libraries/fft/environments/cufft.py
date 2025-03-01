# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class cuFFT:

    cmake_minimum_version = None
    cmake_packages = ["CUDA"]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["cufft"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["cufft.h", "cufftXt.h"], 'cuda': ["cufft.h", "cufftXt.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
