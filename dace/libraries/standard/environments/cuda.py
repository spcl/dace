# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class CUDA:

    cmake_minimum_version = None
    cmake_packages = ["CUDA"]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = []
    init_code = ""
    finalize_code = ""
