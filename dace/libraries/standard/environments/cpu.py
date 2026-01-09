# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class CPU:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["cstring"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
