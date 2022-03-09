# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from dace import library


@library.environment
class HPTT:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = ["-I${HPTT_ROOT}/include"]
    cmake_link_flags = ["-L${HPTT_ROOT}/lib -lhptt"]
    cmake_files = []

    headers = ["http.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
