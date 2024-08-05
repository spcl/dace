# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library


@dace.library.environment
class IPU:

    cmake_minimum_version = "3.6"
    cmake_packages = ["IPU"]
    cmake_files = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["${IPU_CXX_LIBRARIES}"]
    cmake_compile_flags = ["-I${IPU_CXX_HEADER_DIR}"]
    cmake_link_flags = ["${IPU_LINKER_FLAGS}"]

    headers = ["poplar.h"]
    state_fields = []
    init_code = "This is init code"
    finalize_code = "This is finalize code;"  # actually if we finalize in the dace program we break pytest :)
    dependencies = []
