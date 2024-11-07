# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util

import dace.library


@dace.library.environment
class BLAS:
    """
    General CPU-based BLAS library environment definition.
    """

    cmake_minimum_version = ""
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = []
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_libraries():
        return []

    @staticmethod
    def is_installed():
        return True
