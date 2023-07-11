# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes
import os
from dace import config, library


@library.environment
class HPTT:

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["hptt.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_includes():
        if 'HPTT_ROOT' in os.environ:
            return [os.path.join(os.environ['HPTT_ROOT'], 'include')]
        else:
            return []

    @staticmethod
    def cmake_libraries():
        if 'HPTT_ROOT' in os.environ:
            prefix = config.Config.get('compiler', 'library_prefix')
            suffix = config.Config.get('compiler', 'library_extension')
            libfile = os.path.join(os.environ['HPTT_ROOT'], 'lib', prefix + 'hptt.' + suffix)
            if os.path.isfile(libfile):
                return [libfile]

        return ['hptt']
