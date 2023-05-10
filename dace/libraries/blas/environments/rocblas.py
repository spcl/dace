# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
import ctypes.util
import os
import warnings

from dace import Config


@dace.library.environment
class rocBLAS:

    cmake_minimum_version = None
    cmake_packages = [""]
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["rocblas"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_rocblas.h"], 'cuda': ["../include/dace_rocblas.h"]}
    state_fields = ["dace::blas::RocblasHandle rocblas_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        location = node.location
        if not location or "gpu" not in node.location:
            location = 0
        else:
            try:
                location = int(location["gpu"])
            except ValueError:
                raise ValueError("Invalid GPU identifier: {}".format(location))

        code = """\
const int __dace_cuda_device = {location};
rocblas_handle &__dace_rocblas_handle = __state->rocblas_handle.Get(__dace_cuda_device);
rocblas_set_stream(__dace_rocblas_handle, __dace_current_stream);\n"""

        return code.format(location=location)

    @staticmethod
    def cmake_includes():
        if 'ROCBLAS_ROOT' in os.environ:
            return [os.path.join(os.environ['ROCBLAS_ROOT'], 'include')]
        return []
    
    @staticmethod
    def cmake_libraries():
        if 'ROCBLAS_ROOT' in os.environ:
            prefix = Config.get('compiler', 'library_prefix')
            suffix = Config.get('compiler', 'library_extension')
            libfile = os.path.join(os.environ['ROCBLAS_ROOT'], 'lib', prefix + 'rocblas.' + suffix)
            if os.path.isfile(libfile):
                return [libfile]
        
        path = ctypes.util.find_library('rocblas')
        if path:
            # Attempt to link on Windows
            if path.endswith('.dll'):
                libfile = os.path.join(os.path.dirname(os.path.abspath(path)), '..', 'lib', 'rocblas.lib')
                if os.path.isfile(libfile):
                    return [libfile]
                else:
                    return []

            return [path]

    @staticmethod
    def _find_library():
        # *nix-based search
        blas_path = ctypes.util.find_library('rocblas')
        if blas_path:
            return [blas_path]

        return []

    @staticmethod
    def is_installed():
        return len(rocBLAS._find_library()) > 0
