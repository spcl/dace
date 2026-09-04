# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ctypes.util

import dace.library
from dace.libraries.blas.environments.rocblas import rocBLAS


@dace.library.environment
class rocSOLVER:
    """rocSOLVER: the AMD counterpart of cuSolverDn.

    Deliberately thin. rocSOLVER has no handle type of its own -- every entry point takes a
    ``rocblas_handle`` -- so this environment carries no state field and no create/destroy of its
    own; it depends on :class:`~dace.libraries.blas.environments.rocblas.rocBLAS` and reuses that
    handle, which is also what keeps a graph mixing a factorization with a GEMM on ONE handle and
    therefore one stream. The only thing it adds is the library to link and the header to include.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["rocsolver"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_rocsolver.h"], 'cuda': ["../include/dace_rocsolver.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = [rocBLAS]

    @staticmethod
    def handle_setup_code(node):
        # The rocBLAS handle, verbatim: rocSOLVER shares it, so setting the stream twice on two
        # handles is exactly what we are avoiding.
        return rocBLAS.handle_setup_code(node)

    @staticmethod
    def is_installed():
        return ctypes.util.find_library('rocsolver') is not None
