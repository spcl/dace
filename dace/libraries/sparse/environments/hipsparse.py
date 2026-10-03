# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment for the AMD hipSPARSE backend."""
import ctypes.util
import os
import pathlib
import shutil

import dace.library


@dace.library.environment
class hipSPARSE:
    """hipSPARSE: the AMD counterpart of cuSPARSE.

    hipSPARSE is the PORTABILITY layer, not the native one -- it exposes the cuSPARSE generic API
    call for call (``hipsparseCreateCsr``, ``hipsparseSpMV``, ``hipsparseDnVecDescr_t``), so the
    expansion body is shared with cuSPARSE and only the names differ. rocSPARSE is the native
    library underneath and spells everything differently; using it here would mean a second
    expansion body for no gain, since hipSPARSE dispatches to rocSPARSE anyway.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_libraries = ["hipsparse"]
    # The HIP headers require a platform macro when the compiler is not hipcc.
    cmake_compile_flags = ["-D__HIP_PLATFORM_AMD__"]
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["../include/dace_hipsparse.h"], 'cuda': ["../include/dace_hipsparse.h"]}
    state_fields = ["dace::sparse::HipsparseHandle hipsparse_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_includes():
        """The ROCm include directory, so the HOST compiler can find the HIP headers too.

        Same reason as the rocBLAS environment: a tasklet reaching this through the ``frame`` entry
        is compiled by g++, which gets no HIP include path of its own.
        """
        root = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
        if not root:
            driver = shutil.which("hipcc")
            root = str(pathlib.Path(driver).resolve().parent.parent) if driver else ""
        include = pathlib.Path(root) / "include" if root else None
        return [str(include)] if include and include.is_dir() else []

    @staticmethod
    def handle_setup_code(node):
        return dace.library.reject_gpu_location(node) + """\
hipsparseHandle_t &__dace_hipsparse_handle = __state->hipsparse_handle.Get();
hipsparseSetStream(__dace_hipsparse_handle, __dace_current_stream);\n"""

    @staticmethod
    def is_installed():
        return ctypes.util.find_library('hipsparse') is not None
