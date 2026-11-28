# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace.library
from dace.libraries import hip_dialect
from dace.libraries.lapack.environments.cusolverdn import cuSolverDn


@dace.library.environment
class hipSolverDn:
    """cuSOLVER in the HIP dialect: the same header and handle, linked against hipSOLVER."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = ["hipsolver"]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = cuSolverDn.headers
    state_fields = ["dace::lapack::CusolverDnHandle hipsolverDn_handle;"]
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def handle_setup_code(node):
        return hip_dialect.to_hip(cuSolverDn.handle_setup_code(node), hip_dialect.SOLVER)

    @staticmethod
    def is_installed():
        return hip_dialect.find_rocm_library("hipsolver") is not None
