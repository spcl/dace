# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Environment exposing the thread-block LAPACK kernels to the GPU unit."""

import dace.library


@dace.library.environment
class BlockLapack:
    """``dace/cuda/block_lapack.cuh``: unblocked factorizations and solves run by one thread block. GPU unit only."""

    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_includes: list[str] = []
    cmake_libraries: list[str] = []
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers = {"frame": [], "cuda": ["dace/cuda/block_lapack.cuh"]}
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: list[str] = []
