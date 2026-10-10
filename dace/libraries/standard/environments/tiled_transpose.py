# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Environment exposing the tiled transpose / symmetrize kernels to the CUDA unit."""

import dace.library


@dace.library.environment
class TiledTranspose:
    """``dace/cuda/transpose_tiled.cuh`` -- shared-memory tiled transpose and in-place symmetrize.

    CUDA unit only. The header defines ``__global__`` kernels, which the host translation unit
    cannot parse; the expansions call it through a ``DACE_EXPORTED`` wrapper emitted into the same
    unit, exactly as the CUB-backed nodes do.
    """

    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_includes: list[str] = []
    cmake_libraries: list[str] = []
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers = {"frame": [], "cuda": ["dace/cuda/transpose_tiled.cuh"]}
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: list[str] = []
