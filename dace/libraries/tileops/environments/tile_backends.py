# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One environment per ISA backend of the tile nodes.

The CPU backends (scalar, AVX2, AVX-512, NEON) call per-signature functions the code generator writes into the unit
(:mod:`dace.tile_functions`), so their environments only add the compile flag that lets those functions
inline; SVE and CUDA include their ``dace/tile_ops/<backend>.h`` header. A backend is selected only for a target that
supports it, so its compile flag is safe to add.
"""

import dace.library


class TileOpsHeaderOnly:
    """The fields of an environment that adds a header and nothing else."""

    cmake_minimum_version = None
    cmake_packages: list[str] = []
    cmake_variables: dict[str, str] = {}
    cmake_includes: list[str] = []
    cmake_libraries: list[str] = []
    cmake_compile_flags: list[str] = []
    cmake_link_flags: list[str] = []
    cmake_files: list[str] = []

    headers: dict[str, list[str]] = {}
    state_fields: list[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: list[str] = []


@dace.library.environment
class TileOpsScalar(TileOpsHeaderOnly):
    """The portable scalar backend: the reference every other backend is checked against."""


@dace.library.environment
class TileOpsAVX512(TileOpsHeaderOnly):
    """AVX-512; ``-mavx512f`` lets the ``_mm512`` functions inline into the kernel."""

    cmake_compile_flags = ["-mavx512f"]


@dace.library.environment
class TileOpsAVX2(TileOpsHeaderOnly):
    """AVX2; ``-mavx2`` lets the ``_mm256`` functions inline into the kernel."""

    cmake_compile_flags = ["-mavx2"]


@dace.library.environment
class TileOpsNeon(TileOpsHeaderOnly):
    """AArch64 Advanced SIMD, which is baseline there and needs no flag."""


@dace.library.environment
class TileOpsSVE(TileOpsHeaderOnly):
    """ARM SVE, which AArch64 does not enable by default."""

    cmake_compile_flags = ["-march=armv8-a+sve"]
    headers = {"frame": ["dace/tile_ops/arm_sve.h"]}


@dace.library.environment
class TileOpsCUDA(TileOpsHeaderOnly):
    """CUDA and HIP device code, where the fp16 ops use the native ``half2`` intrinsics.

    The calls are emitted inside the kernel, so the header goes into the device translation unit. It is also listed
    for the host one: the VLEN=1 overloads are plain ``inline`` so a host-side tile op resolves, and the
    ``__CUDACC__`` guard keeps the device bodies out of the host frame.
    """

    headers = {"frame": ["dace/tile_ops/cuda.h"], "cuda": ["dace/tile_ops/cuda.h"]}


@dace.library.environment
class TileOpsCUDABlock(TileOpsHeaderOnly):
    """The thread-block collectives of the ``block`` lowering (``cub``, ``hipcub`` on HIP), only in device code."""

    headers = {"cuda": ["dace/tile_ops/cuda_block.h"]}
