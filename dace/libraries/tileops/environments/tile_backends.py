# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""One environment per ISA backend of the tile-op headers (``dace/tile_ops/<backend>.h``).

An ISA expansion of a tile node declares its backend's environment, so expanding the node includes that one header.
The backends expose the same ``dace::tileops`` signatures and differ in the instructions behind them; a backend is
selected only for a target that supports it, so its compile flag is safe to add.
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

    headers = {"frame": ["dace/tile_ops/scalar.h"]}


@dace.library.environment
class TileOpsAVX512(TileOpsHeaderOnly):
    """AVX-512; ``-mavx512f`` enables the ``_mm512`` paths."""

    cmake_compile_flags = ["-mavx512f"]
    headers = {"frame": ["dace/tile_ops/avx512.h"]}


@dace.library.environment
class TileOpsAVX2(TileOpsHeaderOnly):
    """AVX2; the header refuses to compile without ``-mavx2``."""

    cmake_compile_flags = ["-mavx2"]
    headers = {"frame": ["dace/tile_ops/avx2.h"]}


@dace.library.environment
class TileOpsNeon(TileOpsHeaderOnly):
    """AArch64 Advanced SIMD, which is baseline there and needs no flag."""

    headers = {"frame": ["dace/tile_ops/arm_neon.h"]}


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
