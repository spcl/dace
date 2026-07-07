# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-backend DaCe environments for the K=1 tile-op intrinsic lowerings.

Each environment pulls in exactly one of the per-backend tile-op headers
(``dace/tile_ops/<backend>.h``). A tile-node's chosen-backend expansion declares
``environments = [TileOps<Backend>]`` so expanding the node ``#include``s the
right header — there is no joint dispatch header. The five backends (scalar /
avx512 / avx2 / arm_neon / arm_sve) expose the same function signatures; the ISA
backends add their ``-m``/``-march`` flag (safe because a backend is selected
only on a host that supports it).
"""
import dace.library


@dace.library.environment
class TileOpsScalar:
    """Portable scalar K=1 tile-op backend (``dace/tile_ops/scalar.h``)."""

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["dace/tile_ops/scalar.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []


@dace.library.environment
class TileOpsAVX512:
    """AVX-512 K=1 tile-op backend (``dace/tile_ops/avx512.h``).

    Adds ``-mavx512f`` so the ``_mm512`` SIMD paths are enabled (the header
    falls back to scalar where the flag is absent). Selected only on an
    AVX-512-capable host, so the flag is always safe for the chosen target.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = ["-mavx512f"]
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["dace/tile_ops/avx512.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []


@dace.library.environment
class TileOpsAVX2:
    """AVX2 K=1 tile-op backend (``dace/tile_ops/avx2.h``).

    Adds ``-mavx2`` (the header ``#error``\\ s without it). Selected only on an
    AVX2-capable host.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = ["-mavx2"]
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["dace/tile_ops/avx2.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []


@dace.library.environment
class TileOpsNeon:
    """ARM NEON (AArch64 Advanced SIMD) K=1 tile-op backend
    (``dace/tile_ops/arm_neon.h``). NEON is baseline on AArch64, so no extra
    compile flag is needed; selected only when targeting AArch64.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["dace/tile_ops/arm_neon.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []


@dace.library.environment
class TileOpsSVE:
    """ARM SVE K=1 tile-op backend (``dace/tile_ops/arm_sve.h``).

    Adds ``-march=armv8-a+sve`` (SVE is not baseline on AArch64). Selected only
    when targeting an SVE-capable AArch64 host.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = ["-march=armv8-a+sve"]
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ["dace/tile_ops/arm_sve.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []


@dace.library.environment
class TileOpsCUDA:
    """NVIDIA CUDA (device) K=1 tile-op backend (``dace/tile_ops/cuda.h``).

    The fp16 elementwise ops use the native ``half2`` (FP16x2) intrinsics from
    ``<cuda_fp16.h>``; fp8 (no native arithmetic) computes through ``float``.
    The header is ``__CUDACC__``-guarded, so listing it in the host frame is
    harmless -- the ``__device__`` bodies only materialise under nvcc. Selected
    only when the tile map is GPU-scheduled.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    # The tile-op calls are emitted INSIDE the GPU kernel (device code), so the
    # header must land in the CUDA (``.cu``) TU -- the ``'cuda'`` key. It is also
    # kept in the ``'frame'`` (host ``.cpp``) TU: the K=1 VLEN=1 overloads are
    # ``inline`` (not ``__device__``) so a host-side tile op still resolves, and
    # the ``__CUDACC__`` guard makes the device bodies inert in the host frame.
    headers = {'frame': ["dace/tile_ops/cuda.h"], 'cuda': ["dace/tile_ops/cuda.h"]}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []
