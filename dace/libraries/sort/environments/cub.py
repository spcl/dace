# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment exposing the CUDA CUB headers for ``DeviceRadixSort``.

``cub::DeviceRadixSort`` is part of the CUDA toolkit (under ``cub/cub.cuh``), so
no extra CMake package is needed beyond the existing CUDA setup. The environment
declares the include and inherits the standard CUDA environment for the runtime.
"""
import dace.library
from dace.libraries.standard.environments.cuda import CUDA


@dace.library.environment
class CUB:
    """CUB headers + the DaCe-internal scratchpad helper.

    Pulls in ``cub/cub.cuh`` and :file:`dace/runtime/include/dace/cub_scratch.cuh`,
    which exposes ``::dace::cub::get_scratch<Tag>(N)`` / ``release_scratch<Tag>()`` --
    a per-libnode-class persistent device-memory pool that lets CUB libnodes skip
    ``cudaMalloc`` on the hot path of repeated SDFG invocations.

    The pool's lifecycle (pre-allocate at init, release at exit) is owned by each
    CUB libnode's own scratch environment (e.g. :class:`SortScratch`, :class:`ScanScratch`),
    not by this base env -- so an SDFG that uses only one libnode class doesn't
    pay the pre-allocation cost for the other.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    # Surface the CUDA Toolkit include directories on the CXX (g++) side too.
    # CUB libnodes' host-side wrappers (e.g. ``cub::DeviceScan::InclusiveScan``
    # in ``Scan.ExpandCUDA``) land in the SDFG's host ``.cpp`` translation
    # unit; ``enable_language(CUDA)`` only adjusts nvcc's include path, so g++
    # cannot otherwise find ``cub/cub.cuh``.
    #
    # CUDA Toolkit 13+ relocated CUB under ``cccl/`` -- nvcc auto-resolves both
    # paths, g++ needs them explicit. Listing both is safe: the 12.x cccl
    # subdir typically does not exist and CMake silently ignores missing
    # entries when added via ``include_directories``.
    cmake_includes = ['${CUDAToolkit_INCLUDE_DIRS}', '${CUDAToolkit_INCLUDE_DIRS}/cccl']
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': ['cub/cub.cuh', 'dace/cub_scratch.cuh', 'dace/cub_compat.cuh']}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = [CUDA]


#: Initial scratchpad size per (CUB libnode class, CUDA stream) pair (128 MB). The pool
#: is keyed by stream so concurrent libnode launches on different streams do not race;
#: 128 MB per stream covers ``N`` up to ~25 M int32 keys for radix-sort and similar
#: scales for scan, on the assumption that a single SDFG uses only a handful of streams.
#: The grow-on-demand path in the tasklet still handles larger ``N`` by reallocating.
_CUB_INITIAL_BYTES_PER_STREAM = 128 * 1024 * 1024


@dace.library.environment
class SortScratch:
    """Pre-allocate (128 MB on the default stream) and release the ``IntegerSort`` CUB scratch pool.

    Additional streams allocate lazily on first use (via :func:`dace::cub::get_scratch`);
    every per-stream entry is freed at SDFG finalize.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': []}
    state_fields = []
    init_code = f"::dace::cub::get_scratch<::dace::cub::SortTag>({_CUB_INITIAL_BYTES_PER_STREAM}ull, 0);"
    finalize_code = "::dace::cub::release_scratch<::dace::cub::SortTag>();"
    dependencies = [CUB]


@dace.library.environment
class ScanScratch:
    """Pre-allocate (128 MB on the default stream) and release the ``Scan`` CUB scratch pool.

    Additional streams allocate lazily on first use (via :func:`dace::cub::get_scratch`);
    every per-stream entry is freed at SDFG finalize.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': []}
    state_fields = []
    init_code = f"::dace::cub::get_scratch<::dace::cub::ScanTag>({_CUB_INITIAL_BYTES_PER_STREAM}ull, 0);"
    finalize_code = "::dace::cub::release_scratch<::dace::cub::ScanTag>();"
    dependencies = [CUB]


@dace.library.environment
class ReduceScratch:
    """Pre-allocate (128 MB on the default stream) and release the ``Reduce`` CUB scratch pool.

    Used by :class:`~dace.libraries.standard.nodes.reduce.ExpandReduceCUDADevice`
    (``cub::DeviceReduce`` / ``cub::DeviceSegmentedReduce``). Additional streams
    allocate lazily on first use; every per-stream entry is freed at SDFG finalize.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': []}
    state_fields = []
    init_code = f"::dace::cub::get_scratch<::dace::cub::ReduceTag>({_CUB_INITIAL_BYTES_PER_STREAM}ull, 0);"
    finalize_code = "::dace::cub::release_scratch<::dace::cub::ReduceTag>();"
    dependencies = [CUB]
