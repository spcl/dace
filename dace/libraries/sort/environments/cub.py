# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""DaCe library environment exposing the CUDA CUB headers for ``DeviceRadixSort``.

``gpucub::DeviceRadixSort`` is part of the CUDA toolkit (under ``cub/cub.cuh``), so
no extra CMake package is needed beyond the existing CUDA setup. The environment
declares the include and inherits the standard CUDA environment for the runtime.
"""
import dace.library
from dace.codegen import common
from dace.libraries.standard.environments.cuda import CUDA


@dace.library.environment
class CUB:
    """CUB headers + the DaCe-internal scratchpad helper.

    Pulls in ``cub/cub.cuh`` and :file:`dace/runtime/include/dace/cub_scratch.cuh`,
    which exposes ``::dace::cub::get_scratch<Tag>(N)`` / ``release_scratch<Tag>()`` --
    a per-libnode-class persistent device-memory pool that lets CUB libnodes skip
    ``gpuMalloc`` on the hot path of repeated SDFG invocations.

    The pool's lifecycle (pre-allocate at init, release at exit) is owned by each
    CUB libnode's own scratch environment (e.g. :class:`SortScratch`, :class:`ScanScratch`),
    not by this base env -- so an SDFG that uses only one libnode class doesn't
    pay the pre-allocation cost for the other.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}

    @staticmethod
    def cmake_includes():
        """CUDA only. CUB libnodes' host-side wrappers (``gpucub::DeviceScan::InclusiveScan`` in
        ``Scan.ExpandCUDA``) land in the SDFG's host ``.cpp``, and ``enable_language(CUDA)`` only
        adjusts nvcc's include path, so g++ needs the toolkit directories named. Toolkit 13+ moved
        CUB under ``cccl/``; listing both is safe, CMake ignores a missing entry. hipCUB is in the
        ROCm include root the HIP language already carries, so the HIP build needs nothing here --
        and naming a CUDAToolkit variable that was never resolved would expand to nothing useful.
        """
        if common.get_gpu_backend() == 'hip':
            return []
        return ['${CUDAToolkit_INCLUDE_DIRS}', '${CUDAToolkit_INCLUDE_DIRS}/cccl']

    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    # cub/cub.cuh does not compile under a host compiler from CCCL 3 (CUDA 13) on, so only the
    # host-safe scratch header goes to the frame; the wrappers that call CUB live in the .cu.
    # ``dace/cuda/gpucub.cuh`` rather than ``cub/cub.cuh`` directly: it resolves to hipCUB or CUB
    # per backend and defines the ``gpucub`` namespace every expansion emits, which is what lets one
    # expansion serve both instead of a hand-maintained AMD copy of each.
    headers = {
        'frame': ['dace/cub_scratch.cuh'],
        'cuda': ['dace/cuda/gpucub.cuh', 'dace/cub_scratch.cuh', 'dace/cub_compat.cuh'],
    }
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

    #: CUDA unit only: the affine expansion's kernels and its map monoid, and the residue-class
    #: kernels the strided path launches. The host translation unit never sees either -- both
    #: carry ``__global__`` symbols, and the affine header includes ``cub/cub.cuh``, which the
    #: host compiler cannot parse.
    headers = {'frame': [], 'cuda': ['dace/cuda/scan_affine.cuh', 'dace/cuda/scan.cuh']}
    state_fields = []
    init_code = f"::dace::cub::get_scratch<::dace::cub::ScanTag>({_CUB_INITIAL_BYTES_PER_STREAM}ull, 0);"
    finalize_code = "::dace::cub::release_scratch<::dace::cub::ScanTag>();"
    dependencies = [CUB]


@dace.library.environment
class BlockCollectives:
    """Headers only: the in-kernel block collectives (``dace::cuda_scan::detail``) need no scratch.

    A block-wide scan or reduce runs entirely in ``__shared__`` storage declared inside the
    collective, so unlike :class:`ScanScratch` / :class:`ReduceScratch` there is no device pool to
    pre-allocate and nothing to release. Those environments would still work, but each costs a
    128 MB allocation at SDFG init that this path never reads.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    #: CUDA unit only, for the same reason ScanScratch gives: the header includes ``cub/cub.cuh``,
    #: which the host compiler cannot parse.
    headers = {'frame': [], 'cuda': ['dace/cuda/scan.cuh']}
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = [CUB]


@dace.library.environment
class ReduceScratch:
    """Pre-allocate (128 MB on the default stream) and release the ``Reduce`` CUB scratch pool.

    Used by :class:`~dace.libraries.standard.nodes.reduce.ExpandReduceCUDADevice`
    (``gpucub::DeviceReduce`` / ``gpucub::DeviceSegmentedReduce``). Additional streams
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


@dace.library.environment
class DetectScratch:
    """Device detection primitives (``dace/cuda/detect.cuh``) plus their scratch pools.

    Used by the CUDA expansions of :class:`~dace.libraries.standard.nodes.find_first.FindFirst`
    and :class:`~dace.libraries.sort.nodes.scatter_conflict_check.ScatterConflictCheck`.

    The FLAG pool is claimed at init, the way the sort / scan / reduce pools are: it is one word,
    its size never depends on the problem, and taking it here keeps the first call off the
    allocator -- which otherwise shows up inside whatever that first call was being timed for. The
    TAG pool is left alone: it is sized by the scattered array's domain, which init does not know,
    and ``get_scratch`` grows it in place on first use. Both are freed at SDFG finalize.

    The header goes to the ``cuda`` file only -- it instantiates ``gpucub::BlockReduce`` and launches
    kernels, neither of which a host compiler can parse.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = []
    cmake_libraries = []
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = {'frame': [], 'cuda': ['dace/cuda/detect.cuh']}
    state_fields = []
    init_code = "::dace::cub::get_scratch<::dace::cub::DetectFlagTag>(sizeof(unsigned long long), 0);"
    finalize_code = ("::dace::cub::release_scratch<::dace::cub::DetectFlagTag>();\n"
                     "::dace::cub::release_scratch<::dace::cub::DetectOwnerTag>();")
    dependencies = [CUB]
