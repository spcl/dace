// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once
#include "cuda/cudacommon.cuh"  // gpu* runtime aliases, and the backend runtime header
#include <cstddef>
#include <unordered_map>

namespace dace {
namespace cub {

/// Per-libnode-class, per-CUDA-stream persistent scratch pool for CUB device primitives.
///
/// CUB ``DeviceRadixSort`` / ``DeviceScan`` / ``DeviceReduce`` all require a
/// temporary device buffer whose size depends on the input length and op. The
/// per-call ``gpuMalloc`` + ``gpuFree`` pattern is fine for one-shot kernels
/// but is prohibitively expensive on the hot path of repeated SDFG invocations
/// (each ``gpuMalloc`` issues a device-wide synchronisation).
///
/// Each *class* of CUB libnode (e.g. ``IntegerSort``, ``Scan``) gets its own
/// pool, tagged by a tag struct -- so an ``IntegerSort`` call and a ``Scan``
/// call never clobber each other's scratch. Within a class, the pool is
/// *further* keyed by ``gpuStream_t``: two libnode instances of the same
/// class running on *different* streams have independent scratch buffers, so
/// concurrent kernel launches on multiple streams cannot race on the pool.
/// Instances on the same stream share that stream's buffer; CUDA serialises
/// kernel execution within a stream, so the sharing is race-free.
///
/// The maps live in C++17 inline-static function-locals, so each compiled SDFG
/// shared library has its own pool table (pool state does not leak between
/// SDFGs). Stream entries are allocated lazily on first use.

namespace _detail {
    struct ScratchEntry {
        void *storage = nullptr;
        std::size_t bytes = 0;
    };

    /// Per-Tag map ``gpuStream_t -> ScratchEntry``. Each ``Tag`` instantiation
    /// owns an independent map; each stream gets a lazy entry on first use.
    template<typename Tag>
    inline std::unordered_map<gpuStream_t, ScratchEntry> &pool_map() {
        static std::unordered_map<gpuStream_t, ScratchEntry> m;
        return m;
    }
}  // namespace _detail

/// Return a device pointer to a scratch buffer at least ``bytes_needed`` bytes
/// in size for the pool tagged ``Tag`` and the given ``stream``. Grows the
/// per-stream entry in place if its current allocation is too small. The
/// returned pointer is valid until the next :func:`get_scratch` call (same
/// ``Tag`` + ``stream``) that grows the entry, or until :func:`release_scratch`
/// is called for that ``Tag``. ``status``, when given, receives the allocation's error code; a
/// failed allocation returns ``nullptr``, which the caller MUST treat as an error rather than pass
/// on to CUB (see below).
template<typename Tag>
inline void *get_scratch(std::size_t bytes_needed, gpuStream_t stream = 0, gpuError_t *status = nullptr) {
    // Keyed per stream, and the default/null stream (0) is just another key -- entries never share.
    auto &e = _detail::pool_map<Tag>()[stream];
    if (status) *status = gpuSuccess;
    // ``!e.storage`` and the 1-byte floor are both load-bearing: CUB reads a NULL workspace as "only
    // report the size" and leaves the output untouched, so handing back a null pointer turns the
    // reduction into a silent no-op rather than an error. A zero-byte request hits that twice --
    // ``0 > 0`` skips the allocation, and gpuMalloc(0) hands back null anyway.
    if (bytes_needed > e.bytes || !e.storage) {
        // Discarded on purpose, and explicitly: the buffer is being replaced either way, and
        // the backend's free is [[nodiscard]] (warnings are errors).
        if (e.storage) (void)gpuFree(e.storage);
        gpuError_t err = gpuMalloc(&e.storage, bytes_needed ? bytes_needed : 1);
        if (err != gpuSuccess) {
            // The entry has to go back to empty: gpuMalloc leaves the pointer unspecified on
            // failure, and a stale non-null one would be handed to the next caller as a live buffer.
            e.storage = nullptr;
            e.bytes = 0;
            if (status) *status = err;
            return nullptr;
        }
        e.bytes = bytes_needed;
    }
    return e.storage;
}

/// Free every per-stream scratch entry tagged ``Tag``. Idempotent: a no-op if
/// no entries were ever allocated or all have already been released. Intended
/// for SDFG-finalize code.
template<typename Tag>
inline void release_scratch() {
    auto &m = _detail::pool_map<Tag>();
    for (auto &kv : m) {
        if (kv.second.storage) (void)gpuFree(kv.second.storage);  // finalize: nothing to report to
    }
    m.clear();
}

// -- Tag structs for the CUB-backed libnodes ---------------------------------

/// Tag for the ``IntegerSort`` libnode's CUB scratch pool.
struct SortTag {};

/// Tag for the ``Scan`` libnode's CUB scratch pool.
struct ScanTag {};

/// Tag for the ``Reduce`` libnode's CUB scratch pool
/// (``gpucub::DeviceReduce`` / ``gpucub::DeviceSegmentedReduce``).
struct ReduceTag {};

/// Tag for the device detection primitives' result word
/// (``dace/cuda/detect.cuh``: the find-first answer, the collision and sign flags).
struct DetectFlagTag {};

/// Tag for the device collision check's tag array, sized by the scattered array's domain.
struct DetectOwnerTag {};

}  // namespace cub
}  // namespace dace
