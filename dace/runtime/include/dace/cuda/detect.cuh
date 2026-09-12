// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_DETECT_CUH
#define __DACE_DETECT_CUH

// Device counterparts of the host detection primitives in ``dace/detect.h``: duplicate
// detection for a scatter index, an all-positive sign check, and a short-circuiting
// find-first over a predicate. Same contracts, same return values; only the machine
// differs, so a libnode picks an expansion and nothing else changes.
//
// All three end in a block-wide reduction folded by ``gpucub::BlockReduce`` and ONE atomic
// per block, which is the shape DaCe's own GPU WCR lowering emits (see
// ``drain_gpu_block_reduction`` in ``dace/codegen/targets/cpu.py``, down to naming
// ``BLOCK_REDUCE_WARP_REDUCTIONS``): a per-thread atomic on a single word is correct but
// serializes the whole grid on one cache line.
//
// CUB is spelled ``::gpucub::`` throughout: this code lives in namespace ``dace``, where a bare
// ``gpucub::`` resolves to ``dace::cub`` -- the scratch-pool namespace from ``cub_scratch.cuh``, not
// the library.
//
// This header belongs in the ``.cu`` translation unit. A libnode reaches it the way the
// CUB libnodes do: a ``DACE_EXPORTED`` wrapper appended to the device global code, declared
// in the host global code and called from the tasklet.

#include "gpucub.cuh"  // ::gpucub -- cub on CUDA, hipCUB on HIP

#include "../cub_scratch.cuh"

namespace dace {

//: Threads per block for every detection kernel. One value, because all three kernels are
//: memory bound and the block only has to be wide enough to amortize the reduction.
static constexpr int DETECT_BLOCK_THREADS = 256;

//: Cap on the blocks launched. The kernels are grid-stride, so a cap costs nothing on a long
//: input and keeps the number of tiles in flight bounded -- which is what makes the find-first
//: cancellation reach every block within one tile instead of one whole pass.
static constexpr int DETECT_MAX_BLOCKS = 4096;

namespace detect_detail {

//: Bitwise-or over a 0/1 flag: the block-level fold of "did anything trip".
struct OrOp {
    __device__ __forceinline__ int operator()(const int &a, const int &b) const { return a | b; }
};

//: Minimum over candidate indices: the block-level fold of the find-first argmin.
struct MinOp {
    __device__ __forceinline__ long long operator()(const long long &a, const long long &b) const {
        return a < b ? a : b;
    }
};

//: Maximum over index values: the block-level fold that sizes a self-sized tag buffer.
struct MaxOp {
    __device__ __forceinline__ long long operator()(const long long &a, const long long &b) const {
        return a > b ? a : b;
    }
};

}  // namespace detect_detail

/**
 * Pass 1 of the collision check: tag every in-range slot with the index that wrote it.
 *
 * The last-writer-wins race is benign and load-bearing -- see :cpp:func:`dace::detect_collision`
 * for why two passes are the floor and why out-of-range values are skipped.
 */
template <typename T, typename TagT>
__global__ void detect_tag_kernel(const T *idx, long long n, TagT *owner, long long capacity) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long v = static_cast<long long>(idx[i]);
        if (v >= 0 && v < capacity) owner[v] = (TagT)i;
    }
}

/**
 * Pass 2 of the collision check: OR-reduce ``owner[idx[i]] != i`` into ``flag``.
 *
 * A thread that loses pass 1 reads back a different index and trips its lane; the lanes fold
 * through the block and one atomic per block carries the result out.
 */
template <typename T, typename TagT>
__global__ void detect_verify_kernel(const T *idx, long long n, const TagT *owner, long long capacity,
                                     unsigned long long *flag) {
    typedef ::gpucub::BlockReduce<int, DETECT_BLOCK_THREADS, ::gpucub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockOr;
    __shared__ typename BlockOr::TempStorage tmp;

    int local = 0;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long v = static_cast<long long>(idx[i]);
        if (v >= 0 && v < capacity && static_cast<long long>(owner[v]) != i) local = 1;
    }
    const int any = BlockOr(tmp).Reduce(local, detect_detail::OrOp());
    if (threadIdx.x == 0 && any) atomicOr(flag, 1ull);
}

/**
 * OR-reduce ``!(a[i] > 0)`` into ``bad``: the complement of :cpp:func:`dace::detect_all_positive`.
 *
 * Complemented on purpose -- an "any lane tripped" fold starts from the zero the flag word is
 * memset to, so nothing has to seed it.
 */
template <typename T>
__global__ void detect_all_positive_kernel(const T *a, long long n, unsigned long long *bad) {
    typedef ::gpucub::BlockReduce<int, DETECT_BLOCK_THREADS, ::gpucub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockOr;
    __shared__ typename BlockOr::TempStorage tmp;

    int local = 0;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (!(a[i] > 0)) local = 1;
    }
    const int any = BlockOr(tmp).Reduce(local, detect_detail::OrOp());
    if (threadIdx.x == 0 && any) atomicOr(bad, 1ull);
}

/**
 * The find-first search: a cancelling block-per-tile argmin over the firing indices.
 *
 * One tile is one block-wide sweep. Before each tile the block reads the current answer once
 * and stops if its tile starts past it, so cancellation costs one shared load per tile rather
 * than a per-thread atomic; the tile's own minimum folds through ``gpucub::BlockReduce`` and thread
 * 0 alone does the ``atomicMin`` -- so a tile that fires publishes in one atomic, and a tile
 * that does not fire issues none at all.
 *
 * The published answer only ever decreases and every value it takes is a real firing index, so
 * a block reading a stale one skips less than it could have and never skips the true minimum.
 *
 * ``result`` is unsigned because that is the width ``atomicMin`` covers; indices are
 * non-negative, so unsigned and signed order agree.
 */
template <typename Pred>
__global__ void find_first_kernel(long long begin, long long end, Pred pred, unsigned long long *result) {
    typedef ::gpucub::BlockReduce<long long, DETECT_BLOCK_THREADS, ::gpucub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockMin;
    __shared__ typename BlockMin::TempStorage tmp;
    __shared__ long long stop;

    const long long tile = static_cast<long long>(blockDim.x);
    const long long stride = tile * static_cast<long long>(gridDim.x);
    for (long long base = begin + static_cast<long long>(blockIdx.x) * tile; base < end; base += stride) {
        if (threadIdx.x == 0) stop = static_cast<long long>(*static_cast<volatile unsigned long long *>(result));
        __syncthreads();
        if (base >= stop) break;  // uniform across the block: every thread read the same ``stop``

        const long long i = base + threadIdx.x;
        // Short-circuit: the predicate is only evaluated on an index the tile actually covers.
        const long long cand = (i < end && i < stop && pred(i)) ? i : end;
        const long long best = BlockMin(tmp).Reduce(cand, detect_detail::MinOp());
        if (threadIdx.x == 0 && best < stop) atomicMin(result, static_cast<unsigned long long>(best));
        __syncthreads();  // the single TempStorage is reused by the next tile
    }
}

/**
 * :cpp:func:`dace::detect_collision` on the device: 1 if ``idx`` holds a duplicate, else 0.
 *
 * ``owner`` is a device tag buffer of at least ``capacity`` elements and needs no
 * initialization. The answer lands in ``*out`` on the host, so the call synchronizes ``stream``.
 */
template <typename T, typename TagT>
inline gpuError_t detect_collision_device(const T *idx, long long n, TagT *owner, long long capacity, long long *out,
                                          gpuStream_t stream) {
    *out = 0;
    if (n <= 0) return gpuSuccess;

    gpuError_t status = gpuSuccess;
    unsigned long long *flag =
        static_cast<unsigned long long *>(
            ::dace::cub::get_scratch< ::dace::cub::DetectFlagTag>(sizeof(unsigned long long), stream, &status));
    if (flag == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;

    long long blocks = (n + DETECT_BLOCK_THREADS - 1) / DETECT_BLOCK_THREADS;
    if (blocks > DETECT_MAX_BLOCKS) blocks = DETECT_MAX_BLOCKS;

    status = gpuMemsetAsync(flag, 0, sizeof(unsigned long long), stream);
    if (status != gpuSuccess) return status;
    detect_tag_kernel<<<static_cast<unsigned>(blocks), DETECT_BLOCK_THREADS, 0, stream>>>(idx, n, owner, capacity);
    detect_verify_kernel<<<static_cast<unsigned>(blocks), DETECT_BLOCK_THREADS, 0, stream>>>(idx, n, owner, capacity, flag);

    unsigned long long host_flag = 0;
    status = gpuMemcpyAsync(&host_flag, flag, sizeof(unsigned long long), gpuMemcpyDeviceToHost, stream);
    if (status != gpuSuccess) return status;
    status = gpuStreamSynchronize(stream);
    *out = static_cast<long long>(host_flag != 0);
    return status;
}

/**
 * Largest in-range value in ``idx``, OR-free: a block max folded by ``gpucub::BlockReduce`` and one
 * ``atomicMax`` per block. Negative entries clamp to 0 -- they are skipped by the check anyway,
 * and an unsigned atomic would read them as huge.
 */
template <typename T>
__global__ void detect_max_kernel(const T *idx, long long n, unsigned long long *mx) {
    typedef ::gpucub::BlockReduce<long long, DETECT_BLOCK_THREADS, ::gpucub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockMax;
    __shared__ typename BlockMax::TempStorage tmp;

    long long local = 0;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const long long v = static_cast<long long>(idx[i]);
        if (v > local) local = v;
    }
    const long long best = BlockMax(tmp).Reduce(local, detect_detail::MaxOp());
    if (threadIdx.x == 0 && best > 0) atomicMax(mx, static_cast<unsigned long long>(best));
}

/**
 * :cpp:func:`dace::detect_collision` on the device, sizing its own tag buffer.
 *
 * The buffer comes from the persistent per-stream CUB scratch pool, so the repeat call pays no
 * allocation; ``capacity`` still has to be known, since sizing it from ``max(idx)`` would cost a
 * device reduction plus a round trip before the check can even start.
 */
template <typename T>
inline gpuError_t detect_collision_device(const T *idx, long long n, long long capacity, long long *out,
                                          gpuStream_t stream) {
    gpuError_t status = gpuSuccess;
    long long *owner = static_cast<long long *>(::dace::cub::get_scratch< ::dace::cub::DetectOwnerTag>(
        static_cast<size_t>(capacity) * sizeof(long long), stream, &status));
    if (owner == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;
    return detect_collision_device(idx, n, owner, capacity, out, stream);
}

/**
 * :cpp:func:`dace::detect_collision` on the device, sizing the tag buffer from ``max(idx)``.
 *
 * Costs an extra device reduction AND an extra round trip before the check can start, because the
 * tag buffer cannot be allocated until the maximum is known on the host. Prefer the overload
 * taking a capacity wherever the scattered array's domain is known.
 */
template <typename T>
inline gpuError_t detect_collision_device(const T *idx, long long n, long long *out, gpuStream_t stream) {
    *out = 0;
    if (n <= 0) return gpuSuccess;

    gpuError_t status = gpuSuccess;
    unsigned long long *mx =
        static_cast<unsigned long long *>(
            ::dace::cub::get_scratch< ::dace::cub::DetectFlagTag>(sizeof(unsigned long long), stream, &status));
    if (mx == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;

    long long blocks = (n + DETECT_BLOCK_THREADS - 1) / DETECT_BLOCK_THREADS;
    if (blocks > DETECT_MAX_BLOCKS) blocks = DETECT_MAX_BLOCKS;

    status = gpuMemsetAsync(mx, 0, sizeof(unsigned long long), stream);
    if (status != gpuSuccess) return status;
    detect_max_kernel<<<static_cast<unsigned>(blocks), DETECT_BLOCK_THREADS, 0, stream>>>(idx, n, mx);

    unsigned long long host_max = 0;
    status = gpuMemcpyAsync(&host_max, mx, sizeof(unsigned long long), gpuMemcpyDeviceToHost, stream);
    if (status != gpuSuccess) return status;
    status = gpuStreamSynchronize(stream);
    if (status != gpuSuccess) return status;
    return detect_collision_device(idx, n, static_cast<long long>(host_max) + 1, out, stream);
}

/**
 * :cpp:func:`dace::detect_all_positive` on the device: ``*out`` is 1 iff every element is > 0.
 */
template <typename T>
inline gpuError_t detect_all_positive_device(const T *a, long long n, long long *out, gpuStream_t stream) {
    *out = 1;
    if (n <= 0) return gpuSuccess;

    gpuError_t status = gpuSuccess;
    unsigned long long *bad =
        static_cast<unsigned long long *>(
            ::dace::cub::get_scratch< ::dace::cub::DetectFlagTag>(sizeof(unsigned long long), stream, &status));
    if (bad == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;

    long long blocks = (n + DETECT_BLOCK_THREADS - 1) / DETECT_BLOCK_THREADS;
    if (blocks > DETECT_MAX_BLOCKS) blocks = DETECT_MAX_BLOCKS;

    status = gpuMemsetAsync(bad, 0, sizeof(unsigned long long), stream);
    if (status != gpuSuccess) return status;
    detect_all_positive_kernel<<<static_cast<unsigned>(blocks), DETECT_BLOCK_THREADS, 0, stream>>>(a, n, bad);

    unsigned long long host_bad = 0;
    status = gpuMemcpyAsync(&host_bad, bad, sizeof(unsigned long long), gpuMemcpyDeviceToHost, stream);
    if (status != gpuSuccess) return status;
    status = gpuStreamSynchronize(stream);
    *out = static_cast<long long>(host_bad == 0);
    return status;
}

/**
 * :cpp:func:`dace::find_first_index` on the device: ``*out`` is the smallest firing index in
 * ``[begin, end)``, or ``end``.
 *
 * ``pred`` is a device functor taking ``long long``. It is a functor and not a lambda so the
 * caller needs no ``--extended-lambda``: a libnode expansion appends the struct to the device
 * global code next to the wrapper that instantiates this.
 */
template <typename Pred>
inline gpuError_t find_first_index_device(long long begin, long long end, Pred pred, long long *out,
                                          gpuStream_t stream) {
    *out = end;
    if (begin >= end) return gpuSuccess;

    gpuError_t status = gpuSuccess;
    unsigned long long *result =
        static_cast<unsigned long long *>(
            ::dace::cub::get_scratch< ::dace::cub::DetectFlagTag>(sizeof(unsigned long long), stream, &status));
    if (result == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;

    const long long span = end - begin;
    long long blocks = (span + DETECT_BLOCK_THREADS - 1) / DETECT_BLOCK_THREADS;
    if (blocks > DETECT_MAX_BLOCKS) blocks = DETECT_MAX_BLOCKS;

    const unsigned long long sentinel = static_cast<unsigned long long>(end);
    status = gpuMemcpyAsync(result, &sentinel, sizeof(unsigned long long), gpuMemcpyHostToDevice, stream);
    if (status != gpuSuccess) return status;
    find_first_kernel<<<static_cast<unsigned>(blocks), DETECT_BLOCK_THREADS, 0, stream>>>(begin, end, pred, result);

    unsigned long long host_result = sentinel;
    status = gpuMemcpyAsync(&host_result, result, sizeof(unsigned long long), gpuMemcpyDeviceToHost, stream);
    if (status != gpuSuccess) return status;
    status = gpuStreamSynchronize(stream);
    *out = static_cast<long long>(host_result);
    return status;
}

}  // namespace dace

#endif  // __DACE_DETECT_CUH
