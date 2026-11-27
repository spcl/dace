// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_DETECT_CUH
#define __DACE_DETECT_CUH

// Device versions of dace/detect.h: duplicate detection, all-positive check, find-first. Include from the .cu
// translation unit. Spell CUB ``::gpucub::``: inside ``dace`` a bare ``gpucub::`` resolves to ``dace::cub``.

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
  __device__ __forceinline__ int operator()(const int& a, const int& b) const { return a | b; }
};

//: Minimum over candidate indices: the block-level fold of the find-first argmin.
struct MinOp {
  __device__ __forceinline__ long long operator()(const long long& a, const long long& b) const {
    return a < b ? a : b;
  }
};

//: Maximum over index values: the block-level fold that sizes a self-sized tag buffer.
struct MaxOp {
  __device__ __forceinline__ long long operator()(const long long& a, const long long& b) const {
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
__global__ void detect_tag_kernel(const T* idx, long long n, TagT* owner, long long capacity) {
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
__global__ void detect_verify_kernel(const T* idx, long long n, const TagT* owner, long long capacity,
                                     unsigned long long* flag) {
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
__global__ void detect_all_positive_kernel(const T* a, long long n, unsigned long long* bad) {
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
 * Find-first as a cancelling block-per-tile argmin. The published answer only decreases and is always a real
 * firing index, so a stale read never skips the true minimum.
 */
template <typename Pred>
__global__ void find_first_kernel(long long begin, long long end, Pred pred, unsigned long long* result) {
  typedef ::gpucub::BlockReduce<long long, DETECT_BLOCK_THREADS, ::gpucub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockMin;
  __shared__ typename BlockMin::TempStorage tmp;
  __shared__ long long stop;

  const long long tile = static_cast<long long>(blockDim.x);
  const long long stride = tile * static_cast<long long>(gridDim.x);
  for (long long base = begin + static_cast<long long>(blockIdx.x) * tile; base < end; base += stride) {
    if (threadIdx.x == 0) stop = static_cast<long long>(*static_cast<volatile unsigned long long*>(result));
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
inline gpuError_t detect_collision_device(const T* idx, long long n, TagT* owner, long long capacity, int64_t* out,
                                          gpuStream_t stream) {
  *out = 0;
  if (n <= 0) return gpuSuccess;

  gpuError_t status = gpuSuccess;
  unsigned long long* flag = static_cast<unsigned long long*>(
      ::dace::cub::get_scratch<::dace::cub::DetectFlagTag>(sizeof(unsigned long long), stream, &status));
  if (flag == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;

  long long blocks = (n + DETECT_BLOCK_THREADS - 1) / DETECT_BLOCK_THREADS;
  if (blocks > DETECT_MAX_BLOCKS) blocks = DETECT_MAX_BLOCKS;

  status = gpuMemsetAsync(flag, 0, sizeof(unsigned long long), stream);
  if (status != gpuSuccess) return status;
  detect_tag_kernel<<<static_cast<unsigned>(blocks), DETECT_BLOCK_THREADS, 0, stream>>>(idx, n, owner, capacity);
  detect_verify_kernel<<<static_cast<unsigned>(blocks), DETECT_BLOCK_THREADS, 0, stream>>>(idx, n, owner, capacity,
                                                                                           flag);

  unsigned long long host_flag = 0;
  status = gpuMemcpyAsync(&host_flag, flag, sizeof(unsigned long long), gpuMemcpyDeviceToHost, stream);
  if (status != gpuSuccess) return status;
  status = gpuStreamSynchronize(stream);
  *out = static_cast<int64_t>(host_flag != 0);
  return status;
}

/**
 * Open addressing over a power-of-two table of at least ``2n`` slots, keyed by value: a lane that finds its value
 * already in the table sets ``flag``. A slot holding ``~0`` is empty, which no non-negative value can be; negative
 * values are skipped, as the capacity check skips them.
 */
template <typename T>
__global__ void detect_hash_kernel(const T* idx, long long n, unsigned long long* keys, unsigned long long mask,
                                   unsigned long long* flag) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const long long v = static_cast<long long>(idx[i]);
    if (v < 0) continue;
    const unsigned long long key = static_cast<unsigned long long>(v);
    unsigned long long slot = (key * 0x9E3779B97F4A7C15ull) & mask;
    while (true) {
      const unsigned long long seen = atomicCAS(&keys[slot], ~0ull, key);
      if (seen == ~0ull) break;
      if (seen == key) {
        *flag = 1ull;
        break;
      }
      slot = (slot + 1) & mask;
    }
  }
}

/**
 * :cpp:func:`dace::detect_collision` on the device, sizing its own tag buffer.
 *
 * The buffer comes from the persistent per-stream CUB scratch pool, so the repeat call pays no
 * allocation; ``capacity`` still has to be known, since sizing it from ``max(idx)`` would cost a
 * device reduction plus a round trip before the check can even start.
 */
template <typename T>
inline gpuError_t detect_collision_device(const T* idx, long long n, long long capacity, int64_t* out,
                                          gpuStream_t stream) {
  gpuError_t status = gpuSuccess;
  long long* owner = static_cast<long long*>(::dace::cub::get_scratch<::dace::cub::DetectOwnerTag>(
      static_cast<size_t>(capacity) * sizeof(long long), stream, &status));
  if (owner == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;
  return detect_collision_device(idx, n, owner, capacity, out, stream);
}

/**
 * :cpp:func:`dace::detect_collision` on the device when the scattered array's extent is unknown.
 *
 * A hash table of the values in one pass: its size follows ``n``, not the values, so an index array of huge or
 * uninitialized values cannot make the check allocate past the device. Prefer the overload taking a capacity wherever
 * the scattered array's domain is known.
 */
template <typename T>
inline gpuError_t detect_collision_device(const T* idx, long long n, int64_t* out, gpuStream_t stream) {
  *out = 0;
  if (n <= 0) return gpuSuccess;

  unsigned long long slots = 1;
  while (slots < 2ull * static_cast<unsigned long long>(n)) slots <<= 1;
  gpuError_t status = gpuSuccess;
  unsigned long long* keys = static_cast<unsigned long long*>(::dace::cub::get_scratch<::dace::cub::DetectOwnerTag>(
      slots * sizeof(unsigned long long), stream, &status));
  if (keys == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;
  unsigned long long* flag = static_cast<unsigned long long*>(
      ::dace::cub::get_scratch<::dace::cub::DetectFlagTag>(sizeof(unsigned long long), stream, &status));
  if (flag == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;

  long long blocks = (n + DETECT_BLOCK_THREADS - 1) / DETECT_BLOCK_THREADS;
  if (blocks > DETECT_MAX_BLOCKS) blocks = DETECT_MAX_BLOCKS;

  status = gpuMemsetAsync(keys, 0xFF, slots * sizeof(unsigned long long), stream);
  if (status != gpuSuccess) return status;
  status = gpuMemsetAsync(flag, 0, sizeof(unsigned long long), stream);
  if (status != gpuSuccess) return status;
  detect_hash_kernel<<<static_cast<unsigned>(blocks), DETECT_BLOCK_THREADS, 0, stream>>>(idx, n, keys, slots - 1,
                                                                                         flag);

  unsigned long long host_flag = 0;
  status = gpuMemcpyAsync(&host_flag, flag, sizeof(unsigned long long), gpuMemcpyDeviceToHost, stream);
  if (status != gpuSuccess) return status;
  status = gpuStreamSynchronize(stream);
  *out = static_cast<int64_t>(host_flag != 0);
  return status;
}

/**
 * :cpp:func:`dace::detect_all_positive` on the device: ``*out`` is 1 iff every element is > 0.
 */
template <typename T>
inline gpuError_t detect_all_positive_device(const T* a, long long n, int64_t* out, gpuStream_t stream) {
  *out = 1;
  if (n <= 0) return gpuSuccess;

  gpuError_t status = gpuSuccess;
  unsigned long long* bad = static_cast<unsigned long long*>(
      ::dace::cub::get_scratch<::dace::cub::DetectFlagTag>(sizeof(unsigned long long), stream, &status));
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
  *out = static_cast<int64_t>(host_bad == 0);
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
inline gpuError_t find_first_index_device(long long begin, long long end, Pred pred, int64_t* out,
                                          gpuStream_t stream) {
  *out = end;
  if (begin >= end) return gpuSuccess;

  gpuError_t status = gpuSuccess;
  unsigned long long* result = static_cast<unsigned long long*>(
      ::dace::cub::get_scratch<::dace::cub::DetectFlagTag>(sizeof(unsigned long long), stream, &status));
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
