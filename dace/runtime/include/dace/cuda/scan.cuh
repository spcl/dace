// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA strided inclusive scan: ``s`` independent inclusive scans over the residue classes mod ``s``, the
// device side of ``dace::scan::strided_inclusive_<op>``. Many classes take one thread each; few classes
// (``s == 1`` included) are cut into chunks and scanned in three phases -- chunk totals, a scan of
// the totals, and a rescan of every chunk entered with its carry -- so a long class still fills the
// device.

#ifndef __DACE_CUDA_SCAN_CUH
#define __DACE_CUDA_SCAN_CUH

#include "cudacommon.cuh"  // the backend runtime header, plus the gpu* aliases used below
#include "gpucub.cuh"      // gpucub:: -> hipcub on AMD, cub on NVIDIA
#include "../cub_scratch.cuh"  // the per-stream scratch the chunked path keeps its chunk totals in
#include <algorithm>
#include <limits>

namespace dace {
namespace cuda_scan {

namespace detail {

//: Many residue classes: one thread per class walks it, seeded from its first element so every
//: operator needs no identity. Neighbouring threads read neighbouring classes, so each step of
//: the walk is one coalesced load across the block.
template <typename T, typename Op>
__global__ void strided_per_class_kernel(const T* __restrict__ in, T* __restrict__ out, long n, long s, Op op) {
  const long k = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
  if (k >= s || k >= n) return;
  T acc = in[k];
  out[k] = acc;
  for (long j = k + s; j < n; j += s) {
    acc = op(acc, in[j]);
    out[j] = acc;
  }
}

// Small ``s``: one block per residue class, Blelloch scan within chunks and a running total across chunks.

//: One block per class, and CUB does the in-block scan. Hand-rolling a Blelloch tree in shared
//: memory works (and did, at 48/48 on a host simulation) but CUB's ``BlockScan`` is the tuned
//: version of the same idea: a warp-level scan through shuffle instructions, then a scan across
//: warp totals, then a broadcast add -- no shared-memory round trip for the within-warp part and
//: no bank-conflict padding to get right. ``BlockScanRunningPrefixOp`` is CUB's own name for the
//: carry this loop needs, so the chunk loop below is the documented usage rather than a variation.
template <typename T, typename Op>
struct ScanRunningPrefix {
  T running;
  Op op;
  __device__ __forceinline__ ScanRunningPrefix(T start, Op o) : running(start), op(o) {}
  /// CUB calls this once per chunk, on thread 0, with the chunk's total; it returns the value to
  /// seed that chunk with.
  __device__ __forceinline__ T operator()(T block_aggregate) {
    const T seed = running;
    running = op(running, block_aggregate);
    return seed;
  }
};

template <typename T>
struct ScanSum {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};
template <typename T>
struct ScanProduct {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; }
};
template <typename T>
struct ScanMin {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a < b ? a : b; }
};
template <typename T>
struct ScanMax {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};

//: Block-wide collective scan of ``m`` elements spaced ``s`` apart, entered with ``seed`` folded in
//: front of the first element (``identity`` enters with nothing). Every thread of the block must call
//: it; it returns after ``__syncthreads()``.
template <typename T, typename Op, int BLOCK>
__device__ void block_inclusive_scan_strided(const T* __restrict__ in, T* __restrict__ out, long m, long s, Op op,
                                             T identity, T seed) {
  typedef gpucub::BlockScan<T, BLOCK> BlockScanT;
  __shared__ typename BlockScanT::TempStorage storage;
  ScanRunningPrefix<T, Op> carry(seed, op);

  for (long base = 0; base < m; base += BLOCK) {
    const long g = base + (long)threadIdx.x;
    // Past the end reads the identity, so a short final chunk needs no special case. Every
    // thread must still reach the scan below: it carries a barrier.
    T value = (g < m) ? in[g * s] : identity;
    BlockScanT(storage).InclusiveScan(value, value, op, carry);
    if (g < m) out[g * s] = value;
    __syncthreads();  // before the next iteration reuses ``storage``
  }
  // An empty range runs the loop zero times and so reaches no barrier. The caller is entitled to
  // the postcondition regardless, and a second call would otherwise reuse ``storage`` unfenced.
  __syncthreads();
}

template <typename T, typename Op, int BLOCK>
__device__ void block_inclusive_scan_strided(const T* __restrict__ in, T* __restrict__ out, long m, long s, Op op,
                                             T identity) {
  block_inclusive_scan_strided<T, Op, BLOCK>(in, out, m, s, op, identity, identity);
}

//: Few residue classes: the classes alone cannot fill the device, so each class is cut into chunks
//: of ``chunk`` elements and every (class, chunk) pair gets a BLOCK. Block ``b`` is class
//: ``b % s`` of chunk ``b / s``: the blocks reading one stretch of memory are adjacent, so the
//: lines one class leaves are still in cache when the next class reads them.
struct ChunkCoordinates {
  long k;      // the residue class
  long begin;  // first element of the chunk, counted within the class
  long count;  // elements of the class in the chunk
};

__device__ __forceinline__ ChunkCoordinates chunk_coordinates(long n, long s, long chunk) {
  const long k = (long)blockIdx.x % s;
  const long begin = ((long)blockIdx.x / s) * chunk;
  const long m = (n > k) ? ((n - k + s - 1) / s) : 0;  // elements in the class: k, k+s, ... < n
  const long count = (m > begin) ? ((m - begin < chunk) ? (m - begin) : chunk) : 0;
  return ChunkCoordinates{k, begin, count};
}

//: Phase 1: the fold of every (class, chunk), into ``totals[chunk * s + class]``.
template <typename T, typename Op, int BLOCK>
__global__ void strided_chunk_totals_kernel(const T* __restrict__ in, T* __restrict__ totals, long n, long s,
                                            long chunk, Op op, T identity) {
  const ChunkCoordinates at = chunk_coordinates(n, s, chunk);
  const T* first = in + at.k + at.begin * s;
  T acc = identity;
  for (long g = (long)threadIdx.x; g < at.count; g += BLOCK) acc = op(acc, first[g * s]);
  typedef gpucub::BlockReduce<T, BLOCK> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage storage;
  const T total = BlockReduceT(storage).Reduce(acc, op);
  if (threadIdx.x == 0) totals[blockIdx.x] = total;
}

//: Phase 2 runs ``strided_blocked_kernel`` over ``totals``, whose layout makes the chunks of one class a
//: residue class of their own. Phase 3: rescan every (class, chunk), entered with the scanned total of
//: the chunks before it.
template <typename T, typename Op, int BLOCK>
__global__ void strided_chunk_scan_kernel(const T* __restrict__ in, T* __restrict__ out, const T* __restrict__ totals,
                                          long n, long s, long chunk, Op op, T identity) {
  const ChunkCoordinates at = chunk_coordinates(n, s, chunk);
  const T seed = ((long)blockIdx.x >= s) ? totals[(long)blockIdx.x - s] : identity;
  const long offset = at.k + at.begin * s;
  block_inclusive_scan_strided<T, Op, BLOCK>(in + offset, out + offset, at.count, s, op, identity, seed);
}

template <typename T, typename Op, int BLOCK>
__global__ void strided_blocked_kernel(const T* __restrict__ in, T* __restrict__ out, long n, long s, Op op,
                                       T identity) {
  const long k = (long)blockIdx.x;
  if (k >= s) return;  // uniform across the block: no barrier has been reached yet
  // Elements in this class: j = k, k+s, k+2s, ... < n.
  const long m = (n > k) ? ((n - k + s - 1) / s) : 0;
  block_inclusive_scan_strided<T, Op, BLOCK>(in + k, out + k, m, s, op, identity);
}

//: Below this many residue classes the one-thread-per-class kernel cannot fill the device, and the
//: blocked path takes over. Above it the classes ARE the parallelism and their stride makes the
//: cross-thread access coalesced, which the blocked path gives up. A starting point, to be settled
//: by measurement, not a measured optimum.
constexpr long kBlockedBelow = 4096;
constexpr int kBlockThreads = 256;
//: Elements of one class one block scans: 16 per thread, enough to amortise a block's launch and
//: its carry, few enough that a class of 1e8 elements still splits into thousands of blocks.
constexpr long kChunkElements = 16L * kBlockThreads;

//: The ``s`` residue-class scans of ``in[0:n]`` under ``op`` (``identity`` is ``op``'s identity), on
//: ``stream``. Returns the first launch or allocation error.
template <typename T, typename Op>
inline gpuError_t strided_inclusive(const T* in, T* out, long n, long s, Op op, T identity, gpuStream_t stream) {
  if (s <= 0 || n <= 0) return gpuSuccess;
  if (s >= kBlockedBelow) {
    const long blocks = (s + kBlockThreads - 1) / kBlockThreads;
    strided_per_class_kernel<T, Op><<<dim3((unsigned)blocks), dim3(kBlockThreads), 0, stream>>>(in, out, n, s, op);
    return gpuGetLastError();
  }
  const long longest = (n + s - 1) / s;  // class 0 is the longest
  const long chunks = (longest + kChunkElements - 1) / kChunkElements;
  if (chunks <= 1) {
    // One chunk per class: a block scans its class outright, with nothing to carry in.
    strided_blocked_kernel<T, Op, kBlockThreads><<<dim3((unsigned)s), dim3(kBlockThreads), 0, stream>>>(
        in, out, n, s, op, identity);
    return gpuGetLastError();
  }
  gpuError_t status = gpuSuccess;
  T* totals = static_cast<T*>(::dace::cub::get_scratch<::dace::cub::ScanTag>(
      (std::size_t)(chunks * s) * sizeof(T), stream, &status));
  if (totals == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;
  const dim3 grid((unsigned)(chunks * s));
  strided_chunk_totals_kernel<T, Op, kBlockThreads><<<grid, dim3(kBlockThreads), 0, stream>>>(
      in, totals, n, s, kChunkElements, op, identity);
  strided_blocked_kernel<T, Op, kBlockThreads><<<dim3((unsigned)s), dim3(kBlockThreads), 0, stream>>>(
      totals, totals, chunks * s, s, op, identity);
  strided_chunk_scan_kernel<T, Op, kBlockThreads><<<grid, dim3(kBlockThreads), 0, stream>>>(
      in, out, totals, n, s, kChunkElements, op, identity);
  return gpuGetLastError();
}

//: The identity of ``min``: infinity where the type has one, so an infinite element is not
//: replaced by the largest finite value it is compared against.
template <typename T>
constexpr T min_identity() {
  return std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
}

template <typename T>
constexpr T max_identity() {
  return std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity()
                                              : std::numeric_limits<T>::lowest();
}

}  // namespace detail

template <typename T>
inline gpuError_t strided_inclusive_sum(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::strided_inclusive(in, out, n, s, detail::ScanSum<T>(), T(0), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_product(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::strided_inclusive(in, out, n, s, detail::ScanProduct<T>(), T(1), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_min(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::strided_inclusive(in, out, n, s, detail::ScanMin<T>(), detail::min_identity<T>(), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_max(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::strided_inclusive(in, out, n, s, detail::ScanMax<T>(), detail::max_identity<T>(), stream);
}

}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_CUH
