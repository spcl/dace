// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA batches of independent inclusive scans over one buffer: the residue classes mod ``s`` of a strided
// scan (``strided_inclusive_<op>``, the device side of ``dace::scan::strided_inclusive_<op>``) and the
// equal consecutive rows of a segmented one (``segmented_inclusive_<op>``). Many interleaved classes take
// one thread each; otherwise every scan is cut into chunks and scanned in three phases -- chunk totals,
// a scan of the totals, and a rescan of every chunk entered with its carry -- so a long scan still
// fills the device.

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

//: ``count`` independent scans over one buffer of ``n`` elements: scan ``k`` reads
//: ``in[k * pitch + g * step]`` for ``g`` below its length. The residue classes of a strided scan are
//: ``{n, s, 1, s, n}``; the rows of a segmented one ``{n, rows, n / rows, 1, n / rows}``.
struct Segments {
  long n;
  long count;
  long pitch;
  long step;
  long cap;  // the longest a scan may be
  __host__ __device__ __forceinline__ long length(long k) const {
    const long first = k * pitch;
    if (first >= n) return 0;
    const long m = (n - first + step - 1) / step;
    return m < cap ? m : cap;
  }
};

//: Scans too long for one block are cut into chunks of ``chunk`` elements and every (scan, chunk) pair
//: gets a BLOCK. Block ``b`` is scan ``b % count`` of chunk ``b / count``: the blocks reading one
//: stretch of memory are adjacent, so the lines one residue class leaves are still in cache when the
//: next class reads them.
struct ChunkCoordinates {
  long offset;  // index of the chunk's first element in the buffer
  long count;   // elements of the scan in the chunk
};

__device__ __forceinline__ ChunkCoordinates chunk_coordinates(const Segments& seg, long chunk) {
  const long k = (long)blockIdx.x % seg.count;
  const long begin = ((long)blockIdx.x / seg.count) * chunk;
  const long m = seg.length(k);
  const long count = (m > begin) ? ((m - begin < chunk) ? (m - begin) : chunk) : 0;
  return ChunkCoordinates{k * seg.pitch + begin * seg.step, count};
}

//: Phase 1: the fold of every (scan, chunk), into ``totals[chunk * count + scan]``.
template <typename T, typename Op, int BLOCK>
__global__ void chunk_totals_kernel(const T* __restrict__ in, T* __restrict__ totals, Segments seg, long chunk, Op op,
                                    T identity) {
  const ChunkCoordinates at = chunk_coordinates(seg, chunk);
  const T* first = in + at.offset;
  T acc = identity;
  for (long g = (long)threadIdx.x; g < at.count; g += BLOCK) acc = op(acc, first[g * seg.step]);
  typedef gpucub::BlockReduce<T, BLOCK> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage storage;
  const T total = BlockReduceT(storage).Reduce(acc, op);
  if (threadIdx.x == 0) totals[blockIdx.x] = total;
}

//: Phase 2 runs ``segment_scan_kernel`` over ``totals``, whose layout makes the chunks of one scan a
//: residue class of their own. Phase 3: rescan every (scan, chunk), entered with the scanned total of
//: the chunks before it.
template <typename T, typename Op, int BLOCK>
__global__ void chunk_scan_kernel(const T* __restrict__ in, T* __restrict__ out, const T* __restrict__ totals,
                                  Segments seg, long chunk, Op op, T identity) {
  const ChunkCoordinates at = chunk_coordinates(seg, chunk);
  const T seed = ((long)blockIdx.x >= seg.count) ? totals[(long)blockIdx.x - seg.count] : identity;
  block_inclusive_scan_strided<T, Op, BLOCK>(in + at.offset, out + at.offset, at.count, seg.step, op, identity, seed);
}

//: One block per scan, each scanned outright.
template <typename T, typename Op, int BLOCK>
__global__ void segment_scan_kernel(const T* __restrict__ in, T* __restrict__ out, Segments seg, Op op, T identity) {
  const long k = (long)blockIdx.x;
  block_inclusive_scan_strided<T, Op, BLOCK>(in + k * seg.pitch, out + k * seg.pitch, seg.length(k), seg.step, op,
                                             identity);
}

//: Below this many residue classes the one-thread-per-class kernel cannot fill the device, and the
//: blocked path takes over. Above it the classes ARE the parallelism and their stride makes the
//: cross-thread access coalesced, which the blocked path gives up. A starting point, to be settled
//: by measurement, not a measured optimum.
constexpr long kBlockedBelow = 4096;
constexpr int kBlockThreads = 256;
//: Elements of one scan one block covers: 16 per thread, enough to amortise a block's launch and its
//: carry, few enough that a scan of 1e8 elements still splits into thousands of blocks.
constexpr long kChunkElements = 16L * kBlockThreads;

//: The ``seg.count`` scans of ``seg`` under ``op`` (``identity`` is ``op``'s identity), on ``stream``.
//: Scan 0 is the longest. Returns the first launch or allocation error.
template <typename T, typename Op>
inline gpuError_t segments_inclusive(const T* in, T* out, Segments seg, Op op, T identity, gpuStream_t stream) {
  if (seg.count <= 0 || seg.n <= 0) return gpuSuccess;
  if (seg.pitch == 1 && seg.count >= kBlockedBelow) {
    // Many interleaved classes: one thread each, neighbouring threads on neighbouring elements.
    const long blocks = (seg.count + kBlockThreads - 1) / kBlockThreads;
    strided_per_class_kernel<T, Op><<<dim3((unsigned)blocks), dim3(kBlockThreads), 0, stream>>>(in, out, seg.n,
                                                                                               seg.step, op);
    return gpuGetLastError();
  }
  const long chunks = (seg.length(0) + kChunkElements - 1) / kChunkElements;
  if (chunks <= 1) {
    segment_scan_kernel<T, Op, kBlockThreads><<<dim3((unsigned)seg.count), dim3(kBlockThreads), 0, stream>>>(
        in, out, seg, op, identity);
    return gpuGetLastError();
  }
  gpuError_t status = gpuSuccess;
  T* totals = static_cast<T*>(::dace::cub::get_scratch<::dace::cub::ScanTag>(
      (std::size_t)(chunks * seg.count) * sizeof(T), stream, &status));
  if (totals == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;
  const dim3 grid((unsigned)(chunks * seg.count));
  chunk_totals_kernel<T, Op, kBlockThreads><<<grid, dim3(kBlockThreads), 0, stream>>>(in, totals, seg, kChunkElements,
                                                                                      op, identity);
  const Segments of_totals{chunks * seg.count, seg.count, 1, seg.count, chunks};
  segment_scan_kernel<T, Op, kBlockThreads><<<dim3((unsigned)seg.count), dim3(kBlockThreads), 0, stream>>>(
      totals, totals, of_totals, op, identity);
  chunk_scan_kernel<T, Op, kBlockThreads><<<grid, dim3(kBlockThreads), 0, stream>>>(in, out, totals, seg,
                                                                                    kChunkElements, op, identity);
  return gpuGetLastError();
}

//: The residue classes of ``in[0:n]`` mod ``s``.
inline Segments residue_classes(long n, long s) { return Segments{n, s, 1, s, n}; }

//: ``rows`` consecutive equal rows of ``in[0:n]``.
inline Segments rows_of(long n, long rows) {
  const long length = rows > 0 ? n / rows : 0;
  return Segments{n, rows, length, 1, length};
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
  return detail::segments_inclusive(in, out, detail::residue_classes(n, s), detail::ScanSum<T>(), T(0), stream);
}

template <typename T>
inline gpuError_t segmented_inclusive_sum(const T* in, T* out, long n, long rows, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::rows_of(n, rows), detail::ScanSum<T>(), T(0), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_product(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::residue_classes(n, s), detail::ScanProduct<T>(), T(1), stream);
}

template <typename T>
inline gpuError_t segmented_inclusive_product(const T* in, T* out, long n, long rows, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::rows_of(n, rows), detail::ScanProduct<T>(), T(1), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_min(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::residue_classes(n, s), detail::ScanMin<T>(), detail::min_identity<T>(), stream);
}

template <typename T>
inline gpuError_t segmented_inclusive_min(const T* in, T* out, long n, long rows, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::rows_of(n, rows), detail::ScanMin<T>(), detail::min_identity<T>(), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_max(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::residue_classes(n, s), detail::ScanMax<T>(), detail::max_identity<T>(), stream);
}

template <typename T>
inline gpuError_t segmented_inclusive_max(const T* in, T* out, long n, long rows, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::rows_of(n, rows), detail::ScanMax<T>(), detail::max_identity<T>(), stream);
}

}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_CUH
