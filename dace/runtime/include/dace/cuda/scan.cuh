// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA batches of independent inclusive scans over one buffer: the residue classes mod ``s`` of a strided
// scan (``strided_inclusive_<op>``, the device side of ``dace::scan::strided_inclusive_<op>``) and the
// equal consecutive rows of a segmented one (``segmented_inclusive_<op>``). A strided scan with at least a
// warp of classes is cut into class tiles by row chunks, loads coalesced across classes; one with very many
// classes takes one thread each. Every other scan is cut into chunks and scanned in three phases -- chunk
// totals, a scan of the totals, and a rescan of every chunk entered with its carry -- so a long scan still
// fills the device.

#ifndef __DACE_CUDA_SCAN_CUH
#define __DACE_CUDA_SCAN_CUH

#include "cudacommon.cuh"  // the backend runtime header, plus the gpu* aliases used below
#include "gpucub.cuh"      // gpucub:: -> hipcub on AMD, cub on NVIDIA
#include "../cub_scratch.cuh"  // the per-stream scratch the chunked path keeps its chunk totals in
#include "scan_plan.h"         // which strided path runs, host-only so it is testable without a device
#include <algorithm>
#include <limits>

namespace dace {
namespace cuda_scan {

namespace detail {

using namespace ::dace::cuda_scan::plan;

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

//: Block-wide collective scan of ``m`` elements, read ``si`` and written ``so`` apart, the carry entering at
//: ``seed``: inclusive ``out[g] = seed op in[0] op ... op in[g]``, exclusive (``EXCLUSIVE``) ``out[g] = seed op
//: in[0] op ... op in[g-1]``. A lane past the end feeds only later lanes and the final chunk's total, which
//: no output reads, so it reads ``seed`` and no operator needs an identity. Every thread of the block must
//: call it; it returns after ``__syncthreads()``.
template <typename T, typename Op, int BLOCK, bool EXCLUSIVE = false>
__device__ void block_scan(const T* in, long si, T* out, long so, long m, Op op, T seed) {
  typedef gpucub::BlockScan<T, BLOCK> BlockScanT;
  __shared__ typename BlockScanT::TempStorage storage;
  ScanRunningPrefix<T, Op> carry(seed, op);

  for (long base = 0; base < m; base += BLOCK) {
    const long g = base + (long)threadIdx.x;
    // Every thread must reach the scan below, in range or not: it carries a barrier.
    T value = (g < m) ? in[g * si] : seed;
    if constexpr (EXCLUSIVE) {
      BlockScanT(storage).ExclusiveScan(value, value, op, carry);
    } else {
      BlockScanT(storage).InclusiveScan(value, value, op, carry);
    }
    if (g < m) out[g * so] = value;
    __syncthreads();  // before the next iteration reuses ``storage``
  }
  // An empty range runs the loop zero times and so reaches no barrier. The caller is entitled to
  // the postcondition regardless, and a second call would otherwise reuse ``storage`` unfenced.
  __syncthreads();
}

//: The inclusive :func:`block_scan` of ``m`` elements spaced ``s`` apart, entered with ``seed`` (``identity``
//: when not given; the padding no longer reads it).
template <typename T, typename Op, int BLOCK>
__device__ void block_inclusive_scan_strided(const T* __restrict__ in, T* __restrict__ out, long m, long s, Op op,
                                             T identity, T seed) {
  block_scan<T, Op, BLOCK>(in, s, out, s, m, op, seed);
}

template <typename T, typename Op, int BLOCK>
__device__ void block_inclusive_scan_strided(const T* __restrict__ in, T* __restrict__ out, long m, long s, Op op,
                                             T identity) {
  block_scan<T, Op, BLOCK>(in, s, out, s, m, op, identity);
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

//: The tiled path's grid: ``tiles`` class tiles by ``chunks`` row chunks of ``chunk_rows`` rows each.
struct TilePlan {
  long rows;
  long tiles;
  long chunks;
  long chunk_rows;
};

inline TilePlan tile_plan(long n, long s) {
  const long rows = (n + s - 1) / s;
  const long tiles = (s + kTileClasses - 1) / kTileClasses;
  const long by_rows = (rows + kTileMinRowsPerThread * kTileRowGroups - 1) / (kTileMinRowsPerThread * kTileRowGroups);
  const long by_device = (kTileTargetBlocks + tiles - 1) / tiles;
  long chunks = by_rows < by_device ? by_rows : by_device;
  if (chunks < 1 || rows < kTileOneChunkBelow) chunks = 1;
  const long chunk_rows = (rows + chunks - 1) / chunks;
  return TilePlan{rows, tiles, (rows + chunk_rows - 1) / chunk_rows, chunk_rows};
}

//: This thread's class and row range in the tiled grid: block ``b`` is tile ``b % tiles`` of chunk ``b / tiles``,
//: and row group ``g`` takes the ``g``-th slice of the chunk's rows.
struct TileSlot {
  long cls;
  long chunk;
  long lo;
  long hi;
};

//: ``hi`` stops at the class's last element, so the row loops are countable; a lane past the classes has none.
__device__ __forceinline__ TileSlot tile_slot(const TilePlan& plan, long n, long s) {
  const long tile = (long)blockIdx.x % plan.tiles;
  const long chunk = (long)blockIdx.x / plan.tiles;
  const long group = (long)threadIdx.x / kTileClasses;
  const long cls = tile * kTileClasses + (long)threadIdx.x % kTileClasses;
  const long begin = chunk * plan.chunk_rows;
  const long end = (begin + plan.chunk_rows < plan.rows) ? begin + plan.chunk_rows : plan.rows;
  const long per = (end - begin + kTileRowGroups - 1) / kTileRowGroups;
  const long lo = (begin + group * per < end) ? begin + group * per : end;
  long hi = (lo + per < end) ? lo + per : end;
  const long own_rows = (cls < s) ? (n - cls + s - 1) / s : 0;
  if (hi > own_rows) hi = own_rows;
  return TileSlot{cls, chunk, lo, hi};
}

template <typename T, typename Op>
__device__ __forceinline__ T tile_fold(const T* __restrict__ in, long s, const TileSlot& at, Op op, T identity) {
  T acc = identity;
#pragma unroll 4
  for (long r = at.lo; r < at.hi; ++r) acc = op(acc, in[r * s + at.cls]);
  return acc;
}

//: Phase 1 of the tiled path: the fold of every (class, chunk), into ``totals[chunk * s + class]``.
template <typename T, typename Op>
__global__ void __launch_bounds__(kTileThreads)
    tile_totals_kernel(const T* __restrict__ in, T* __restrict__ totals, long n, long s, TilePlan plan, Op op,
                       T identity) {
  alignas(T) __shared__ unsigned char partial_bytes[kTileRowGroups * kTileClasses * sizeof(T)];
  T* partial = reinterpret_cast<T*>(partial_bytes);  // raw: a class element type need not be trivially constructible
  const TileSlot at = tile_slot(plan, n, s);
  const int lane = (int)threadIdx.x % kTileClasses;
  partial[threadIdx.x] = tile_fold(in, s, at, op, identity);
  __syncthreads();
  if ((int)threadIdx.x < kTileClasses && at.cls < s) {
    T total = partial[lane];
    for (int g = 1; g < kTileRowGroups; ++g) total = op(total, partial[g * kTileClasses + lane]);
    totals[at.chunk * s + at.cls] = total;
  }
}

//: Phase 2: the exclusive scan of every class's chunk totals, one thread per class, in place.
template <typename T, typename Op>
__global__ void tile_carry_kernel(T* __restrict__ totals, long s, long chunks, Op op, T identity) {
  const long k = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
  if (k >= s) return;
  T carry = identity;
  for (long q = 0; q < chunks; ++q) {
    const T t = totals[q * s + k];
    totals[q * s + k] = carry;
    carry = op(carry, t);
  }
}

//: Phase 3: every row group walks its slice entered with the chunk's carry and the folds of the groups above it.
template <typename T, typename Op>
__global__ void __launch_bounds__(kTileThreads)
    tile_scan_kernel(const T* __restrict__ in, T* __restrict__ out, const T* __restrict__ carries, long n, long s,
                     TilePlan plan, Op op, T identity) {
  alignas(T) __shared__ unsigned char partial_bytes[kTileRowGroups * kTileClasses * sizeof(T)];
  T* partial = reinterpret_cast<T*>(partial_bytes);  // raw: a class element type need not be trivially constructible
  const TileSlot at = tile_slot(plan, n, s);
  const int lane = (int)threadIdx.x % kTileClasses;
  const int group = (int)threadIdx.x / kTileClasses;
  partial[threadIdx.x] = tile_fold(in, s, at, op, identity);
  __syncthreads();
  if (at.cls >= s) return;
  T acc = (carries != nullptr) ? carries[at.chunk * s + at.cls] : identity;
  for (int g = 0; g < group; ++g) acc = op(acc, partial[g * kTileClasses + lane]);
#pragma unroll 4
  for (long r = at.lo; r < at.hi; ++r) {
    acc = op(acc, in[r * s + at.cls]);
    out[r * s + at.cls] = acc;
  }
}

//: The residue classes of ``in[0:n]`` mod ``s`` as a matrix of rows by ``s`` classes, cut into class tiles by row
//: chunks so loads stay coalesced across classes and the chunks fill the device.
template <typename T, typename Op>
inline gpuError_t tiled_inclusive(const T* in, T* out, long n, long s, Op op, T identity, gpuStream_t stream) {
  const TilePlan plan = tile_plan(n, s);
  const dim3 grid((unsigned)(plan.tiles * plan.chunks));
  if (plan.chunks <= 1) {
    tile_scan_kernel<T, Op><<<grid, dim3(kTileThreads), 0, stream>>>(in, out, nullptr, n, s, plan, op, identity);
    return gpuGetLastError();
  }
  gpuError_t status = gpuSuccess;
  T* totals = static_cast<T*>(
      ::dace::cub::get_scratch<::dace::cub::ScanTag>((std::size_t)(plan.chunks * s) * sizeof(T), stream, &status));
  if (totals == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;
  tile_totals_kernel<T, Op><<<grid, dim3(kTileThreads), 0, stream>>>(in, totals, n, s, plan, op, identity);
  tile_carry_kernel<T, Op><<<dim3((unsigned)((s + kBlockThreads - 1) / kBlockThreads)), dim3(kBlockThreads), 0,
                             stream>>>(totals, s, plan.chunks, op, identity);
  tile_scan_kernel<T, Op><<<grid, dim3(kTileThreads), 0, stream>>>(in, out, totals, n, s, plan, op, identity);
  return gpuGetLastError();
}

//: The strided scan of ``in[0:n]`` at stride ``s`` under ``op`` (``identity`` its identity), per
//: :func:`choose_strided_plan`.
template <typename T, typename Op>
inline gpuError_t strided_inclusive(const T* in, T* out, long n, long s, Op op, T identity, gpuStream_t stream) {
  if (n <= 0 || s <= 0) return gpuSuccess;
  switch (choose_strided_plan(s)) {
    case StridedPlan::PER_CLASS: {
      const long classes = s < n ? s : n;
      const long blocks = (classes + kBlockThreads - 1) / kBlockThreads;
      strided_per_class_kernel<T, Op><<<dim3((unsigned)blocks), dim3(kBlockThreads), 0, stream>>>(in, out, n, s, op);
      return gpuGetLastError();
    }
    case StridedPlan::TILED:
      return tiled_inclusive(in, out, n, s, op, identity, stream);
    case StridedPlan::BLOCKED:
      break;
  }
  return segments_inclusive(in, out, residue_classes(n, s), op, identity, stream);
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
inline gpuError_t segmented_inclusive_sum(const T* in, T* out, long n, long rows, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::rows_of(n, rows), detail::ScanSum<T>(), T(0), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_product(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::strided_inclusive(in, out, n, s, detail::ScanProduct<T>(), T(1), stream);
}

template <typename T>
inline gpuError_t segmented_inclusive_product(const T* in, T* out, long n, long rows, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::rows_of(n, rows), detail::ScanProduct<T>(), T(1), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_min(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::strided_inclusive(in, out, n, s, detail::ScanMin<T>(), detail::min_identity<T>(), stream);
}

template <typename T>
inline gpuError_t segmented_inclusive_min(const T* in, T* out, long n, long rows, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::rows_of(n, rows), detail::ScanMin<T>(), detail::min_identity<T>(), stream);
}

template <typename T>
inline gpuError_t strided_inclusive_max(const T* in, T* out, long n, long s, gpuStream_t stream) {
  return detail::strided_inclusive(in, out, n, s, detail::ScanMax<T>(), detail::max_identity<T>(), stream);
}

template <typename T>
inline gpuError_t segmented_inclusive_max(const T* in, T* out, long n, long rows, gpuStream_t stream) {
  return detail::segments_inclusive(in, out, detail::rows_of(n, rows), detail::ScanMax<T>(), detail::max_identity<T>(), stream);
}

}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_CUH
