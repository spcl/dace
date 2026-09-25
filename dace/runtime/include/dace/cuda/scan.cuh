// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA strided inclusive scan: ``s`` independent inclusive scans over the residue classes mod ``s``, the
// device side of ``dace::scan::strided_inclusive_<op>``. ``s == 1`` uses ``gpucub::DeviceScan``.

#ifndef __DACE_CUDA_SCAN_CUH
#define __DACE_CUDA_SCAN_CUH

#include "cudacommon.cuh"  // the backend runtime header, plus the gpu* aliases used below
#include "gpucub.cuh"      // gpucub:: -> hipcub on AMD, cub on NVIDIA
#include <algorithm>
#include <limits>

namespace dace {
namespace cuda_scan {

namespace detail {

template <typename T>
__global__ void strided_inclusive_sum_kernel(const T* __restrict__ in, T* __restrict__ out, long n, long s) {
  long k = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
  if (k >= s) return;
  T acc = T(0);
  for (long j = k; j < n; j += s) {
    acc = acc + in[j];
    out[j] = acc;
  }
}

template <typename T>
__global__ void strided_inclusive_product_kernel(const T* __restrict__ in, T* __restrict__ out, long n, long s) {
  long k = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
  if (k >= s) return;
  T acc = T(1);
  for (long j = k; j < n; j += s) {
    acc = acc * in[j];
    out[j] = acc;
  }
}

template <typename T>
__global__ void strided_inclusive_min_kernel(const T* __restrict__ in, T* __restrict__ out, long n, long s) {
  long k = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
  if (k >= s || k >= n) return;
  T acc = in[k];
  out[k] = acc;
  for (long j = k + s; j < n; j += s) {
    acc = (in[j] < acc) ? in[j] : acc;
    out[j] = acc;
  }
}

template <typename T>
__global__ void strided_inclusive_max_kernel(const T* __restrict__ in, T* __restrict__ out, long n, long s) {
  long k = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
  if (k >= s || k >= n) return;
  T acc = in[k];
  out[k] = acc;
  for (long j = k + s; j < n; j += s) {
    acc = (in[j] > acc) ? in[j] : acc;
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

//: Block-wide collective scan of ``m`` elements spaced ``s`` apart. Every thread of the block must call it;
//: it returns after ``__syncthreads()``.
template <typename T, typename Op, int BLOCK>
__device__ void block_inclusive_scan_strided(const T* __restrict__ in, T* __restrict__ out, long m, long s, Op op,
                                             T identity) {
  typedef gpucub::BlockScan<T, BLOCK> BlockScanT;
  __shared__ typename BlockScanT::TempStorage storage;
  ScanRunningPrefix<T, Op> carry(identity, op);

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
__global__ void strided_blocked_kernel(const T* __restrict__ in, T* __restrict__ out, long n, long s, Op op,
                                       T identity) {
  const long k = (long)blockIdx.x;
  if (k >= s) return;  // uniform across the block: no barrier has been reached yet
  // Elements in this class: j = k, k+s, k+2s, ... < n.
  const long m = (n > k) ? ((n - k + s - 1) / s) : 0;
  block_inclusive_scan_strided<T, Op, BLOCK>(in + k, out + k, m, s, op, identity);
}

//: Below this many residue classes the one-thread-per-class kernels cannot fill the device, and the
//: blocked Blelloch path takes over. Above it the classes ARE the parallelism and their stride
//: makes the cross-thread access coalesced, which the blocked path gives up. A starting point, to
//: be settled by measurement, not a measured optimum.
#define DACE_SCAN_BLOCKED_BELOW 4096

inline dim3 launch_dims(long s) {
  // Pick a sensible block size; the kernel is occupancy-limited only when
  // ``s`` is small (single block, partial occupancy). For the LoopToScan
  // composite-body shape (``s = inner_size``, often 1k-100k) the grid is
  // wide enough that block size barely matters.
  const long threads = 256;
  const long blocks = (s + threads - 1) / threads;
  return dim3((unsigned)blocks, 1u, 1u);
}

}  // namespace detail

template <typename T>
inline void strided_inclusive_sum(const T* in, T* out, long n, long s, gpuStream_t stream) {
  if (s <= 0) return;
  if (s < DACE_SCAN_BLOCKED_BELOW) {
    // Too few classes to fill the device one thread each: give every class a BLOCK.
    constexpr int kBlock = 256;
    detail::strided_blocked_kernel<T, detail::ScanSum<T>, kBlock>
        <<<dim3((unsigned)s, 1u, 1u), dim3(kBlock, 1u, 1u), 0, stream>>>(in, out, n, s, detail::ScanSum<T>(), T(0));
    return;
  }
  dim3 grid = detail::launch_dims(s);
  detail::strided_inclusive_sum_kernel<T><<<grid, dim3(256, 1u, 1u), 0, stream>>>(in, out, n, s);
}

template <typename T>
inline void strided_inclusive_product(const T* in, T* out, long n, long s, gpuStream_t stream) {
  if (s <= 0) return;
  if (s < DACE_SCAN_BLOCKED_BELOW) {
    // Too few classes to fill the device one thread each: give every class a BLOCK.
    constexpr int kBlock = 256;
    detail::strided_blocked_kernel<T, detail::ScanProduct<T>, kBlock>
        <<<dim3((unsigned)s, 1u, 1u), dim3(kBlock, 1u, 1u), 0, stream>>>(in, out, n, s, detail::ScanProduct<T>(), T(1));
    return;
  }
  dim3 grid = detail::launch_dims(s);
  detail::strided_inclusive_product_kernel<T><<<grid, dim3(256, 1u, 1u), 0, stream>>>(in, out, n, s);
}

template <typename T>
inline void strided_inclusive_min(const T* in, T* out, long n, long s, gpuStream_t stream) {
  if (s <= 0) return;
  if (s < DACE_SCAN_BLOCKED_BELOW) {
    // Too few classes to fill the device one thread each: give every class a BLOCK.
    constexpr int kBlock = 256;
    detail::strided_blocked_kernel<T, detail::ScanMin<T>, kBlock>
        <<<dim3((unsigned)s, 1u, 1u), dim3(kBlock, 1u, 1u), 0, stream>>>(in, out, n, s, detail::ScanMin<T>(),
                                                                         std::numeric_limits<T>::max());
    return;
  }
  dim3 grid = detail::launch_dims(s);
  detail::strided_inclusive_min_kernel<T><<<grid, dim3(256, 1u, 1u), 0, stream>>>(in, out, n, s);
}

template <typename T>
inline void strided_inclusive_max(const T* in, T* out, long n, long s, gpuStream_t stream) {
  if (s <= 0) return;
  if (s < DACE_SCAN_BLOCKED_BELOW) {
    // Too few classes to fill the device one thread each: give every class a BLOCK.
    constexpr int kBlock = 256;
    detail::strided_blocked_kernel<T, detail::ScanMax<T>, kBlock>
        <<<dim3((unsigned)s, 1u, 1u), dim3(kBlock, 1u, 1u), 0, stream>>>(in, out, n, s, detail::ScanMax<T>(),
                                                                         std::numeric_limits<T>::lowest());
    return;
  }
  dim3 grid = detail::launch_dims(s);
  detail::strided_inclusive_max_kernel<T><<<grid, dim3(256, 1u, 1u), 0, stream>>>(in, out, n, s);
}

}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_CUH
