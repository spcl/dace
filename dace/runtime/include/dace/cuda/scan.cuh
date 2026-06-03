// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA strided inclusive scan: ``s`` independent inclusive scans, one per
// residue class mod ``s`` over a flat input/output buffer of length ``n``.
// Mirrors the OpenMP ``dace::scan::strided_inclusive_<op>`` family from
// :file:`dace/runtime/include/dace/scan.hpp` but uses the GPU device side.
//
// One thread per residue class: thread ``k`` (``0 <= k < s``) walks
// ``in[k], in[k+s], in[k+2s], ...`` sequentially and writes the running
// accumulator into ``out[k], out[k+s], ...``. The cross-thread memory
// pattern is coalesced when ``s`` is a multiple of the warp size and the
// underlying 2D buffer is C row-major with the scan axis as the slow
// axis (the LoopToScan composite-body rewrite emits buffers in exactly
// that shape, so this is the common case).
//
// Falls back to ``cub::DeviceScan::InclusiveScan`` whenever ``s == 1``; the
// libnode expansion picks the right path.

#ifndef __DACE_CUDA_SCAN_CUH
#define __DACE_CUDA_SCAN_CUH

#include <cuda_runtime.h>
#include <algorithm>

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
inline void strided_inclusive_sum(const T* in, T* out, long n, long s, cudaStream_t stream) {
    if (s <= 0) return;
    dim3 grid = detail::launch_dims(s);
    detail::strided_inclusive_sum_kernel<T><<<grid, dim3(256, 1u, 1u), 0, stream>>>(in, out, n, s);
}

template <typename T>
inline void strided_inclusive_product(const T* in, T* out, long n, long s, cudaStream_t stream) {
    if (s <= 0) return;
    dim3 grid = detail::launch_dims(s);
    detail::strided_inclusive_product_kernel<T><<<grid, dim3(256, 1u, 1u), 0, stream>>>(in, out, n, s);
}

template <typename T>
inline void strided_inclusive_min(const T* in, T* out, long n, long s, cudaStream_t stream) {
    if (s <= 0) return;
    dim3 grid = detail::launch_dims(s);
    detail::strided_inclusive_min_kernel<T><<<grid, dim3(256, 1u, 1u), 0, stream>>>(in, out, n, s);
}

template <typename T>
inline void strided_inclusive_max(const T* in, T* out, long n, long s, cudaStream_t stream) {
    if (s <= 0) return;
    dim3 grid = detail::launch_dims(s);
    detail::strided_inclusive_max_kernel<T><<<grid, dim3(256, 1u, 1u), 0, stream>>>(in, out, n, s);
}

}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_CUH
