// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA first-order linear recurrence: ``out[k] = c[k]*out[k-1] + d[k]``, entered at
// ``out[-1] = seed``. The device counterpart of ``dace::scan::inclusive_affine`` in
// :file:`dace/runtime/include/dace/scan.hpp`, and the same monoid: the carry is the affine MAP
// ``x -> a*x + b`` rather than a value, and map composition is associative, so a plain prefix scan
// over the maps computes the recurrence.
//
// The seed is folded into element 0 rather than passed to cub as an init value, and that choice is
// numerically load-bearing rather than a convenience. Element 0's map comes out as the CONSTANT
// map ``{0, c[0]*seed + d[0]}``, and composition multiplies the accumulated ``a`` by the left
// operand's ``a``, so every prefix that includes element 0 carries ``a == 0``. The coefficient
// product therefore never spans more than the segment cub composes before it applies a block
// prefix -- the same bound ``fold_affine`` gives the host lowering, and the reason neither forms
// the whole-prefix product that the closed form ``out[k] = P[k]*(seed + sum_j d[j]/P[j])`` loses to
// overflow.

#ifndef __DACE_CUDA_SCAN_AFFINE_CUH
#define __DACE_CUDA_SCAN_AFFINE_CUH

#include "cudacommon.cuh"  // the backend runtime header, plus the gpu* aliases used below
#include "gpucub.cuh"     // ::gpucub -- cub on CUDA, hipCUB on HIP

#include "../cub_scratch.cuh"

namespace dace {
namespace cuda_scan {

/// The affine map ``x -> a*x + b``. Trivially copyable, which is what cub requires of a scan
/// element type.
template <typename E>
struct affine_map {
    E a;
    E b;
};

/// Compose two affine maps: ``y`` applied AFTER ``x``, matching the host header's argument order
/// and cub's (accumulated prefix on the left, next element on the right).
template <typename E>
struct affine_compose {
    __device__ __forceinline__ affine_map<E> operator()(const affine_map<E>& x, const affine_map<E>& y) const {
        return affine_map<E>{y.a * x.a, y.a * x.b + y.b};
    }
};

namespace detail {

/// ``m[k] = {c[k], d[k]}``, except ``m[0]``, which absorbs the seed and comes out constant.
///
/// ``seed_ptr`` wins when it is non-null; that is the device-resident seed, which host code
/// issuing the launch must not dereference. A host-readable seed arrives in ``seed_val``.
template <typename E, typename C, typename D, typename S>
__global__ void affine_pack_kernel(const C* __restrict__ c, const D* __restrict__ d, affine_map<E>* __restrict__ m,
                                   const S* __restrict__ seed_ptr, E seed_val, long long n) {
    long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (k >= n) return;
    const E ck = static_cast<E>(c[k]);
    const E dk = static_cast<E>(d[k]);
    if (k == 0) {
        const E s = (seed_ptr != nullptr) ? static_cast<E>(*seed_ptr) : seed_val;
        m[0] = affine_map<E>{static_cast<E>(0), ck * s + dk};
    } else {
        m[k] = affine_map<E>{ck, dk};
    }
}

/// Every composed prefix is constant (``a == 0``), so ``b`` IS the recurrence's value.
template <typename E>
__global__ void affine_unpack_kernel(const affine_map<E>* __restrict__ m, E* __restrict__ out, long long n) {
    long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (k < n) out[k] = m[k].b;
}

}  // namespace detail

/// ``out[k] = c[k]*out[k-1] + d[k]`` over ``k in [0, n)``, on ``stream``.
///
/// The map buffer and cub's workspace come from ONE block of the ``ScanTag`` scratch pool, laid
/// out maps-first; the scan runs in place over the maps, which ``gpucub::DeviceScan`` supports.
template <typename E, typename C, typename D, typename S>
inline gpuError_t inclusive_affine(const C* coef, const D* delta, const S* seed_ptr, E seed_val, E* out, long long n,
                                    gpuStream_t stream) {
    using M = affine_map<E>;
    if (n <= 0) return gpuSuccess;

    affine_compose<E> op;
    std::size_t cub_bytes = 0;
    gpuError_t err = ::gpucub::DeviceScan::InclusiveScan(nullptr, cub_bytes, static_cast<M*>(nullptr),
                                                       static_cast<M*>(nullptr), op, n, stream);
    if (err != gpuSuccess) return err;

    // 256-byte alignment for the workspace that follows: cub's temporary layout assumes an
    // allocation at least as aligned as gpuMalloc's, and the pool hands back exactly that.
    const std::size_t map_bytes = ((static_cast<std::size_t>(n) * sizeof(M)) + 255u) & ~static_cast<std::size_t>(255u);
    void* scratch = ::dace::cub::get_scratch< ::dace::cub::ScanTag>(map_bytes + cub_bytes, stream, &err);
    if (scratch == nullptr) return (err != gpuSuccess) ? err : gpuErrorMemoryAllocation;
    M* maps = reinterpret_cast<M*>(scratch);
    void* workspace = static_cast<char*>(scratch) + map_bytes;

    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
    detail::affine_pack_kernel<E, C, D, S><<<blocks, threads, 0, stream>>>(coef, delta, maps, seed_ptr, seed_val, n);
    err = gpuGetLastError();
    if (err != gpuSuccess) return err;

    err = ::gpucub::DeviceScan::InclusiveScan(workspace, cub_bytes, maps, maps, op, n, stream);
    if (err != gpuSuccess) return err;

    detail::affine_unpack_kernel<E><<<blocks, threads, 0, stream>>>(maps, out, n);
    return gpuGetLastError();
}

}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_AFFINE_CUH
