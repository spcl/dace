// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA recurrence ``out[k] = c[k]*out[k-s] + d[k]`` from ``out[r-s] = seed[r]``, as a prefix scan over affine maps.
// The seed folds into each residue class's first element, so no coefficient product spans the whole prefix and
// overflows. The maps are laid out class by class (``k = r + j*s`` at ``r*len + j``); a class's first map is
// constant, so the one scan restarts at every class boundary and computes all ``s`` chains.

#ifndef __DACE_CUDA_SCAN_AFFINE_CUH
#define __DACE_CUDA_SCAN_AFFINE_CUH

#include "cudacommon.cuh"  // the backend runtime header, plus the gpu* aliases used below
#include "gpucub.cuh"      // ::gpucub -- cub on CUDA, hipCUB on HIP

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

/// The map of element ``k = r + j*stride`` at ``r*len + j``: ``{c[k], d[k]}``, except a class's first, which
/// absorbs the seed and comes out constant, and a slot past ``n``, which is the identity.
///
/// ``seed_ptr`` wins when it is non-null; that is the device-resident seed (one per class), which host code
/// issuing the launch must not dereference. A host-readable seed arrives in ``seed_val``.
template <typename E, typename C, typename D, typename S>
__global__ void affine_pack_kernel(const C* __restrict__ c, const D* __restrict__ d, affine_map<E>* __restrict__ m,
                                   const S* __restrict__ seed_ptr, E seed_val, long long n, long long stride,
                                   long long len) {
  long long slot = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (slot >= stride * len) return;
  const long long r = slot / len, j = slot % len, k = r + j * stride;
  if (k >= n) {
    m[slot] = affine_map<E>{static_cast<E>(1), static_cast<E>(0)};
    return;
  }
  const E ck = static_cast<E>(c[k]);
  const E dk = static_cast<E>(d[k]);
  if (j == 0) {
    const E s = (seed_ptr != nullptr) ? static_cast<E>(seed_ptr[r]) : seed_val;
    m[slot] = affine_map<E>{static_cast<E>(0), ck * s + dk};
  } else {
    m[slot] = affine_map<E>{ck, dk};
  }
}

/// Every composed prefix is constant (``a == 0``), so ``b`` IS the recurrence's value.
template <typename E>
__global__ void affine_unpack_kernel(const affine_map<E>* __restrict__ m, E* __restrict__ out, long long n,
                                     long long stride, long long len) {
  long long slot = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (slot >= stride * len) return;
  const long long k = slot / len + (slot % len) * stride;
  if (k < n) out[k] = m[slot].b;
}

}  // namespace detail

/// ``out[k] = c[k]*out[k-stride] + d[k]`` over ``k in [0, n)``, on ``stream``.
///
/// The map buffer and cub's workspace come from ONE block of the ``ScanTag`` scratch pool, laid
/// out maps-first; the scan runs in place over the maps, which ``gpucub::DeviceScan`` supports.
template <typename E, typename C, typename D, typename S>
inline gpuError_t inclusive_affine(const C* coef, const D* delta, const S* seed_ptr, E seed_val, E* out, long long n,
                                   long long stride, gpuStream_t stream) {
  using M = affine_map<E>;
  if (n <= 0) return gpuSuccess;
  if (stride > n) stride = n;
  const long long len = (n + stride - 1) / stride;
  const long long slots = stride * len;

  affine_compose<E> op;
  std::size_t cub_bytes = 0;
  gpuError_t err = ::gpucub::DeviceScan::InclusiveScan(nullptr, cub_bytes, static_cast<M*>(nullptr),
                                                       static_cast<M*>(nullptr), op, slots, stream);
  if (err != gpuSuccess) return err;

  // 256-byte alignment for the workspace that follows: cub's temporary layout assumes an
  // allocation at least as aligned as gpuMalloc's, and the pool hands back exactly that.
  const std::size_t map_bytes =
      ((static_cast<std::size_t>(slots) * sizeof(M)) + 255u) & ~static_cast<std::size_t>(255u);
  void* scratch = ::dace::cub::get_scratch<::dace::cub::ScanTag>(map_bytes + cub_bytes, stream, &err);
  if (scratch == nullptr) return (err != gpuSuccess) ? err : gpuErrorMemoryAllocation;
  M* maps = reinterpret_cast<M*>(scratch);
  void* workspace = static_cast<char*>(scratch) + map_bytes;

  const int threads = 256;
  const unsigned blocks = static_cast<unsigned>((slots + threads - 1) / threads);
  detail::affine_pack_kernel<E, C, D, S>
      <<<blocks, threads, 0, stream>>>(coef, delta, maps, seed_ptr, seed_val, n, stride, len);
  err = gpuGetLastError();
  if (err != gpuSuccess) return err;

  err = ::gpucub::DeviceScan::InclusiveScan(workspace, cub_bytes, maps, maps, op, slots, stream);
  if (err != gpuSuccess) return err;

  detail::affine_unpack_kernel<E><<<blocks, threads, 0, stream>>>(maps, out, n, stride, len);
  return gpuGetLastError();
}

}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_AFFINE_CUH
