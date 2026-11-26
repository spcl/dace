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
#include "scan.cuh"  // the running-prefix carry the block form shares with the scalar ops

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

/// Block-wide ``out[g] = c[g]*out[g-1] + d[g]`` over ``m`` elements from ``out[-1] = seed``, ``c``, ``d`` and
/// ``out`` read or written ``sc``, ``sd`` and ``so`` apart. The carry enters as the constant map ``x -> seed``,
/// so every composed prefix is constant and its ``b`` IS the value; a lane past the end is the identity map.
/// Every thread of the block must call it; it returns after ``__syncthreads()``.
template <typename E, int BLOCK>
__device__ void block_affine_scan(const E* c, long sc, const E* d, long sd, E* out, long so, long m, E seed) {
  typedef affine_map<E> M;
  typedef gpucub::BlockScan<M, BLOCK> BlockScanT;
  __shared__ typename BlockScanT::TempStorage storage;
  affine_compose<E> op;
  ScanRunningPrefix<M, affine_compose<E>> carry(M{static_cast<E>(0), seed}, op);

  for (long base = 0; base < m; base += BLOCK) {
    const long g = base + (long)threadIdx.x;
    M value = (g < m) ? M{c[g * sc], d[g * sd]} : M{static_cast<E>(1), static_cast<E>(0)};
    BlockScanT(storage).InclusiveScan(value, value, op, carry);
    if (g < m) out[g * so] = value.b;
    __syncthreads();  // before the next iteration reuses ``storage``
  }
  __syncthreads();  // the postcondition holds for an empty range too
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

/// The second-order recurrence ``out[k] = c1[k]*out[k-1] + c2[k]*out[k-2] + d[k]`` carries the affine map of the
/// pair ``(out[k-1], out[k-2])``: ``x -> M x + v``, a 2x2 matrix and a 2-vector. Trivially copyable, as cub needs.
template <typename E>
struct linear2_map {
  E m00, m01, m10, m11;
  E v0, v1;
};

/// ``y`` applied AFTER ``x``: ``(My Mx, My vx + vy)``.
template <typename E>
struct linear2_compose {
  __host__ __device__ __forceinline__ linear2_map<E> operator()(const linear2_map<E>& x,
                                                                 const linear2_map<E>& y) const {
    return linear2_map<E>{y.m00 * x.m00 + y.m01 * x.m10, y.m00 * x.m01 + y.m01 * x.m11,
                          y.m10 * x.m00 + y.m11 * x.m10, y.m10 * x.m01 + y.m11 * x.m11,
                          y.m00 * x.v0 + y.m01 * x.v1 + y.v0,  y.m10 * x.v0 + y.m11 * x.v1 + y.v1};
  }
};

namespace detail {

/// Element ``k``'s map ``M = [[c1, c2], [1, 0]], v = [d, 0]``; element 0 absorbs the seeds ``(out[-2], out[-1])``
/// and comes out constant (``M == 0``), so every composed prefix is constant and its ``v0`` IS the value.
template <typename E, typename S>
__global__ void linear2_pack_kernel(const E* __restrict__ c1, const E* __restrict__ c2, const E* __restrict__ d,
                                    linear2_map<E>* __restrict__ m, const S* __restrict__ seed_ptr, E seed0,
                                    E seed1, long long n) {
  const long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (k >= n) return;
  const E zero = static_cast<E>(0);
  if (k == 0) {
    const E s0 = (seed_ptr != nullptr) ? static_cast<E>(seed_ptr[0]) : seed0;
    const E s1 = (seed_ptr != nullptr) ? static_cast<E>(seed_ptr[1]) : seed1;
    m[0] = linear2_map<E>{zero, zero, zero, zero, c1[0] * s1 + c2[0] * s0 + d[0], s1};
  } else {
    m[k] = linear2_map<E>{c1[k], c2[k], static_cast<E>(1), zero, d[k], zero};
  }
}

template <typename E>
__global__ void linear2_unpack_kernel(const linear2_map<E>* __restrict__ m, E* __restrict__ out, long long n) {
  const long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (k < n) out[k] = m[k].v0;
}

}  // namespace detail

/// ``out[k] = c1[k]*out[k-1] + c2[k]*out[k-2] + d[k]`` over ``k in [0, n)`` on ``stream``, entered with
/// ``(out[-2], out[-1])`` from ``seed_ptr`` (device-resident) or, when it is null, ``seed0`` / ``seed1``.
/// One scratch block holds the maps and cub's workspace, as in :func:`inclusive_affine`.
template <typename E, typename S>
inline gpuError_t inclusive_linear2(const E* c1, const E* c2, const E* d, const S* seed_ptr, E seed0, E seed1,
                                    E* out, long long n, gpuStream_t stream) {
  using M = linear2_map<E>;
  if (n <= 0) return gpuSuccess;
  linear2_compose<E> op;
  std::size_t cub_bytes = 0;
  gpuError_t err = ::gpucub::DeviceScan::InclusiveScan(nullptr, cub_bytes, static_cast<M*>(nullptr),
                                                       static_cast<M*>(nullptr), op, n, stream);
  if (err != gpuSuccess) return err;
  const std::size_t map_bytes = ((static_cast<std::size_t>(n) * sizeof(M)) + 255u) & ~static_cast<std::size_t>(255u);
  void* scratch = ::dace::cub::get_scratch<::dace::cub::ScanTag>(map_bytes + cub_bytes, stream, &err);
  if (scratch == nullptr) return (err != gpuSuccess) ? err : gpuErrorMemoryAllocation;
  M* maps = reinterpret_cast<M*>(scratch);
  void* workspace = static_cast<char*>(scratch) + map_bytes;

  const int threads = 256;
  const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
  detail::linear2_pack_kernel<E, S><<<blocks, threads, 0, stream>>>(c1, c2, d, maps, seed_ptr, seed0, seed1, n);
  err = gpuGetLastError();
  if (err != gpuSuccess) return err;
  err = ::gpucub::DeviceScan::InclusiveScan(workspace, cub_bytes, maps, maps, op, n, stream);
  if (err != gpuSuccess) return err;
  detail::linear2_unpack_kernel<E><<<blocks, threads, 0, stream>>>(maps, out, n);
  return gpuGetLastError();
}

/// The linear-fractional recurrence ``out[k] = (p1[k]*out[k-1] + p0[k]) / (q1[k]*out[k-1] + q0[k])``: a cub scan
/// over projective 2x2 matrices, each product rescaled by its largest entry (the same map at any scale).
template <typename E>
struct moebius_map {
  E m00, m01, m10, m11;
};

/// ``y`` applied AFTER ``x``: the product ``My Mx``, rescaled.
template <typename E>
struct moebius_compose {
  __host__ __device__ __forceinline__ moebius_map<E> operator()(const moebius_map<E>& x,
                                                                 const moebius_map<E>& y) const {
    moebius_map<E> m{y.m00 * x.m00 + y.m01 * x.m10, y.m00 * x.m01 + y.m01 * x.m11, y.m10 * x.m00 + y.m11 * x.m10,
                     y.m10 * x.m01 + y.m11 * x.m11};
    E big = fabs(m.m00);
    big = fmax(big, fabs(m.m01));
    big = fmax(big, fabs(m.m10));
    big = fmax(big, fabs(m.m11));
    if (big == static_cast<E>(0)) return m;
    return moebius_map<E>{m.m00 / big, m.m01 / big, m.m10 / big, m.m11 / big};
  }
};

namespace detail {

/// Element ``k``'s matrix; element 0 absorbs the seed and comes out constant (``[[0, v0], [0, v1]]``), so every
/// composed prefix is constant and its value is ``m01 / m11``.
template <typename E, typename S>
__global__ void moebius_pack_kernel(const E* __restrict__ p1, const E* __restrict__ p0, const E* __restrict__ q1,
                                    const E* __restrict__ q0, moebius_map<E>* __restrict__ m,
                                    const S* __restrict__ seed_ptr, E seed_val, long long n) {
  const long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (k >= n) return;
  const E zero = static_cast<E>(0);
  if (k == 0) {
    const E s = (seed_ptr != nullptr) ? static_cast<E>(seed_ptr[0]) : seed_val;
    m[0] = moebius_map<E>{zero, p1[0] * s + p0[0], zero, q1[0] * s + q0[0]};
  } else {
    m[k] = moebius_map<E>{p1[k], p0[k], q1[k], q0[k]};
  }
}

template <typename E>
__global__ void moebius_unpack_kernel(const moebius_map<E>* __restrict__ m, E* __restrict__ out, long long n) {
  const long long k = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
  if (k < n) out[k] = m[k].m01 / m[k].m11;
}

}  // namespace detail

/// ``out[k]`` of the linear-fractional recurrence over ``k in [0, n)`` on ``stream``, entered with ``out[-1]`` from
/// ``seed_ptr`` (device-resident) or, when it is null, ``seed_val``. Scratch as in :func:`inclusive_affine`.
template <typename E, typename S>
inline gpuError_t inclusive_moebius(const E* p1, const E* p0, const E* q1, const E* q0, const S* seed_ptr,
                                    E seed_val, E* out, long long n, gpuStream_t stream) {
  using M = moebius_map<E>;
  if (n <= 0) return gpuSuccess;
  moebius_compose<E> op;
  std::size_t cub_bytes = 0;
  gpuError_t err = ::gpucub::DeviceScan::InclusiveScan(nullptr, cub_bytes, static_cast<M*>(nullptr),
                                                       static_cast<M*>(nullptr), op, n, stream);
  if (err != gpuSuccess) return err;
  const std::size_t map_bytes = ((static_cast<std::size_t>(n) * sizeof(M)) + 255u) & ~static_cast<std::size_t>(255u);
  void* scratch = ::dace::cub::get_scratch<::dace::cub::ScanTag>(map_bytes + cub_bytes, stream, &err);
  if (scratch == nullptr) return (err != gpuSuccess) ? err : gpuErrorMemoryAllocation;
  M* maps = reinterpret_cast<M*>(scratch);
  void* workspace = static_cast<char*>(scratch) + map_bytes;

  const int threads = 256;
  const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
  detail::moebius_pack_kernel<E, S><<<blocks, threads, 0, stream>>>(p1, p0, q1, q0, maps, seed_ptr, seed_val, n);
  err = gpuGetLastError();
  if (err != gpuSuccess) return err;
  err = ::gpucub::DeviceScan::InclusiveScan(workspace, cub_bytes, maps, maps, op, n, stream);
  if (err != gpuSuccess) return err;
  detail::moebius_unpack_kernel<E><<<blocks, threads, 0, stream>>>(maps, out, n);
  return gpuGetLastError();
}

}  // namespace cuda_scan
}  // namespace dace

#endif  // __DACE_CUDA_SCAN_AFFINE_CUH
