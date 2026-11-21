// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Dense LAPACK kernels run by ONE thread block on one small matrix, for library nodes placed inside a GPU kernel.
// Neither CUDA nor HIP has a block-level solver, so these are the unblocked textbook algorithms: the lanes of the
// block share each elimination step and meet at a barrier between steps. Matrices are read through their logical
// row and column strides (``a[i * s0 + j * s1]``), the CPU LAPACKE row-major view. Pivots are 1-based, as in LAPACK.
// Every lane must call every function: each one holds barriers. Only ``threadIdx.x`` / ``blockDim.x`` and
// ``__syncthreads`` are used, so a 64-wide wavefront needs nothing different.

#pragma once

#include "cudacommon.cuh"

namespace dace {
namespace block_lapack {

//: Rows a lane holds in registers across a barrier (``getri``), bounding its order at ``ROWS_PER_LANE * blockDim.x``.
constexpr int ROWS_PER_LANE = 4;

template <typename T>
struct Mat {
  T* p;
  long s0, s1;
  __device__ __forceinline__ T& operator()(long i, long j) const { return p[i * s0 + j * s1]; }
};

template <typename T>
__device__ __forceinline__ void swap_values(T& a, T& b) {
  T t = a;
  a = b;
  b = t;
}

//: ``out = in`` for an ``m`` x ``n`` matrix; nothing when they are one buffer.
template <typename T>
__device__ void copy(const T* in, long i0, long i1, Mat<T> out, long m, long n) {
  if (in != out.p) {
    for (long e = threadIdx.x; e < m * n; e += blockDim.x) out(e / n, e % n) = in[(e / n) * i0 + (e % n) * i1];
  }
  __syncthreads();
}

//: LU with partial pivoting, in place (``?getf2``). Returns LAPACK's info: ``k + 1`` for the first zero pivot.
template <typename T>
__device__ int getrf(Mat<T> a, long m, long n, int* ipiv, long sp) {
  int info = 0;
  const long steps = m < n ? m : n;
  for (long k = 0; k < steps; ++k) {
    // Every lane finds the same pivot, the first largest magnitude
    long p = k;
    T best = fabs(a(k, k));
    for (long i = k + 1; i < m; ++i) {
      const T v = fabs(a(i, k));
      if (v > best) {
        best = v;
        p = i;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) ipiv[k * sp] = (int)(p + 1);
    if (best == T(0)) {
      if (info == 0) info = (int)(k + 1);
      continue;
    }
    if (p != k) {
      for (long j = threadIdx.x; j < n; j += blockDim.x) swap_values(a(k, j), a(p, j));
    }
    __syncthreads();
    const T pivot = a(k, k);
    for (long i = k + 1 + threadIdx.x; i < m; i += blockDim.x) a(i, k) = a(i, k) / pivot;
    __syncthreads();
    const long w = n - k - 1;
    for (long e = threadIdx.x; e < (m - k - 1) * w; e += blockDim.x) {
      const long i = k + 1 + e / w, j = k + 1 + e % w;
      a(i, j) = a(i, j) - a(i, k) * a(k, j);
    }
    __syncthreads();
  }
  return info;
}

//: LU without pivoting, in place; a zero pivot yields the infinities the elimination does.
template <typename T>
__device__ void getrf_nopivot(Mat<T> a, long n) {
  for (long k = 0; k < n; ++k) {
    const T pivot = a(k, k);
    for (long i = k + 1 + threadIdx.x; i < n; i += blockDim.x) a(i, k) = a(i, k) / pivot;
    __syncthreads();
    const long w = n - k - 1;
    for (long e = threadIdx.x; e < w * w; e += blockDim.x) {
      const long i = k + 1 + e / w, j = k + 1 + e % w;
      a(i, j) = a(i, j) - a(i, k) * a(k, j);
    }
    __syncthreads();
  }
}

//: ``B = A^-1 B`` from ``getrf``'s factors and pivots (``?getrs``, no transpose); ``B`` is ``n`` x ``nrhs``.
template <typename T>
__device__ void getrs(Mat<const T> a, long n, const int* ipiv, long sp, Mat<T> b, long nrhs) {
  // A lane owns whole columns of B, so the interchanges need no barrier between them
  for (long c = threadIdx.x; c < nrhs; c += blockDim.x) {
    for (long k = 0; k < n; ++k) {
      const long p = ipiv[k * sp] - 1;
      if (p != k) swap_values(b(k, c), b(p, c));
    }
  }
  __syncthreads();
  for (long k = 0; k < n; ++k) {
    const long rows = n - k - 1;
    for (long e = threadIdx.x; e < rows * nrhs; e += blockDim.x) {
      const long i = k + 1 + e / nrhs, c = e % nrhs;
      b(i, c) = b(i, c) - a(i, k) * b(k, c);
    }
    __syncthreads();
  }
  for (long k = n - 1; k >= 0; --k) {
    for (long c = threadIdx.x; c < nrhs; c += blockDim.x) b(k, c) = b(k, c) / a(k, k);
    __syncthreads();
    for (long e = threadIdx.x; e < k * nrhs; e += blockDim.x) {
      const long i = e / nrhs, c = e % nrhs;
      b(i, c) = b(i, c) - a(i, k) * b(k, c);
    }
    __syncthreads();
  }
}

//: ``A^-1`` in place from ``getrf``'s factors and pivots (``?getri``): invert U, solve ``X L = U^-1``, then undo the
//: interchanges as column swaps. Returns ``k + 1`` for a zero diagonal of U, and ``-1`` beyond the register bound.
template <typename T>
__device__ int getri(Mat<T> a, long n, const int* ipiv, long sp) {
  if (n > ROWS_PER_LANE * (long)blockDim.x) return -1;
  for (long k = 0; k < n; ++k) {
    if (a(k, k) == T(0)) return (int)(k + 1);
  }
  __syncthreads();
  T held[ROWS_PER_LANE] = {};
  for (long k = threadIdx.x; k < n; k += blockDim.x) a(k, k) = T(1) / a(k, k);
  __syncthreads();
  // Column j of U^-1 above the diagonal: the inverted leading block times the column, scaled by -U^-1(j, j)
  for (long j = 1; j < n; ++j) {
    const T scale = -a(j, j);
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const long i = threadIdx.x + r * (long)blockDim.x;
      if (i < j) {
        T sum = T(0);
        for (long k = i; k < j; ++k) sum = sum + a(i, k) * a(k, j);
        held[r] = scale * sum;
      }
    }
    __syncthreads();
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const long i = threadIdx.x + r * (long)blockDim.x;
      if (i < j) a(i, j) = held[r];
    }
    __syncthreads();
  }
  // X L = U^-1, right to left: column j reads the unit-lower column j (rows > j) and the finished columns past it
  for (long j = n - 2; j >= 0; --j) {
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const long i = threadIdx.x + r * (long)blockDim.x;
      if (i < n) {
        T sum = i <= j ? a(i, j) : T(0);
        for (long k = j + 1; k < n; ++k) sum = sum - a(i, k) * a(k, j);
        held[r] = sum;
      }
    }
    __syncthreads();
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const long i = threadIdx.x + r * (long)blockDim.x;
      if (i < n) a(i, j) = held[r];
    }
    __syncthreads();
  }
  // A lane owns whole rows, so the column swaps need no barrier between them
  for (long i = threadIdx.x; i < n; i += blockDim.x) {
    for (long j = n - 2; j >= 0; --j) {
      const long p = ipiv[j * sp] - 1;
      if (p != j) swap_values(a(i, j), a(i, p));
    }
  }
  __syncthreads();
  return 0;
}

//: Cholesky, in place on the ``lower`` (``A = L L^T``) or upper (``A = U^T U``) triangle (``?potf2``); the other
//: triangle is not touched. Returns ``j + 1`` for the first non-positive pivot, which is left in place.
template <typename T>
__device__ int potrf(Mat<T> a, long n, bool lower) {
  // Seen through ``l``, the factor is always lower
  Mat<T> l = lower ? a : Mat<T>{a.p, a.s1, a.s0};
  for (long j = 0; j < n; ++j) {
    T d = l(j, j);
    for (long k = 0; k < j; ++k) d = d - l(j, k) * l(j, k);
    __syncthreads();
    if (!(d > T(0))) {
      if (threadIdx.x == 0) l(j, j) = d;
      __syncthreads();
      return (int)(j + 1);
    }
    d = sqrt(d);
    if (threadIdx.x == 0) l(j, j) = d;
    for (long i = j + 1 + threadIdx.x; i < n; i += blockDim.x) {
      T s = l(i, j);
      for (long k = 0; k < j; ++k) s = s - l(i, k) * l(j, k);
      l(i, j) = s / d;
    }
    __syncthreads();
  }
  return 0;
}

//: Eigenvalues (ascending, into ``w``) and eigenvectors (as columns, into ``v``) of the symmetric matrix whose
//: ``lower`` or upper triangle ``a`` holds, by cyclic Jacobi. ``work`` is ``n * n`` scratch, packed; ``a`` may be
//: ``v``. ``w`` or ``v`` may be null.
template <typename T>
__device__ void syevj(const T* a, long a0, long a1, bool lower, T* work, long n, T* w, long sw, Mat<T> v,
                      int max_sweeps, T eps) {
  Mat<T> x{work, n, 1};
  for (long e = threadIdx.x; e < n * n; e += blockDim.x) {
    const long i = e / n, j = e % n;
    const bool stored = lower ? i >= j : i <= j;
    x(i, j) = stored ? a[i * a0 + j * a1] : a[j * a0 + i * a1];
  }
  __syncthreads();
  if (v.p != nullptr) {
    for (long e = threadIdx.x; e < n * n; e += blockDim.x) v(e / n, e % n) = e / n == e % n ? T(1) : T(0);
  }
  T scale = T(0);
  for (long e = 0; e < n * n; ++e) scale = scale + work[e] * work[e];
  __syncthreads();
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    // Every lane sums the same values in the same order, so all take the same branch
    T off = T(0);
    for (long p = 0; p < n; ++p) {
      for (long q = p + 1; q < n; ++q) off = off + x(p, q) * x(p, q);
    }
    if (off <= eps * eps * scale) break;
    for (long p = 0; p < n; ++p) {
      for (long q = p + 1; q < n; ++q) {
        const T apq = x(p, q);
        T c = T(1), s = T(0);
        if (apq != T(0)) {
          const T theta = (x(q, q) - x(p, p)) / (T(2) * apq);
          const T t = (theta < T(0) ? T(-1) : T(1)) / (fabs(theta) + sqrt(theta * theta + T(1)));
          c = T(1) / sqrt(t * t + T(1));
          s = t * c;
        }
        __syncthreads();
        if (apq == T(0)) continue;
        for (long k = threadIdx.x; k < n; k += blockDim.x) {
          const T kp = x(k, p), kq = x(k, q);
          x(k, p) = c * kp - s * kq;
          x(k, q) = s * kp + c * kq;
          if (v.p != nullptr) {
            const T vp = v(k, p), vq = v(k, q);
            v(k, p) = c * vp - s * vq;
            v(k, q) = s * vp + c * vq;
          }
        }
        __syncthreads();
        for (long k = threadIdx.x; k < n; k += blockDim.x) {
          const T pk = x(p, k), qk = x(q, k);
          x(p, k) = c * pk - s * qk;
          x(q, k) = s * pk + c * qk;
        }
        __syncthreads();
      }
    }
  }
  // Selection sort of the diagonal, carrying the eigenvector columns along
  for (long i = 0; i < n; ++i) {
    long smallest = i;
    for (long j = i + 1; j < n; ++j) {
      if (x(j, j) < x(smallest, smallest)) smallest = j;
    }
    __syncthreads();
    if (smallest == i) continue;
    if (threadIdx.x == 0) swap_values(x(i, i), x(smallest, smallest));
    if (v.p != nullptr) {
      for (long k = threadIdx.x; k < n; k += blockDim.x) swap_values(v(k, i), v(k, smallest));
    }
    __syncthreads();
  }
  if (w != nullptr) {
    for (long i = threadIdx.x; i < n; i += blockDim.x) w[i * sw] = x(i, i);
  }
  __syncthreads();
}

}  // namespace block_lapack
}  // namespace dace
