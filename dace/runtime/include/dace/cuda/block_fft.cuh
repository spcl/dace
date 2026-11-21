// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// A batch of 1-D DFTs by ONE thread block, for an FFT library node inside a GPU kernel. A power-of-two length runs
// an iterative radix-2 Cooley-Tukey transform in the output (bit-reversed scatter, then log2(n) butterfly stages);
// any other length a direct DFT. Both are out of place and need no scratch.

#pragma once

#include "cudacommon.cuh"

namespace dace {
namespace block_fft {

DACE_DFI void sincospi_of(float x, float* s, float* c) { sincospif(x, s, c); }
DACE_DFI void sincospi_of(double x, double* s, double* c) { sincospi(x, s, c); }

/// ``exp(-+2 pi i r / n)`` for ``0 <= r < n``, the angle folded into ``(-pi, pi]`` before it is rounded.
template <typename C>
DACE_DFI C twiddle(long long r, long long n, bool inverse) {
  using T = typename C::value_type;
  long long twice = 2 * r;
  if (twice > n) twice -= 2 * n;
  T s, c;
  sincospi_of(T(twice) / T(n), &s, &c);
  return C(c, inverse ? s : -s);
}

/// ``out[line(l) + k*os] = scale * sum_j in[in_line(l) + j*is] * w^(jk)`` for ``lines`` lines of length ``n``.
/// Every thread of the block must call it with the same arguments; ``lanes`` threads share the work.
template <typename C, typename In, typename InLine, typename OutLine>
__device__ void transform(const In* __restrict__ in, C* __restrict__ out, long long n, long long lines, long long is,
                          long long os, InLine in_line, OutLine out_line, bool inverse,
                          typename C::value_type scale, long long lanes) {
  using T = typename C::value_type;
  const long long tid = threadIdx.x;
  if (n <= 0 || lines <= 0) return;
  if ((n & (n - 1)) != 0) {
    for (long long t = tid; t < lines * n; t += lanes) {
      const long long l = t / n, k = t % n;
      const In* src = in + in_line(l);
      C acc(T(0));
      long long r = 0;  // j*k mod n, stepped instead of multiplied
      for (long long j = 0; j < n; ++j) {
        acc += C(src[j * is]) * twiddle<C>(r, n, inverse);
        r += k;
        if (r >= n) r -= n;
      }
      out[out_line(l) + k * os] = acc * scale;
    }
    __syncthreads();
    return;
  }
  const int bits = __ffsll(n) - 1;
  for (long long t = tid; t < lines * n; t += lanes) {
    const long long l = t / n, j = t % n;
    const long long k = bits == 0 ? 0 : (long long)(__brevll((unsigned long long)j) >> (64 - bits));
    out[out_line(l) + k * os] = C(in[in_line(l) + j * is]) * scale;
  }
  __syncthreads();
  const long long pairs = n / 2;
  for (long long half = 1; half < n; half *= 2) {
    const long long step = n / (2 * half);
    for (long long t = tid; t < lines * pairs; t += lanes) {
      const long long l = t / pairs, b = t % pairs;
      const long long k = b % half, i = (b / half) * 2 * half + k;
      C* line = out + out_line(l);
      const C u = line[i * os], v = line[(i + half) * os] * twiddle<C>(k * step, n, inverse);
      line[i * os] = u + v;
      line[(i + half) * os] = u - v;
    }
    __syncthreads();
  }
}

}  // namespace block_fft
}  // namespace dace
