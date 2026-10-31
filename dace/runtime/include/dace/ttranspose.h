// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cstdint>

namespace dace {

namespace linalg {

// Single-core out-of-place tensor transpose: ``out[perm(i)] = alpha * in[i] + beta * out[perm(i)]``.
// ``in_strides`` and ``out_strides`` are in elements; ``out_strides[a]`` is the output stride of INPUT axis
// ``a``. The axis contiguous in the input and the one contiguous in the output are tiled together, so both
// sides stream through cache lines; every other axis is walked outermost.
template <typename T, int RANK, int TILE = 32>
inline void transpose_blocked(const T* __restrict__ in, T* __restrict__ out, const int64_t* shape,
                              const int64_t* in_strides, const int64_t* out_strides, T alpha, T beta) {
  int read_axis = 0, write_axis = 0;
  for (int a = 1; a < RANK; ++a) {
    if (in_strides[a] < in_strides[read_axis]) read_axis = a;
    if (out_strides[a] < out_strides[write_axis]) write_axis = a;
  }
  for (int a = 0; a < RANK; ++a) {
    if (shape[a] <= 0) return;
  }
  int outer[RANK];
  int n_outer = 0;
  for (int a = 0; a < RANK; ++a) {
    if (a != read_axis && a != write_axis) outer[n_outer++] = a;
  }
  int64_t index[RANK] = {0};
  const int64_t nr = shape[read_axis], nw = shape[write_axis];
  const int64_t ri = in_strides[read_axis], ro = out_strides[read_axis];
  const int64_t wi = in_strides[write_axis], wo = out_strides[write_axis];
  while (true) {
    int64_t in_base = 0, out_base = 0;
    for (int k = 0; k < n_outer; ++k) {
      in_base += index[outer[k]] * in_strides[outer[k]];
      out_base += index[outer[k]] * out_strides[outer[k]];
    }
    const T* src = in + in_base;
    T* dst = out + out_base;
    if (read_axis == write_axis) {
      if (beta == T(0)) {
        for (int64_t r = 0; r < nr; ++r) dst[r * ro] = alpha * src[r * ri];
      } else {
        for (int64_t r = 0; r < nr; ++r) dst[r * ro] = alpha * src[r * ri] + beta * dst[r * ro];
      }
    } else {
      for (int64_t w0 = 0; w0 < nw; w0 += TILE) {
        const int64_t w1 = w0 + TILE < nw ? w0 + TILE : nw;
        for (int64_t r0 = 0; r0 < nr; r0 += TILE) {
          const int64_t r1 = r0 + TILE < nr ? r0 + TILE : nr;
          for (int64_t r = r0; r < r1; ++r) {
            const T* s = src + r * ri;
            T* d = dst + r * ro;
            if (beta == T(0)) {
              for (int64_t w = w0; w < w1; ++w) d[w * wo] = alpha * s[w * wi];
            } else {
              for (int64_t w = w0; w < w1; ++w) d[w * wo] = alpha * s[w * wi] + beta * d[w * wo];
            }
          }
        }
      }
    }
    // Odometer over the outer axes, the last one fastest.
    int k = n_outer - 1;
    while (k >= 0) {
      if (++index[outer[k]] < shape[outer[k]]) break;
      index[outer[k]] = 0;
      --k;
    }
    if (k < 0) return;
  }
}

}  // namespace linalg

}  // namespace dace
