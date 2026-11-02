// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cstdint>

namespace dace {

namespace linalg {

// Single-core out-of-place tensor transpose: ``out[perm(i)] = alpha * in[i] + beta * out[perm(i)]``.
// ``in_strides`` and ``out_strides`` are in elements; ``out_strides[a]`` is the output stride of INPUT axis
// ``a``. Unit axes go, and neighbouring axes the permutation keeps together become one, so a small tensor spends
// its time in a long inner loop rather than in loop control. The axis contiguous in the input and the one contiguous
// in the output are tiled together, so both sides stream through cache lines, the longer one innermost; every other
// axis is walked outermost.
template <typename T, int RANK, int TILE = 32>
inline void transpose_blocked(const T* __restrict__ in, T* __restrict__ out, const int64_t* shape,
                              const int64_t* in_strides, const int64_t* out_strides, T alpha, T beta) {
  int64_t extent[RANK], si[RANK], so[RANK];
  int rank = 0;
  for (int a = 0; a < RANK; ++a) {
    if (shape[a] <= 0) return;
    if (shape[a] == 1) continue;
    if (rank > 0 && si[rank - 1] == in_strides[a] * shape[a] && so[rank - 1] == out_strides[a] * shape[a]) {
      extent[rank - 1] *= shape[a];
      si[rank - 1] = in_strides[a];
      so[rank - 1] = out_strides[a];
    } else {
      extent[rank] = shape[a];
      si[rank] = in_strides[a];
      so[rank] = out_strides[a];
      ++rank;
    }
  }
  if (rank == 0) {
    *out = beta == T(0) ? alpha * *in : alpha * *in + beta * *out;
    return;
  }
  int read_axis = 0, write_axis = 0;
  for (int a = 1; a < rank; ++a) {
    if (si[a] < si[read_axis]) read_axis = a;
    if (so[a] < so[write_axis]) write_axis = a;
  }
  // The inner axis of the tile: a short inner loop pays a mispredicted exit every few elements.
  const bool read_inner = extent[read_axis] > extent[write_axis] && extent[write_axis] < TILE;
  const int outer_tile = read_inner ? write_axis : read_axis, inner_tile = read_inner ? read_axis : write_axis;
  int outer[RANK];
  int n_outer = 0;
  for (int a = 0; a < rank; ++a) {
    if (a != read_axis && a != write_axis) outer[n_outer++] = a;
  }
  int64_t index[RANK] = {0};
  const int64_t nr = extent[outer_tile], nw = extent[inner_tile];
  const int64_t ri = si[outer_tile], ro = so[outer_tile];
  const int64_t wi = si[inner_tile], wo = so[inner_tile];
  const T* src = in;
  T* dst = out;
  while (true) {
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
      const int a = outer[k];
      src += si[a];
      dst += so[a];
      if (++index[a] < extent[a]) break;
      src -= si[a] * extent[a];
      dst -= so[a] * extent[a];
      index[a] = 0;
      --k;
    }
    if (k < 0) return;
  }
}

}  // namespace linalg

}  // namespace dace
