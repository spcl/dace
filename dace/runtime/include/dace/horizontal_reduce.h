// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <algorithm>

// Horizontal reduction of a ``vector_width``-wide array to one scalar, as a balanced pairwise tree: pairs (0,1)(2,3)...
// with an odd last element forwarded, which is the order of ``emit_tree_reduction``.
template <typename T, int vector_width, typename Op>
static inline T horizontal_tree(const T* __restrict__ a, Op op) {
  T buf[vector_width];
  for (int i = 0; i < vector_width; i++) buf[i] = a[i];
  int n = vector_width;
  while (n > 1) {
    int half = n / 2;
    for (int i = 0; i < half; i++) buf[i] = op(buf[2 * i], buf[2 * i + 1]);
    if (n & 1) buf[half] = buf[n - 1];
    n = half + (n & 1);
  }
  return buf[0];
}

template <typename T, int vector_width>
static inline T horizontal_reduce_add(const T* __restrict__ a) {
  return horizontal_tree<T, vector_width>(a, [](T x, T y) { return x + y; });
}
template <typename T, int vector_width>
static inline T horizontal_reduce_mul(const T* __restrict__ a) {
  return horizontal_tree<T, vector_width>(a, [](T x, T y) { return x * y; });
}
template <typename T, int vector_width>
static inline T horizontal_reduce_max(const T* __restrict__ a) {
  return horizontal_tree<T, vector_width>(a, [](T x, T y) { return std::max(x, y); });
}
template <typename T, int vector_width>
static inline T horizontal_reduce_min(const T* __restrict__ a) {
  return horizontal_tree<T, vector_width>(a, [](T x, T y) { return std::min(x, y); });
}
template <typename T, int vector_width>
static inline T horizontal_reduce_band(const T* __restrict__ a) {
  return horizontal_tree<T, vector_width>(a, [](T x, T y) { return x & y; });
}
template <typename T, int vector_width>
static inline T horizontal_reduce_bor(const T* __restrict__ a) {
  return horizontal_tree<T, vector_width>(a, [](T x, T y) { return x | y; });
}
template <typename T, int vector_width>
static inline T horizontal_reduce_bxor(const T* __restrict__ a) {
  return horizontal_tree<T, vector_width>(a, [](T x, T y) { return x ^ y; });
}
