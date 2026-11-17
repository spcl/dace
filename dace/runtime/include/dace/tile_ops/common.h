// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// What the CPU backends of the K=1 tile-op intrinsics (scalar.h, avx2.h, avx512.h, arm_neon.h, arm_sve.h) share: the
// per-lane reference ops, which every scalar tail and fallback path calls so that a backend is bit-for-bit the scalar
// contract, and the loops no backend vectorizes by hand.
//
// Op codes (binop only): + - * / % (C modulo) p (Python modulo) m/M min/max, < l > g = !
// comparisons (yield T(1)/T(0)), & | logical. Unop op codes: n neg, ! not, a abs, e exp, l log, s sqrt, S sin, C cos,
// f floor, c ceil, t tanh. Producers zero-fill inactive lanes and guard the read; array writers (store/scatter) skip
// inactive lanes instead (read-modify-write).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#define STRINGIZE(x) STRINGIZE_IMPL(x)
#define STRINGIZE_IMPL(x) #x
#if defined(__clang__)
#define _dace_tile_vectorize(width) _Pragma(STRINGIZE(clang loop vectorize(enable)))
#else
#define _dace_tile_vectorize(width) _Pragma(STRINGIZE(omp simd))
#endif

namespace dace {
namespace tileops {

// Per-lane binary op.
template <typename T, char Op>
inline T tile_apply(T a, T b) {
  if constexpr (Op == '+')
    return a + b;
  else if constexpr (Op == '-')
    return a - b;
  else if constexpr (Op == '*')
    return a * b;
  else if constexpr (Op == '/')
    return a / b;
  else if constexpr (Op == '%')
    return c_mod(a, b);
  else if constexpr (Op == 'p')
    return py_mod(a, b);
  else if constexpr (Op == 'm')
    return std::min(a, b);
  else if constexpr (Op == 'M')
    return std::max(a, b);
  else if constexpr (Op == '<')
    return (a < b) ? T(1) : T(0);
  else if constexpr (Op == 'l')
    return (a <= b) ? T(1) : T(0);
  else if constexpr (Op == '>')
    return (a > b) ? T(1) : T(0);
  else if constexpr (Op == 'g')
    return (a >= b) ? T(1) : T(0);
  else if constexpr (Op == '=')
    return (a == b) ? T(1) : T(0);
  else if constexpr (Op == '!')
    return (a != b) ? T(1) : T(0);
  else if constexpr (Op == '&')
    return (a && b) ? T(1) : T(0);
  else /* '|' */
    return (a || b) ? T(1) : T(0);
}

// Per-lane unary op. Op codes: n neg, ! not, a abs, e exp, l log, s sqrt, S sin, C cos,
// f floor, c ceil, t tanh.
template <typename T, char Op>
inline T tile_unop_apply(T a) {
  if constexpr (Op == 'n')
    return -a;
  else if constexpr (Op == '!')
    return T(!a);
  else if constexpr (Op == 'a')
    return std::abs(a);
  else if constexpr (Op == 'e')
    return std::exp(a);
  else if constexpr (Op == 'l')
    return std::log(a);
  else if constexpr (Op == 's')
    return std::sqrt(a);
  else if constexpr (Op == 'S')
    return std::sin(a);
  else if constexpr (Op == 'C')
    return std::cos(a);
  else if constexpr (Op == 'f')
    return std::floor(a);
  else if constexpr (Op == 'c')
    return std::ceil(a);
  else /* 't' */
    return std::tanh(a);
}

// tile_unop
// out[i] = <op> a-operand ; ZERO-FILL inactive (operand read is in-tile, safe
// to evaluate even on an inactive lane).
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked>
inline void tile_unop(T* __restrict__ out, const T* __restrict__ a, const bool* __restrict__ mask) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    const T av = Broadcast ? a[0] : a[i];
    if constexpr (Masked)
      out[i] = mask[i] ? tile_unop_apply<T, Op>(av) : T(0);
    else
      out[i] = tile_unop_apply<T, Op>(av);
  }
}

// tile_reduce: horizontal reduction of a VLEN-lane tile to one scalar (Op: '+' sum, '*' prod,
// 'm' min, 'M' max). Balanced log-depth pairwise fold, matching the order the vectorized
// Reduce node's _dace_horizontal_tree uses so both paths agree numerically.
template <typename T, int VLEN, char Op>
inline T tile_reduce(const T* __restrict__ src) {
  T buf[VLEN];
  for (int i = 0; i < VLEN; ++i) buf[i] = src[i];
  int n = VLEN;
  while (n > 1) {
    int half = n / 2;
    for (int i = 0; i < half; ++i) buf[i] = tile_apply<T, Op>(buf[2 * i], buf[2 * i + 1]);
    if (n & 1) buf[half] = buf[n - 1];
    n = half + (n & 1);
  }
  return buf[0];
}

}  // namespace tileops
}  // namespace dace
