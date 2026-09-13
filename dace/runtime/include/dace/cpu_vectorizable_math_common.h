// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Per-op overrides of each arch file's vector_<op>: ``_pscalar`` is a plain scalar loop and must never
// carry ``_dace_vectorize``; ``_av`` adds that hint. Binary and unary ops also have ``_masked`` variants
// that leave lanes with ``mask[i] == false`` unchanged.

#pragma once

#include <algorithm>
#include <cmath>

// Full-unroll hint for fixed-width (constexpr-bounded) lane loops. Normally
// provided by dace/types.h; defined defensively here so the vectorizable-math
// headers can be included standalone (e.g. in isolation tests) without it.
#ifndef DACE_UNROLL
#if defined(__clang__) || defined(__CUDACC__) || defined(__INTEL_LLVM_COMPILER)
#define DACE_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define DACE_UNROLL _Pragma("GCC unroll 64")
#else
#define DACE_UNROLL
#endif
#endif

// Generator macros
// Each macro family defines four functions per op:
//   vector_<name>_pscalar          (pure scalar, no hint)
//   vector_<name>_av               (autovec hint)
//   vector_<name>_pscalar_masked   (pure scalar, masked, RMW on out)
//   vector_<name>_av_masked        (autovec hint, masked, RMW on out)
// ``EXPR`` is the per-lane expression with ``i`` as the lane index.

#define _DACE_VEC_BODY_PSCALAR(EXPR)       \
  for (int i = 0; i < vector_width; i++) { \
    out[i] = (EXPR);                       \
  }

#define _DACE_VEC_BODY_AV(EXPR) \
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) { out[i] = (EXPR); }

#define _DACE_VEC_BODY_PSCALAR_MASKED(EXPR) \
  for (int i = 0; i < vector_width; i++) {  \
    if (mask[i]) out[i] = (EXPR);           \
  }

#define _DACE_VEC_BODY_AV_MASKED(EXPR)                                   \
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) { \
    if (mask[i]) out[i] = (EXPR);                                        \
  }

// Binary op (vec + vec).
#define DACE_VEC_DEFINE_BINOP(NAME, EXPR)                                                                             \
  template <typename T, int vector_width>                                                                             \
  static inline void vector_##NAME##_pscalar(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) { \
    _DACE_VEC_BODY_PSCALAR(EXPR)                                                                                      \
  }                                                                                                                   \
  template <typename T, int vector_width>                                                                             \
  static inline void vector_##NAME##_av(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {      \
    _DACE_VEC_BODY_AV(EXPR)                                                                                           \
  }                                                                                                                   \
  template <typename T, int vector_width>                                                                             \
  static inline void vector_##NAME##_pscalar_masked(T* __restrict__ out, const T* __restrict__ a,                     \
                                                    const T* __restrict__ b, const bool* __restrict__ mask) {         \
    _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                                                               \
  }                                                                                                                   \
  template <typename T, int vector_width>                                                                             \
  static inline void vector_##NAME##_av_masked(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b, \
                                               const bool* __restrict__ mask) {                                       \
    _DACE_VEC_BODY_AV_MASKED(EXPR)                                                                                    \
  }

// Binary op with scalar constant (vec + scalar). EXPR uses ``constant``.
#define DACE_VEC_DEFINE_BINOP_W_SCALAR(NAME, EXPR)                                                                  \
  template <typename T, int vector_width>                                                                           \
  static inline void vector_##NAME##_pscalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {      \
    _DACE_VEC_BODY_PSCALAR(EXPR)                                                                                    \
  }                                                                                                                 \
  template <typename T, int vector_width>                                                                           \
  static inline void vector_##NAME##_av(T* __restrict__ out, const T* __restrict__ a, const T constant) {           \
    _DACE_VEC_BODY_AV(EXPR)                                                                                         \
  }                                                                                                                 \
  template <typename T, int vector_width>                                                                           \
  static inline void vector_##NAME##_pscalar_masked(T* __restrict__ out, const T* __restrict__ a, const T constant, \
                                                    const bool* __restrict__ mask) {                                \
    _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                                                             \
  }                                                                                                                 \
  template <typename T, int vector_width>                                                                           \
  static inline void vector_##NAME##_av_masked(T* __restrict__ out, const T* __restrict__ a, const T constant,      \
                                               const bool* __restrict__ mask) {                                     \
    _DACE_VEC_BODY_AV_MASKED(EXPR)                                                                                  \
  }

// Unary op (vec).
#define DACE_VEC_DEFINE_UNOP(NAME, EXPR)                                                          \
  template <typename T, int vector_width>                                                         \
  static inline void vector_##NAME##_pscalar(T* __restrict__ out, const T* __restrict__ a) {      \
    _DACE_VEC_BODY_PSCALAR(EXPR)                                                                  \
  }                                                                                               \
  template <typename T, int vector_width>                                                         \
  static inline void vector_##NAME##_av(T* __restrict__ out, const T* __restrict__ a) {           \
    _DACE_VEC_BODY_AV(EXPR)                                                                       \
  }                                                                                               \
  template <typename T, int vector_width>                                                         \
  static inline void vector_##NAME##_pscalar_masked(T* __restrict__ out, const T* __restrict__ a, \
                                                    const bool* __restrict__ mask) {              \
    _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                                           \
  }                                                                                               \
  template <typename T, int vector_width>                                                         \
  static inline void vector_##NAME##_av_masked(T* __restrict__ out, const T* __restrict__ a,      \
                                               const bool* __restrict__ mask) {                   \
    _DACE_VEC_BODY_AV_MASKED(EXPR)                                                                \
  }

// Broadcast (scalar -> vec): vector_<name>(out, constant).
#define DACE_VEC_DEFINE_BROADCAST(NAME, EXPR)                                                                          \
  template <typename T, int vector_width>                                                                              \
  static inline void vector_##NAME##_pscalar(T* __restrict__ out, const T constant) {                                  \
    _DACE_VEC_BODY_PSCALAR(EXPR)                                                                                       \
  }                                                                                                                    \
  template <typename T, int vector_width>                                                                              \
  static inline void vector_##NAME##_av(T* __restrict__ out, const T constant) {                                       \
    _DACE_VEC_BODY_AV(EXPR)                                                                                            \
  }                                                                                                                    \
  template <typename T, int vector_width>                                                                              \
  static inline void vector_##NAME##_pscalar_masked(T* __restrict__ out, const T constant,                             \
                                                    const bool* __restrict__ mask) {                                   \
    _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                                                                \
  }                                                                                                                    \
  template <typename T, int vector_width>                                                                              \
  static inline void vector_##NAME##_av_masked(T* __restrict__ out, const T constant, const bool* __restrict__ mask) { \
    _DACE_VEC_BODY_AV_MASKED(EXPR)                                                                                     \
  }

// Op definitions

// Arithmetic (binary vec+vec)
DACE_VEC_DEFINE_BINOP(add, a[i] + b[i])
DACE_VEC_DEFINE_BINOP(sub, a[i] - b[i])
DACE_VEC_DEFINE_BINOP(mult, a[i] * b[i])
DACE_VEC_DEFINE_BINOP(div, a[i] / b[i])
DACE_VEC_DEFINE_BINOP(min, std::min(a[i], b[i]))
DACE_VEC_DEFINE_BINOP(max, std::max(a[i], b[i]))

// Arithmetic with scalar constant
DACE_VEC_DEFINE_BINOP_W_SCALAR(add_w_scalar, a[i] + constant)
DACE_VEC_DEFINE_BINOP_W_SCALAR(sub_w_scalar, a[i] - constant)
DACE_VEC_DEFINE_BINOP_W_SCALAR(mult_w_scalar, a[i] * constant)
DACE_VEC_DEFINE_BINOP_W_SCALAR(div_w_scalar, a[i] / constant)
DACE_VEC_DEFINE_BINOP_W_SCALAR(min_w_scalar, std::min(a[i], constant))
DACE_VEC_DEFINE_BINOP_W_SCALAR(max_w_scalar, std::max(a[i], constant))

// Constant on the LEFT (non-commutative ops only): vector_<op>_w_scalar_c(out,
// constant, a)
DACE_VEC_DEFINE_BINOP_W_SCALAR(sub_w_scalar_c, constant - a[i])
DACE_VEC_DEFINE_BINOP_W_SCALAR(div_w_scalar_c, constant / a[i])

// Copy / broadcast
DACE_VEC_DEFINE_UNOP(copy, a[i])
DACE_VEC_DEFINE_BROADCAST(copy_w_scalar, constant)

// Comparisons (numeric output 1.0 / 0.0)
DACE_VEC_DEFINE_BINOP(gt, (a[i] > b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(lt, (a[i] < b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(ge, (a[i] >= b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(le, (a[i] <= b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(eq, (a[i] == b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(ne, (a[i] != b[i]) ? T(1) : T(0))

DACE_VEC_DEFINE_BINOP_W_SCALAR(gt_w_scalar, (a[i] > constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(lt_w_scalar, (a[i] < constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(ge_w_scalar, (a[i] >= constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(le_w_scalar, (a[i] <= constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(eq_w_scalar, (a[i] == constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(ne_w_scalar, (a[i] != constant) ? T(1) : T(0))

DACE_VEC_DEFINE_BINOP_W_SCALAR(gt_w_scalar_c, (constant > a[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(lt_w_scalar_c, (constant < a[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(ge_w_scalar_c, (constant >= a[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(le_w_scalar_c, (constant <= a[i]) ? T(1) : T(0))

// Elementwise transcendentals
DACE_VEC_DEFINE_UNOP(exp, std::exp(a[i]))
DACE_VEC_DEFINE_UNOP(log, std::log(a[i]))
DACE_VEC_DEFINE_UNOP(log2, std::log2(a[i]))
DACE_VEC_DEFINE_UNOP(log10, std::log10(a[i]))
DACE_VEC_DEFINE_UNOP(sin, std::sin(a[i]))
DACE_VEC_DEFINE_UNOP(cos, std::cos(a[i]))
DACE_VEC_DEFINE_UNOP(tan, std::tan(a[i]))
DACE_VEC_DEFINE_UNOP(asin, std::asin(a[i]))
DACE_VEC_DEFINE_UNOP(acos, std::acos(a[i]))
DACE_VEC_DEFINE_UNOP(atan, std::atan(a[i]))
DACE_VEC_DEFINE_UNOP(sinh, std::sinh(a[i]))
DACE_VEC_DEFINE_UNOP(cosh, std::cosh(a[i]))
DACE_VEC_DEFINE_UNOP(tanh, std::tanh(a[i]))
DACE_VEC_DEFINE_UNOP(sqrt, std::sqrt(a[i]))
DACE_VEC_DEFINE_UNOP(cbrt, std::cbrt(a[i]))
DACE_VEC_DEFINE_UNOP(abs, std::abs(a[i]))
DACE_VEC_DEFINE_UNOP(floor, std::floor(a[i]))
DACE_VEC_DEFINE_UNOP(ceil, std::ceil(a[i]))
DACE_VEC_DEFINE_UNOP(round, std::round(a[i]))
DACE_VEC_DEFINE_UNOP(neg, -a[i])

// Pow (binary vec, scalar exponent)
DACE_VEC_DEFINE_BINOP_W_SCALAR(pow_w_scalar, std::pow(a[i], constant))

// vector_select: ``out = cond ? t : e``. The masked variant never writes inactive lanes, which may lie past
// the array end in a masked remainder.
template <typename T, int vector_width, typename CondT = bool>
static inline void vector_select_pscalar(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                                         const T* __restrict__ e) {
  for (int i = 0; i < vector_width; i++) out[i] = cond[i] ? t[i] : e[i];
}

template <typename T, int vector_width, typename CondT = bool>
static inline void vector_select_av(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                                    const T* __restrict__ e) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) out[i] = cond[i] ? t[i] : e[i];
}

template <typename T, int vector_width, typename CondT = bool>
static inline void vector_select_av_masked(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                                           const T* __restrict__ e, const bool* __restrict__ mask) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) {
    if (mask[i]) out[i] = cond[i] ? t[i] : e[i];
  }
}

// Integer floor/ceil division (``dace::math::int_floor`` / ``int_ceil``).
// Arch-independent -- integer division is not a single SIMD instruction, so
// these lower to the same per-lane division on every backend. Bodies match
// the scalar definitions in ``dace/math.h`` (``a / b`` and
// ``(a + b - 1) / b``) so the vectorized path stays bit-identical to the
// non-vectorized one (TSVC s276: ``int_floor(LEN_1D, 2)`` lane condition).
template <typename T, int vector_width>
static inline void vector_int_floor(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) out[i] = a[i] / b[i];
}

template <typename T, int vector_width>
static inline void vector_int_floor_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) out[i] = a[i] / constant;
}

template <typename T, int vector_width>
static inline void vector_int_floor_w_scalar_c(T* __restrict__ out, const T constant, const T* __restrict__ b) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) out[i] = constant / b[i];
}

template <typename T, int vector_width>
static inline void vector_int_ceil(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) out[i] = (a[i] + b[i] - 1) / b[i];
}

template <typename T, int vector_width>
static inline void vector_int_ceil_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) out[i] = (a[i] + constant - 1) / constant;
}

template <typename T, int vector_width>
static inline void vector_int_ceil_w_scalar_c(T* __restrict__ out, const T constant, const T* __restrict__ b) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) out[i] = (constant + b[i] - 1) / b[i];
}

// Horizontal reduction to one scalar: ``_dace_horizontal_tree_<op>`` is a balanced pairwise tree, pairs
// (0,1)(2,3)... with an odd last element forwarded, matching ``emit_tree_reduction``.
#define _DACE_HREDUCE_TREE(NAME, OP)                                          \
  template <typename T, int vector_width>                                     \
  static inline T _dace_horizontal_tree_##NAME(const T* __restrict__ a) {     \
    T buf[vector_width];                                                      \
    for (int i = 0; i < vector_width; i++) buf[i] = a[i];                     \
    int n = vector_width;                                                     \
    while (n > 1) {                                                           \
      int half = n / 2;                                                       \
      for (int i = 0; i < half; i++) buf[i] = OP(buf[2 * i], buf[2 * i + 1]); \
      if (n & 1) buf[half] = buf[n - 1];                                      \
      n = half + (n & 1);                                                     \
    }                                                                         \
    return buf[0];                                                            \
  }

#define _DACE_HREDUCE_ADD(x, y) ((x) + (y))
#define _DACE_HREDUCE_MUL(x, y) ((x) * (y))
#define _DACE_HREDUCE_MAXOP(x, y) (std::max((x), (y)))
#define _DACE_HREDUCE_MINOP(x, y) (std::min((x), (y)))
#define _DACE_HREDUCE_AND(x, y) ((x) & (y))
#define _DACE_HREDUCE_OR(x, y) ((x) | (y))
#define _DACE_HREDUCE_XOR(x, y) ((x) ^ (y))

_DACE_HREDUCE_TREE(add, _DACE_HREDUCE_ADD)
_DACE_HREDUCE_TREE(mul, _DACE_HREDUCE_MUL)
_DACE_HREDUCE_TREE(max, _DACE_HREDUCE_MAXOP)
_DACE_HREDUCE_TREE(min, _DACE_HREDUCE_MINOP)
_DACE_HREDUCE_TREE(band, _DACE_HREDUCE_AND)
_DACE_HREDUCE_TREE(bor, _DACE_HREDUCE_OR)
_DACE_HREDUCE_TREE(bxor, _DACE_HREDUCE_XOR)

#undef _DACE_HREDUCE_TREE
#undef _DACE_HREDUCE_ADD
#undef _DACE_HREDUCE_MUL
#undef _DACE_HREDUCE_MAXOP
#undef _DACE_HREDUCE_MINOP
#undef _DACE_HREDUCE_AND
#undef _DACE_HREDUCE_OR
#undef _DACE_HREDUCE_XOR

// Cleanup macros -- keep namespace tight; downstream files do not need them.
#undef DACE_VEC_DEFINE_BINOP
#undef DACE_VEC_DEFINE_BINOP_W_SCALAR
#undef DACE_VEC_DEFINE_UNOP
#undef DACE_VEC_DEFINE_BROADCAST
#undef _DACE_VEC_BODY_PSCALAR
#undef _DACE_VEC_BODY_AV
#undef _DACE_VEC_BODY_PSCALAR_MASKED
#undef _DACE_VEC_BODY_AV_MASKED
