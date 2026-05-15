// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Arch-independent per-op escape-hatch siblings (Option F overlay).
//
// Every arch file (scalar / avx2 / avx512 / arm_neon / arm_sve) includes
// this header. The unsuffixed names ``vector_<op>`` are owned by the
// arch file (best implementation for that backend); the suffixed names
// here are the per-op overrides the emitter can pick via the
// ``intrin_ops_disabled`` / ``prefer_intrin`` knobs.
//
//   vector_<op>_pscalar    pure scalar loop, NO _dace_vectorize hint.
//                          Deterministic; for debugging / when SIMD
//                          autovec has precision or correctness issues
//                          on a specific op.
//   vector_<op>_av         scalar loop + _dace_vectorize hint. Lets the
//                          compiler decide whether/how to vectorize.
//
// For binary / unary ops, both have ``_masked`` siblings: lanes where
// ``mask[i] == false`` leave ``out[i]`` unchanged (read-modify-write per
// lane). Mask is ``bool[W]``. The ``vector_select`` op is already a
// conditional and only has ``_pscalar`` / ``_av`` (no _masked).
//
// Suffix ``_pscalar`` (not ``_scalar``) is used to avoid collision with
// the ``_w_scalar`` operand-type suffix (``vector_add_w_scalar`` = a
// vector plus a scalar constant).
//
// Critical invariant: ``_pscalar`` must NEVER carry ``_dace_vectorize``.
// That is the entire point of the variant. Verified by code structure
// below: only the ``_AV`` macros expand to bodies with ``_dace_vectorize``.

#pragma once

#include <algorithm>
#include <cmath>

// ----------------------------------------------------------------------------
// Generator macros
// ----------------------------------------------------------------------------
// Each macro family defines four functions per op:
//   vector_<name>_pscalar          (pure scalar, no hint)
//   vector_<name>_av               (autovec hint)
//   vector_<name>_pscalar_masked   (pure scalar, masked, RMW on out)
//   vector_<name>_av_masked        (autovec hint, masked, RMW on out)
// ``EXPR`` is the per-lane expression with ``i`` as the lane index.

#define _DACE_VEC_BODY_PSCALAR(EXPR)                                            \
    for (int i = 0; i < vector_width; i++) { out[i] = (EXPR); }

#define _DACE_VEC_BODY_AV(EXPR)                                                 \
    _dace_vectorize(vector_width)                                               \
    for (int i = 0; i < vector_width; i++) { out[i] = (EXPR); }

#define _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                     \
    for (int i = 0; i < vector_width; i++) { if (mask[i]) out[i] = (EXPR); }

#define _DACE_VEC_BODY_AV_MASKED(EXPR)                                          \
    _dace_vectorize(vector_width)                                               \
    for (int i = 0; i < vector_width; i++) { if (mask[i]) out[i] = (EXPR); }

// Binary op (vec + vec).
#define DACE_VEC_DEFINE_BINOP(NAME, EXPR)                                       \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_pscalar(T* __restrict__ out,                        \
                                    const T* __restrict__ a,                    \
                                    const T* __restrict__ b) {                  \
    _DACE_VEC_BODY_PSCALAR(EXPR)                                                \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_av(T* __restrict__ out,                             \
                               const T* __restrict__ a,                         \
                               const T* __restrict__ b) {                       \
    _DACE_VEC_BODY_AV(EXPR)                                                     \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_pscalar_masked(T* __restrict__ out,                 \
                                           const T* __restrict__ a,             \
                                           const T* __restrict__ b,             \
                                           const bool* __restrict__ mask) {     \
    _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                         \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_av_masked(T* __restrict__ out,                      \
                                      const T* __restrict__ a,                  \
                                      const T* __restrict__ b,                  \
                                      const bool* __restrict__ mask) {          \
    _DACE_VEC_BODY_AV_MASKED(EXPR)                                              \
}

// Binary op with scalar constant (vec + scalar). EXPR uses ``constant``.
#define DACE_VEC_DEFINE_BINOP_W_SCALAR(NAME, EXPR)                              \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_pscalar(T* __restrict__ out,                        \
                                    const T* __restrict__ a,                    \
                                    const T constant) {                         \
    _DACE_VEC_BODY_PSCALAR(EXPR)                                                \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_av(T* __restrict__ out,                             \
                               const T* __restrict__ a,                         \
                               const T constant) {                              \
    _DACE_VEC_BODY_AV(EXPR)                                                     \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_pscalar_masked(T* __restrict__ out,                 \
                                           const T* __restrict__ a,             \
                                           const T constant,                    \
                                           const bool* __restrict__ mask) {     \
    _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                         \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_av_masked(T* __restrict__ out,                      \
                                      const T* __restrict__ a,                  \
                                      const T constant,                         \
                                      const bool* __restrict__ mask) {          \
    _DACE_VEC_BODY_AV_MASKED(EXPR)                                              \
}

// Unary op (vec).
#define DACE_VEC_DEFINE_UNOP(NAME, EXPR)                                        \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_pscalar(T* __restrict__ out,                        \
                                    const T* __restrict__ a) {                  \
    _DACE_VEC_BODY_PSCALAR(EXPR)                                                \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_av(T* __restrict__ out,                             \
                               const T* __restrict__ a) {                       \
    _DACE_VEC_BODY_AV(EXPR)                                                     \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_pscalar_masked(T* __restrict__ out,                 \
                                           const T* __restrict__ a,             \
                                           const bool* __restrict__ mask) {     \
    _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                         \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_av_masked(T* __restrict__ out,                      \
                                      const T* __restrict__ a,                  \
                                      const bool* __restrict__ mask) {          \
    _DACE_VEC_BODY_AV_MASKED(EXPR)                                              \
}

// Broadcast (scalar -> vec): vector_<name>(out, constant).
#define DACE_VEC_DEFINE_BROADCAST(NAME, EXPR)                                   \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_pscalar(T* __restrict__ out,                        \
                                    const T constant) {                         \
    _DACE_VEC_BODY_PSCALAR(EXPR)                                                \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_av(T* __restrict__ out,                             \
                               const T constant) {                              \
    _DACE_VEC_BODY_AV(EXPR)                                                     \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_pscalar_masked(T* __restrict__ out,                 \
                                           const T constant,                    \
                                           const bool* __restrict__ mask) {     \
    _DACE_VEC_BODY_PSCALAR_MASKED(EXPR)                                         \
}                                                                                \
template<typename T, int vector_width>                                          \
inline void vector_##NAME##_av_masked(T* __restrict__ out,                      \
                                      const T constant,                         \
                                      const bool* __restrict__ mask) {          \
    _DACE_VEC_BODY_AV_MASKED(EXPR)                                              \
}

// ============================================================================
// Op definitions
// ============================================================================

// Arithmetic (binary vec+vec)
DACE_VEC_DEFINE_BINOP(add,  a[i] + b[i])
DACE_VEC_DEFINE_BINOP(sub,  a[i] - b[i])
DACE_VEC_DEFINE_BINOP(mult, a[i] * b[i])
DACE_VEC_DEFINE_BINOP(div,  a[i] / b[i])
DACE_VEC_DEFINE_BINOP(min,  std::min(a[i], b[i]))
DACE_VEC_DEFINE_BINOP(max,  std::max(a[i], b[i]))

// Arithmetic with scalar constant
DACE_VEC_DEFINE_BINOP_W_SCALAR(add_w_scalar,  a[i] + constant)
DACE_VEC_DEFINE_BINOP_W_SCALAR(sub_w_scalar,  a[i] - constant)
DACE_VEC_DEFINE_BINOP_W_SCALAR(mult_w_scalar, a[i] * constant)
DACE_VEC_DEFINE_BINOP_W_SCALAR(div_w_scalar,  a[i] / constant)
DACE_VEC_DEFINE_BINOP_W_SCALAR(min_w_scalar,  std::min(a[i], constant))
DACE_VEC_DEFINE_BINOP_W_SCALAR(max_w_scalar,  std::max(a[i], constant))

// Constant on the LEFT (non-commutative ops only): vector_<op>_w_scalar_c(out, constant, a)
DACE_VEC_DEFINE_BINOP_W_SCALAR(sub_w_scalar_c, constant - a[i])
DACE_VEC_DEFINE_BINOP_W_SCALAR(div_w_scalar_c, constant / a[i])

// Copy / broadcast
DACE_VEC_DEFINE_UNOP(copy, a[i])
DACE_VEC_DEFINE_BROADCAST(copy_w_scalar, constant)

// Comparisons (numeric output 1.0 / 0.0)
DACE_VEC_DEFINE_BINOP(gt, (a[i] >  b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(lt, (a[i] <  b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(ge, (a[i] >= b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(le, (a[i] <= b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(eq, (a[i] == b[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP(ne, (a[i] != b[i]) ? T(1) : T(0))

DACE_VEC_DEFINE_BINOP_W_SCALAR(gt_w_scalar, (a[i] >  constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(lt_w_scalar, (a[i] <  constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(ge_w_scalar, (a[i] >= constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(le_w_scalar, (a[i] <= constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(eq_w_scalar, (a[i] == constant) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(ne_w_scalar, (a[i] != constant) ? T(1) : T(0))

DACE_VEC_DEFINE_BINOP_W_SCALAR(gt_w_scalar_c, (constant >  a[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(lt_w_scalar_c, (constant <  a[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(ge_w_scalar_c, (constant >= a[i]) ? T(1) : T(0))
DACE_VEC_DEFINE_BINOP_W_SCALAR(le_w_scalar_c, (constant <= a[i]) ? T(1) : T(0))

// Elementwise transcendentals
DACE_VEC_DEFINE_UNOP(exp,   std::exp(a[i]))
DACE_VEC_DEFINE_UNOP(log,   std::log(a[i]))
DACE_VEC_DEFINE_UNOP(log2,  std::log2(a[i]))
DACE_VEC_DEFINE_UNOP(log10, std::log10(a[i]))
DACE_VEC_DEFINE_UNOP(sin,   std::sin(a[i]))
DACE_VEC_DEFINE_UNOP(cos,   std::cos(a[i]))
DACE_VEC_DEFINE_UNOP(tan,   std::tan(a[i]))
DACE_VEC_DEFINE_UNOP(asin,  std::asin(a[i]))
DACE_VEC_DEFINE_UNOP(acos,  std::acos(a[i]))
DACE_VEC_DEFINE_UNOP(atan,  std::atan(a[i]))
DACE_VEC_DEFINE_UNOP(sinh,  std::sinh(a[i]))
DACE_VEC_DEFINE_UNOP(cosh,  std::cosh(a[i]))
DACE_VEC_DEFINE_UNOP(tanh,  std::tanh(a[i]))
DACE_VEC_DEFINE_UNOP(sqrt,  std::sqrt(a[i]))
DACE_VEC_DEFINE_UNOP(cbrt,  std::cbrt(a[i]))
DACE_VEC_DEFINE_UNOP(abs,   std::abs(a[i]))
DACE_VEC_DEFINE_UNOP(floor, std::floor(a[i]))
DACE_VEC_DEFINE_UNOP(ceil,  std::ceil(a[i]))
DACE_VEC_DEFINE_UNOP(round, std::round(a[i]))
DACE_VEC_DEFINE_UNOP(neg,   -a[i])

// Pow (binary vec, scalar exponent)
DACE_VEC_DEFINE_BINOP_W_SCALAR(pow_w_scalar, std::pow(a[i], constant))

// ============================================================================
// vector_select: 3-input conditional ``out = cond ? t : e``.
//
// The masked variant is load-bearing for a masked remainder: an active
// lane performs the select; an INACTIVE lane is left *untouched* — the
// same gated-store form the ``*_av_masked`` binops use
// (``if (mask[i]) out[i] = EXPR``). It must NOT write inactive lanes,
// not even with ``e``: a masked remainder over ``R < W`` lanes binds
// ``out = arr + tile_i`` so the trailing ``W - R`` lanes index past the
// array end; storing there (any value) corrupts the heap (TSVC s2710
// masked-merge-65 segfault). Skipping the store leaves that memory
// alone and, for the in-place RMW merge target where ``e`` aliases the
// destination, preserves ``arr``'s old value on inactive valid lanes
// exactly as ``out[i] = e`` would have — minus the OOB.
// ============================================================================
template<typename T, int vector_width, typename CondT = bool>
inline void vector_select_pscalar(T* __restrict__ out,
                                  const CondT* __restrict__ cond,
                                  const T* __restrict__ t,
                                  const T* __restrict__ e) {
    for (int i = 0; i < vector_width; i++) out[i] = cond[i] ? t[i] : e[i];
}

template<typename T, int vector_width, typename CondT = bool>
inline void vector_select_av(T* __restrict__ out,
                             const CondT* __restrict__ cond,
                             const T* __restrict__ t,
                             const T* __restrict__ e) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) out[i] = cond[i] ? t[i] : e[i];
}

template<typename T, int vector_width, typename CondT = bool>
inline void vector_select_av_masked(T* __restrict__ out,
                                    const CondT* __restrict__ cond,
                                    const T* __restrict__ t,
                                    const T* __restrict__ e,
                                    const bool* __restrict__ mask) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) { if (mask[i]) out[i] = cond[i] ? t[i] : e[i]; }
}

// ----------------------------------------------------------------------------
// Cleanup macros — keep namespace tight; downstream files do not need them.
// ----------------------------------------------------------------------------
#undef DACE_VEC_DEFINE_BINOP
#undef DACE_VEC_DEFINE_BINOP_W_SCALAR
#undef DACE_VEC_DEFINE_UNOP
#undef DACE_VEC_DEFINE_BROADCAST
#undef _DACE_VEC_BODY_PSCALAR
#undef _DACE_VEC_BODY_AV
#undef _DACE_VEC_BODY_PSCALAR_MASKED
#undef _DACE_VEC_BODY_AV_MASKED
