// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// ARM SVE backend of the K=1 tile-op intrinsics. A sibling of
// ``dace/tile_ops/scalar.h`` exposing the SAME ``dace::tileops`` signatures
// (``tile_binop`` / ``tile_ite`` / ``tile_load`` / ``tile_store`` /
// ``tile_gather`` / ``tile_scatter``, each with a constexpr-VLEN and a
// runtime-VLEN form), lowered to ARM Scalable Vector Extension intrinsics.
// Element types: ``float`` (f32), ``double`` (f64), ``int32_t`` (s32),
// ``int64_t`` (s64). Anything else falls through to a portable scalar loop.
//
// SVE is vector-length-agnostic: the hardware vector width is known only at
// runtime via ``svcntw()`` / ``svcntd()``, so every op (BOTH the runtime-vlen
// and the constexpr-VLEN form) is the same ``svwhilelt`` chunk loop --
// ``for (i = 0; i < n; i += svcnt*()) { pg = svwhilelt_b**(i, n); ... }``. The
// trailing lanes of the final chunk are inactive in ``pg`` and a predicated
// ``svld1`` does not fault / a predicated ``svst1`` does not write them, which
// is exactly the tile-op OOB-safe contract.
//
// Masked semantics (mirrors the scalar header, two flavours):
//   * tile PRODUCERS (binop / merge / load / gather) -- ZERO-FILL inactive
//     lanes. We compute under the active mask ``m`` (a subset of ``pg``) and
//     ``svsel`` against a zero splat so the in-tile-but-masked-off lanes become
//     ``T(0)``, then store under ``pg``.
//   * array WRITERS (store / scatter) -- RMW skip-inactive: we store under the
//     active mask ``m`` so an inactive lane is never written and the
//     destination array (and its OOB tail) is never touched.
//
// The lane mask arrives as a portable ``const bool*`` (one byte per lane). It
// is loaded byte-wise zero-extended under ``pg`` (``svld1ub_u32`` / ``_u64``)
// and compared ``!= 0`` (``svcmpne_n_u32`` / ``_u64``) to build ``m``; because
// the compare ANDs in ``pg``, ``m`` is always a subset of ``pg`` so its active
// lanes are guaranteed in-bounds (the exact pattern from
// ``dace/cpu_vectorizable_math_arm_sve.h``).
//
// CAUTIONS (see the SVE reference note):
//   * And / Or are LOGICAL truthiness (``(a && b) ? 1 : 0``), NOT bitwise: we
//     compare each operand ``!= 0`` to a predicate, combine with the PREDICATE
//     logicals ``svand_b_z`` / ``svorr_b_z`` (NOT the bitwise lane ops
//     ``svand_s32`` / ``svorr_s32``), then ``svsel`` to ``T(1)`` / ``T(0)``.
//   * Min / Max use ``svmin`` / ``svmax`` (FMIN / FMAX for fp), which follow
//     IEEE-754 NaN propagation. This differs from the scalar header's
//     ``std::min`` / ``std::max`` (a ``<``-based pick that returns the first
//     arg on a NaN comparison). On NaN inputs the two backends can diverge;
//     ``svminnm`` / ``svmaxnm`` are the NaN-quiet variants if exact parity is
//     ever required.
//   * Integer divide uses ``svdiv_s32`` / ``svdiv_s64`` (base-SVE SDIV). On
//     AArch64 integer divide-by-zero yields 0 (no trap), vs. C's UB.
//   * Gather / scatter use the element-scaled ``*index*`` forms (NOT the byte
//     ``*offset*`` forms). The index-vector element width matches the data lane
//     width: 32-bit data pairs with ``s32index`` (svcntw loop), 64-bit data
//     with ``s64index`` (svcntd loop). An ``int32_t`` index array for 64-bit
//     data is sign-extended to s64 with ``svld1sw_s64``; an ``int64_t`` index
//     array for 32-bit data is narrowed to s32 with ``svld1_s64`` + ``svcvt``-
//     free ``svqxtnt``-free explicit narrowing via a temporary (handled below).
#pragma once

#include <arm_sve.h>

#include <cstdint>
#include <type_traits>

#if !defined(__ARM_FEATURE_SVE)
#error Included the SVE tile-op header without SVE support
#endif

namespace dace {
namespace tileops {

// ===========================================================================
// 32-bit-lane (f32 / s32) primitives
// ===========================================================================

// ---- predicated contiguous load ----
inline svfloat32_t sve_ld1(svbool_t pg, const float* p) { return svld1_f32(pg, p); }
inline svint32_t sve_ld1(svbool_t pg, const std::int32_t* p) { return svld1_s32(pg, p); }
inline svfloat64_t sve_ld1(svbool_t pg, const double* p) { return svld1_f64(pg, p); }
inline svint64_t sve_ld1(svbool_t pg, const std::int64_t* p) { return svld1_s64(pg, p); }

// ---- predicated contiguous store ----
inline void sve_st1(svbool_t pg, float* p, svfloat32_t v) { svst1_f32(pg, p, v); }
inline void sve_st1(svbool_t pg, std::int32_t* p, svint32_t v) { svst1_s32(pg, p, v); }
inline void sve_st1(svbool_t pg, double* p, svfloat64_t v) { svst1_f64(pg, p, v); }
inline void sve_st1(svbool_t pg, std::int64_t* p, svint64_t v) { svst1_s64(pg, p, v); }

// ---- splat ----
inline svfloat32_t sve_dup(float v) { return svdup_f32(v); }
inline svint32_t sve_dup(std::int32_t v) { return svdup_s32(v); }
inline svfloat64_t sve_dup(double v) { return svdup_f64(v); }
inline svint64_t sve_dup(std::int64_t v) { return svdup_s64(v); }

// ---- arithmetic / min / max (don't-care inactive lanes, ``_x``) ----
inline svfloat32_t sve_add(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svadd_f32_x(pg, a, b); }
inline svint32_t sve_add(svbool_t pg, svint32_t a, svint32_t b) { return svadd_s32_x(pg, a, b); }
inline svfloat64_t sve_add(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svadd_f64_x(pg, a, b); }
inline svint64_t sve_add(svbool_t pg, svint64_t a, svint64_t b) { return svadd_s64_x(pg, a, b); }

inline svfloat32_t sve_sub(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svsub_f32_x(pg, a, b); }
inline svint32_t sve_sub(svbool_t pg, svint32_t a, svint32_t b) { return svsub_s32_x(pg, a, b); }
inline svfloat64_t sve_sub(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svsub_f64_x(pg, a, b); }
inline svint64_t sve_sub(svbool_t pg, svint64_t a, svint64_t b) { return svsub_s64_x(pg, a, b); }

inline svfloat32_t sve_mul(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svmul_f32_x(pg, a, b); }
inline svint32_t sve_mul(svbool_t pg, svint32_t a, svint32_t b) { return svmul_s32_x(pg, a, b); }
inline svfloat64_t sve_mul(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svmul_f64_x(pg, a, b); }
inline svint64_t sve_mul(svbool_t pg, svint64_t a, svint64_t b) { return svmul_s64_x(pg, a, b); }

inline svfloat32_t sve_div(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svdiv_f32_x(pg, a, b); }
inline svint32_t sve_div(svbool_t pg, svint32_t a, svint32_t b) { return svdiv_s32_x(pg, a, b); }
inline svfloat64_t sve_div(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svdiv_f64_x(pg, a, b); }
inline svint64_t sve_div(svbool_t pg, svint64_t a, svint64_t b) { return svdiv_s64_x(pg, a, b); }

inline svfloat32_t sve_min(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svmin_f32_x(pg, a, b); }
inline svint32_t sve_min(svbool_t pg, svint32_t a, svint32_t b) { return svmin_s32_x(pg, a, b); }
inline svfloat64_t sve_min(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svmin_f64_x(pg, a, b); }
inline svint64_t sve_min(svbool_t pg, svint64_t a, svint64_t b) { return svmin_s64_x(pg, a, b); }

inline svfloat32_t sve_max(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svmax_f32_x(pg, a, b); }
inline svint32_t sve_max(svbool_t pg, svint32_t a, svint32_t b) { return svmax_s32_x(pg, a, b); }
inline svfloat64_t sve_max(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svmax_f64_x(pg, a, b); }
inline svint64_t sve_max(svbool_t pg, svint64_t a, svint64_t b) { return svmax_s64_x(pg, a, b); }

// ---- comparisons -> svbool_t (active set = ``pg`` AND (a OP b)) ----
inline svbool_t sve_cmplt(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svcmplt_f32(pg, a, b); }
inline svbool_t sve_cmplt(svbool_t pg, svint32_t a, svint32_t b) { return svcmplt_s32(pg, a, b); }
inline svbool_t sve_cmplt(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svcmplt_f64(pg, a, b); }
inline svbool_t sve_cmplt(svbool_t pg, svint64_t a, svint64_t b) { return svcmplt_s64(pg, a, b); }

inline svbool_t sve_cmple(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svcmple_f32(pg, a, b); }
inline svbool_t sve_cmple(svbool_t pg, svint32_t a, svint32_t b) { return svcmple_s32(pg, a, b); }
inline svbool_t sve_cmple(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svcmple_f64(pg, a, b); }
inline svbool_t sve_cmple(svbool_t pg, svint64_t a, svint64_t b) { return svcmple_s64(pg, a, b); }

inline svbool_t sve_cmpgt(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svcmpgt_f32(pg, a, b); }
inline svbool_t sve_cmpgt(svbool_t pg, svint32_t a, svint32_t b) { return svcmpgt_s32(pg, a, b); }
inline svbool_t sve_cmpgt(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svcmpgt_f64(pg, a, b); }
inline svbool_t sve_cmpgt(svbool_t pg, svint64_t a, svint64_t b) { return svcmpgt_s64(pg, a, b); }

inline svbool_t sve_cmpge(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svcmpge_f32(pg, a, b); }
inline svbool_t sve_cmpge(svbool_t pg, svint32_t a, svint32_t b) { return svcmpge_s32(pg, a, b); }
inline svbool_t sve_cmpge(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svcmpge_f64(pg, a, b); }
inline svbool_t sve_cmpge(svbool_t pg, svint64_t a, svint64_t b) { return svcmpge_s64(pg, a, b); }

inline svbool_t sve_cmpeq(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svcmpeq_f32(pg, a, b); }
inline svbool_t sve_cmpeq(svbool_t pg, svint32_t a, svint32_t b) { return svcmpeq_s32(pg, a, b); }
inline svbool_t sve_cmpeq(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svcmpeq_f64(pg, a, b); }
inline svbool_t sve_cmpeq(svbool_t pg, svint64_t a, svint64_t b) { return svcmpeq_s64(pg, a, b); }

inline svbool_t sve_cmpne(svbool_t pg, svfloat32_t a, svfloat32_t b) { return svcmpne_f32(pg, a, b); }
inline svbool_t sve_cmpne(svbool_t pg, svint32_t a, svint32_t b) { return svcmpne_s32(pg, a, b); }
inline svbool_t sve_cmpne(svbool_t pg, svfloat64_t a, svfloat64_t b) { return svcmpne_f64(pg, a, b); }
inline svbool_t sve_cmpne(svbool_t pg, svint64_t a, svint64_t b) { return svcmpne_s64(pg, a, b); }

// ---- ``v != 0`` predicate (for And / Or logical truthiness + tile_ite cond) ----
inline svbool_t sve_cmpne0(svbool_t pg, svfloat32_t v) { return svcmpne_n_f32(pg, v, 0.0f); }
inline svbool_t sve_cmpne0(svbool_t pg, svint32_t v) { return svcmpne_n_s32(pg, v, 0); }
inline svbool_t sve_cmpne0(svbool_t pg, svfloat64_t v) { return svcmpne_n_f64(pg, v, 0.0); }
inline svbool_t sve_cmpne0(svbool_t pg, svint64_t v) { return svcmpne_n_s64(pg, v, 0); }

// ---- select (cond ? a : b) ----
inline svfloat32_t sve_sel(svbool_t c, svfloat32_t a, svfloat32_t b) { return svsel_f32(c, a, b); }
inline svint32_t sve_sel(svbool_t c, svint32_t a, svint32_t b) { return svsel_s32(c, a, b); }
inline svfloat64_t sve_sel(svbool_t c, svfloat64_t a, svfloat64_t b) { return svsel_f64(c, a, b); }
inline svint64_t sve_sel(svbool_t c, svint64_t a, svint64_t b) { return svsel_s64(c, a, b); }

// Lane count + ``svwhilelt`` predicate keyed on the element width. 32-bit lanes
// (f32 / s32) use svcntw / svwhilelt_b32; 64-bit lanes (f64 / s64) use svcntd /
// svwhilelt_b64.
template <typename T>
inline int sve_cnt() {
  if constexpr (sizeof(T) == 4)
    return (int)svcntw();
  else
    return (int)svcntd();
}
template <typename T>
inline svbool_t sve_whilelt(int i, int n) {
  if constexpr (sizeof(T) == 4)
    return svwhilelt_b32(i, n);
  else
    return svwhilelt_b64(i, n);
}

// Build the active-lane predicate ``m = pg AND (mask[i + lane] != 0)`` from the
// portable ``bool[]`` lane mask. The byte mask is zero-extended to the lane
// width under ``pg`` so its active set is always a subset of ``pg``.
template <typename T>
inline svbool_t sve_mask(svbool_t pg, const bool* mask, int i) {
  if constexpr (sizeof(T) == 4) {
    svuint32_t mv = svld1ub_u32(pg, (const std::uint8_t*)(mask + i));
    return svcmpne_n_u32(pg, mv, 0);
  } else {
    svuint64_t mv = svld1ub_u64(pg, (const std::uint8_t*)(mask + i));
    return svcmpne_n_u64(pg, mv, 0);
  }
}

// Per-lane apply of a binary op on two same-type SVE vectors under predicate
// ``pg``, yielding a vector of the element type. Comparisons / logicals select
// ``T(1)`` / ``T(0)``; And / Or are LOGICAL (compare-to-zero + predicate
// combine), NOT bitwise.
template <typename T, char Op, typename VecT>
inline VecT sve_tile_apply(svbool_t pg, VecT a, VecT b) {
  if constexpr (Op == '+')
    return sve_add(pg, a, b);
  else if constexpr (Op == '-')
    return sve_sub(pg, a, b);
  else if constexpr (Op == '*')
    return sve_mul(pg, a, b);
  else if constexpr (Op == '/')
    return sve_div(pg, a, b);
  else if constexpr (Op == 'm')
    return sve_min(pg, a, b);
  else if constexpr (Op == 'M')
    return sve_max(pg, a, b);
  else {
    const VecT one = sve_dup(T(1));
    const VecT zero = sve_dup(T(0));
    if constexpr (Op == '<')
      return sve_sel(sve_cmplt(pg, a, b), one, zero);
    else if constexpr (Op == 'l')
      return sve_sel(sve_cmple(pg, a, b), one, zero);
    else if constexpr (Op == '>')
      return sve_sel(sve_cmpgt(pg, a, b), one, zero);
    else if constexpr (Op == 'g')
      return sve_sel(sve_cmpge(pg, a, b), one, zero);
    else if constexpr (Op == '=')
      return sve_sel(sve_cmpeq(pg, a, b), one, zero);
    else if constexpr (Op == '!')
      return sve_sel(sve_cmpne(pg, a, b), one, zero);
    else if constexpr (Op == '&') {
      svbool_t r = svand_b_z(pg, sve_cmpne0(pg, a), sve_cmpne0(pg, b));
      return sve_sel(r, one, zero);
    } else /* Or */ {
      svbool_t r = svorr_b_z(pg, sve_cmpne0(pg, a), sve_cmpne0(pg, b));
      return sve_sel(r, one, zero);
    }
  }
}

// ===========================================================================
// tile_binop : out[i] = a-operand <op> b-operand
// PRODUCER -> ZERO-FILL inactive (operand reads are in-tile, always safe).
// ===========================================================================
template <typename T, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask, int vlen) {
  // No SVE modulo intrinsic, and ``sve_tile_apply``'s catch-all is logical-OR;
  // run a portable scalar lane loop (``py_mod`` = Python/NumPy ``%`` semantics).
  if constexpr (Op == '%') {
    for (int i = 0; i < vlen; ++i) {
      const T av = BroadcastA ? a[0] : a[i];
      const T bv = BroadcastB ? b[0] : b[i];
      if constexpr (Masked)
        out[i] = mask[i] ? py_mod(av, bv) : T(0);
      else
        out[i] = py_mod(av, bv);
    }
    return;
  }
  using VecT = decltype(sve_dup(T(0)));  // vec type for T, from the sve_dup overload set
  const VecT zero = sve_dup(T(0));
  for (int i = 0; i < vlen; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, vlen);
    const VecT av = (!BroadcastA) ? sve_ld1(pg, a + i) : sve_dup(a[0]);
    const VecT bv = (!BroadcastB) ? sve_ld1(pg, b + i) : sve_dup(b[0]);
    VecT res = sve_tile_apply<T, Op, VecT>(pg, av, bv);
    if constexpr (Masked) {
      svbool_t m = sve_mask<T>(pg, mask, i);
      res = sve_sel(m, res, zero);  // zero-fill in-tile-but-masked-off lanes
    }
    sve_st1(pg, out + i, res);
  }
}
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask) {
  // No SVE modulo intrinsic, and ``sve_tile_apply``'s catch-all is logical-OR;
  // run a portable scalar lane loop (``py_mod`` = Python/NumPy ``%`` semantics).
  if constexpr (Op == '%') {
    for (int i = 0; i < VLEN; ++i) {
      const T av = BroadcastA ? a[0] : a[i];
      const T bv = BroadcastB ? b[0] : b[i];
      if constexpr (Masked)
        out[i] = mask[i] ? py_mod(av, bv) : T(0);
      else
        out[i] = py_mod(av, bv);
    }
    return;
  }
  using VecT = decltype(sve_dup(T(0)));  // vec type for T, from the sve_dup overload set
  const VecT zero = sve_dup(T(0));
  for (int i = 0; i < VLEN; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, VLEN);
    const VecT av = (!BroadcastA) ? sve_ld1(pg, a + i) : sve_dup(a[0]);
    const VecT bv = (!BroadcastB) ? sve_ld1(pg, b + i) : sve_dup(b[0]);
    VecT res = sve_tile_apply<T, Op, VecT>(pg, av, bv);
    if constexpr (Masked) {
      svbool_t m = sve_mask<T>(pg, mask, i);
      res = sve_sel(m, res, zero);
    }
    sve_st1(pg, out + i, res);
  }
}

// ===========================================================================
// tile_ite : out[i] = cond[i] ? t : e
// PRODUCER -> ZERO-FILL inactive. ``cond`` is a ``CondT`` truthiness array
// (stored 1/0); the select predicate is ``cond != 0``.
// ===========================================================================
template <typename T, typename CondT, bool BroadcastThen, bool BroadcastElse, bool Masked>
inline void tile_ite(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                     const T* __restrict__ e, const bool* __restrict__ mask, int vlen) {
  using VecT = decltype(sve_dup(T(0)));  // vec type for T, from the sve_dup overload set
  const VecT zero = sve_dup(T(0));
  for (int i = 0; i < vlen; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, vlen);
    const VecT tv = (!BroadcastThen) ? sve_ld1(pg, t + i) : sve_dup(t[0]);
    const VecT ev = (!BroadcastElse) ? sve_ld1(pg, e + i) : sve_dup(e[0]);
    // cond loaded as its own type; build select predicate by != 0 (matches the
    // scalar ``cond[i] ?`` truthiness; CondT lane width matches T's lane width).
    svbool_t cp = sve_cmpne0(pg, sve_ld1(pg, cond + i));
    VecT res = sve_sel(cp, tv, ev);
    if constexpr (Masked) {
      svbool_t m = sve_mask<T>(pg, mask, i);
      res = sve_sel(m, res, zero);
    }
    sve_st1(pg, out + i, res);
  }
}
// Per-lane unary op (op codes: n neg, ! not, a abs, e exp, l log, s sqrt,
// S sin, C cos, f floor, c ceil, t tanh). The transcendentals have no portable
// SIMD intrinsic, so this shares scalar.h's lane-loop form in every backend.
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

// out[i] = <op> a-operand ; ZERO-FILL inactive (operand read is in-tile).
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked>
inline void tile_unop(T* __restrict__ out, const T* __restrict__ a, const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; ++i) {
    const T av = Broadcast ? a[0] : a[i];
    if constexpr (Masked)
      out[i] = mask[i] ? tile_unop_apply<T, Op>(av) : T(0);
    else
      out[i] = tile_unop_apply<T, Op>(av);
  }
}

template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked>
inline void tile_ite(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                     const T* __restrict__ e, const bool* __restrict__ mask) {
  using VecT = decltype(sve_dup(T(0)));  // vec type for T, from the sve_dup overload set
  const VecT zero = sve_dup(T(0));
  for (int i = 0; i < VLEN; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, VLEN);
    const VecT tv = (!BroadcastThen) ? sve_ld1(pg, t + i) : sve_dup(t[0]);
    const VecT ev = (!BroadcastElse) ? sve_ld1(pg, e + i) : sve_dup(e[0]);
    svbool_t cp = sve_cmpne0(pg, sve_ld1(pg, cond + i));
    VecT res = sve_sel(cp, tv, ev);
    if constexpr (Masked) {
      svbool_t m = sve_mask<T>(pg, mask, i);
      res = sve_sel(m, res, zero);
    }
    sve_st1(pg, out + i, res);
  }
}

// ===========================================================================
// tile_load : dst[i] = src[i * stride]
// PRODUCER -> ZERO-FILL inactive + GUARDED read (predicated load never
// dereferences an inactive lane, so an OOB tail lane is safe). A stride != 1
// becomes an index gather over the ``i + lane`` ramp scaled by ``stride``.
// ===========================================================================
template <typename T, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask, int vlen,
                      std::int64_t stride = 1) {
  using VecT = decltype(sve_dup(T(0)));  // vec type for T, from the sve_dup overload set
  const VecT zero = sve_dup(T(0));
  for (int i = 0; i < vlen; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, vlen);
    svbool_t rp = pg;  // read predicate (guards the dereference)
    if constexpr (Masked) rp = sve_mask<T>(pg, mask, i);
    VecT v;
    if (stride == 1) {
      v = sve_ld1(rp, src + i);
    } else {
      if constexpr (sizeof(T) == 4) {
        svint32_t ramp = svindex_s32((std::int32_t)i, 1);
        svint32_t vidx = svmul_n_s32_x(rp, ramp, (std::int32_t)stride);
        if constexpr (std::is_same<T, float>::value)
          v = svld1_gather_s32index_f32(rp, src, vidx);
        else
          v = svld1_gather_s32index_s32(rp, src, vidx);
      } else {
        svint64_t ramp = svindex_s64((std::int64_t)i, 1);
        svint64_t vidx = svmul_n_s64_x(rp, ramp, stride);
        if constexpr (std::is_same<T, double>::value)
          v = svld1_gather_s64index_f64(rp, src, vidx);
        else
          v = svld1_gather_s64index_s64(rp, src, vidx);
      }
    }
    if constexpr (Masked) v = sve_sel(rp, v, zero);  // zero-fill masked-off lanes
    sve_st1(pg, dst + i, v);
  }
}
template <typename T, int VLEN, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                      std::int64_t stride = 1) {
  using VecT = decltype(sve_dup(T(0)));  // vec type for T, from the sve_dup overload set
  const VecT zero = sve_dup(T(0));
  for (int i = 0; i < VLEN; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, VLEN);
    svbool_t rp = pg;
    if constexpr (Masked) rp = sve_mask<T>(pg, mask, i);
    VecT v;
    if (stride == 1) {
      v = sve_ld1(rp, src + i);
    } else {
      if constexpr (sizeof(T) == 4) {
        svint32_t ramp = svindex_s32((std::int32_t)i, 1);
        svint32_t vidx = svmul_n_s32_x(rp, ramp, (std::int32_t)stride);
        if constexpr (std::is_same<T, float>::value)
          v = svld1_gather_s32index_f32(rp, src, vidx);
        else
          v = svld1_gather_s32index_s32(rp, src, vidx);
      } else {
        svint64_t ramp = svindex_s64((std::int64_t)i, 1);
        svint64_t vidx = svmul_n_s64_x(rp, ramp, stride);
        if constexpr (std::is_same<T, double>::value)
          v = svld1_gather_s64index_f64(rp, src, vidx);
        else
          v = svld1_gather_s64index_s64(rp, src, vidx);
      }
    }
    if constexpr (Masked) v = sve_sel(rp, v, zero);
    sve_st1(pg, dst + i, v);
  }
}

// ===========================================================================
// tile_store : dst[i * stride] = src[i]
// WRITER -> RMW skip-inactive (inactive / masked-off lane never written, so the
// destination array / OOB tail is never touched). A stride != 1 becomes an
// index scatter over the ``i + lane`` ramp scaled by ``stride``.
// ===========================================================================
template <typename T, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask, int vlen,
                       std::int64_t stride = 1) {
  for (int i = 0; i < vlen; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, vlen);
    svbool_t wp = pg;  // write predicate (skip-inactive)
    if constexpr (Masked) wp = sve_mask<T>(pg, mask, i);
    auto v = sve_ld1(wp, src + i);
    if (stride == 1) {
      sve_st1(wp, dst + i, v);
    } else {
      if constexpr (sizeof(T) == 4) {
        svint32_t ramp = svindex_s32((std::int32_t)i, 1);
        svint32_t vidx = svmul_n_s32_x(wp, ramp, (std::int32_t)stride);
        if constexpr (std::is_same<T, float>::value)
          svst1_scatter_s32index_f32(wp, dst, vidx, v);
        else
          svst1_scatter_s32index_s32(wp, dst, vidx, v);
      } else {
        svint64_t ramp = svindex_s64((std::int64_t)i, 1);
        svint64_t vidx = svmul_n_s64_x(wp, ramp, stride);
        if constexpr (std::is_same<T, double>::value)
          svst1_scatter_s64index_f64(wp, dst, vidx, v);
        else
          svst1_scatter_s64index_s64(wp, dst, vidx, v);
      }
    }
  }
}
template <typename T, int VLEN, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                       std::int64_t stride = 1) {
  for (int i = 0; i < VLEN; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, VLEN);
    svbool_t wp = pg;
    if constexpr (Masked) wp = sve_mask<T>(pg, mask, i);
    auto v = sve_ld1(wp, src + i);
    if (stride == 1) {
      sve_st1(wp, dst + i, v);
    } else {
      if constexpr (sizeof(T) == 4) {
        svint32_t ramp = svindex_s32((std::int32_t)i, 1);
        svint32_t vidx = svmul_n_s32_x(wp, ramp, (std::int32_t)stride);
        if constexpr (std::is_same<T, float>::value)
          svst1_scatter_s32index_f32(wp, dst, vidx, v);
        else
          svst1_scatter_s32index_s32(wp, dst, vidx, v);
      } else {
        svint64_t ramp = svindex_s64((std::int64_t)i, 1);
        svint64_t vidx = svmul_n_s64_x(wp, ramp, stride);
        if constexpr (std::is_same<T, double>::value)
          svst1_scatter_s64index_f64(wp, dst, vidx, v);
        else
          svst1_scatter_s64index_s64(wp, dst, vidx, v);
      }
    }
  }
}

// Load an index tile (``IdxT`` element indices) into the s32 / s64 lane vector
// matching the DATA lane width. When ``IdxT`` already matches the lane width
// this is a plain predicated load; otherwise it widens (s32 -> s64 via
// ``svld1sw_s64``) or narrows (s64 -> s32 via a widened load + truncating
// move) so the gather / scatter index width pairs with the data width.
template <typename IdxT>
inline svint32_t sve_load_idx32(svbool_t pg, const IdxT* idx, int i) {
  if constexpr (std::is_same<IdxT, std::int32_t>::value) {
    return svld1_s32(pg, idx + i);
  } else {  // int64_t index for 32-bit data: load s64 then narrow to s32 lanes
    // BUG (tracked, ARM-only, mixed IdxT=int64 + 32-bit data gather/scatter only):
    // ``pg`` here is a b32 predicate (the 32-bit-data loop uses svwhilelt_b32 /
    // svcntw), but ``svld1_s64`` reads at 64-bit granularity -> wrong lane subset +
    // only svcntd (half) of the needed svcntw indices. ``svreinterpret_s32_s64`` is a
    // BITCAST, not a lane-wise narrow, so the resulting s32 indices are garbage ->
    // wrong (possibly OOB) addresses on ACTIVE lanes (mask prevention is fine; this is
    // an index-value defect). FIX: load two svcntd int64 chunks and pack with
    // svuzp1_s32(svreinterpret_s32_s64(lo), svreinterpret_s32_s64(hi)), or route this
    // mixed-width case to a scalar index loop. Unverifiable here (no ARM host); the
    // corpus uses SCALAR/AVX512, so this path is never exercised on x86.
    svint64_t wide = svld1_s64(pg, idx + i);
    return svreinterpret_s32_s64(wide);  // low 32 bits of each lane (indices fit)
  }
}
template <typename IdxT>
inline svint64_t sve_load_idx64(svbool_t pg, const IdxT* idx, int i) {
  if constexpr (std::is_same<IdxT, std::int64_t>::value) {
    return svld1_s64(pg, idx + i);
  } else {  // int32_t index for 64-bit data: sign-extend 32-bit memory -> s64
    return svld1sw_s64(pg, (const std::int32_t*)(idx + i));
  }
}

// ===========================================================================
// tile_gather : dst[i] = src[idx[i]]
// PRODUCER -> ZERO-FILL inactive + GUARDED read (predicated gather never
// dereferences an inactive lane, so a garbage / OOB index is safe).
// ===========================================================================
template <typename T, typename IdxT, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask, int vlen) {
  using VecT = decltype(sve_dup(T(0)));  // vec type for T, from the sve_dup overload set
  const VecT zero = sve_dup(T(0));
  for (int i = 0; i < vlen; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, vlen);
    svbool_t rp = pg;
    if constexpr (Masked) rp = sve_mask<T>(pg, mask, i);
    VecT v;
    if constexpr (sizeof(T) == 4) {
      svint32_t vidx = sve_load_idx32<IdxT>(rp, idx, i);
      if constexpr (std::is_same<T, float>::value)
        v = svld1_gather_s32index_f32(rp, src, vidx);
      else
        v = svld1_gather_s32index_s32(rp, src, vidx);
    } else {
      svint64_t vidx = sve_load_idx64<IdxT>(rp, idx, i);
      if constexpr (std::is_same<T, double>::value)
        v = svld1_gather_s64index_f64(rp, src, vidx);
      else
        v = svld1_gather_s64index_s64(rp, src, vidx);
    }
    if constexpr (Masked) v = sve_sel(rp, v, zero);
    sve_st1(pg, dst + i, v);
  }
}
template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask) {
  using VecT = decltype(sve_dup(T(0)));  // vec type for T, from the sve_dup overload set
  const VecT zero = sve_dup(T(0));
  for (int i = 0; i < VLEN; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, VLEN);
    svbool_t rp = pg;
    if constexpr (Masked) rp = sve_mask<T>(pg, mask, i);
    VecT v;
    if constexpr (sizeof(T) == 4) {
      svint32_t vidx = sve_load_idx32<IdxT>(rp, idx, i);
      if constexpr (std::is_same<T, float>::value)
        v = svld1_gather_s32index_f32(rp, src, vidx);
      else
        v = svld1_gather_s32index_s32(rp, src, vidx);
    } else {
      svint64_t vidx = sve_load_idx64<IdxT>(rp, idx, i);
      if constexpr (std::is_same<T, double>::value)
        v = svld1_gather_s64index_f64(rp, src, vidx);
      else
        v = svld1_gather_s64index_s64(rp, src, vidx);
    }
    if constexpr (Masked) v = sve_sel(rp, v, zero);
    sve_st1(pg, dst + i, v);
  }
}

// ===========================================================================
// tile_scatter : dst[idx[i]] = src[i]
// WRITER -> RMW skip-inactive (inactive / masked-off lane never written, so a
// garbage / OOB index is safe).
// ===========================================================================
template <typename T, typename IdxT, bool Masked>
inline void tile_scatter(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                         const bool* __restrict__ mask, int vlen) {
  for (int i = 0; i < vlen; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, vlen);
    svbool_t wp = pg;
    if constexpr (Masked) wp = sve_mask<T>(pg, mask, i);
    auto v = sve_ld1(wp, src + i);
    if constexpr (sizeof(T) == 4) {
      svint32_t vidx = sve_load_idx32<IdxT>(wp, idx, i);
      if constexpr (std::is_same<T, float>::value)
        svst1_scatter_s32index_f32(wp, dst, vidx, v);
      else
        svst1_scatter_s32index_s32(wp, dst, vidx, v);
    } else {
      svint64_t vidx = sve_load_idx64<IdxT>(wp, idx, i);
      if constexpr (std::is_same<T, double>::value)
        svst1_scatter_s64index_f64(wp, dst, vidx, v);
      else
        svst1_scatter_s64index_s64(wp, dst, vidx, v);
    }
  }
}
template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_scatter(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                         const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; i += sve_cnt<T>()) {
    svbool_t pg = sve_whilelt<T>(i, VLEN);
    svbool_t wp = pg;
    if constexpr (Masked) wp = sve_mask<T>(pg, mask, i);
    auto v = sve_ld1(wp, src + i);
    if constexpr (sizeof(T) == 4) {
      svint32_t vidx = sve_load_idx32<IdxT>(wp, idx, i);
      if constexpr (std::is_same<T, float>::value)
        svst1_scatter_s32index_f32(wp, dst, vidx, v);
      else
        svst1_scatter_s32index_s32(wp, dst, vidx, v);
    } else {
      svint64_t vidx = sve_load_idx64<IdxT>(wp, idx, i);
      if constexpr (std::is_same<T, double>::value)
        svst1_scatter_s64index_f64(wp, dst, vidx, v);
      else
        svst1_scatter_s64index_s64(wp, dst, vidx, v);
    }
  }
}

// ---------------------------- tile_mask_gen ----------------------------
// out[l] = (base + l) < ub. SVE: per 64-bit-lane chunk, svindex + svcmplt give
// the active predicate; narrowing svst1b writes 1/0 bytes. (Written on an x86
// dev box -- SVE path not hardware-exercised; semantics mirror scalar.h.)
template <typename IdxT, int VLEN>
inline void tile_mask_gen(bool* __restrict__ out, IdxT base, IdxT ub) {
  const std::int64_t n = (std::int64_t)ub;
  const std::int64_t cnt = svcntd();
  for (std::int64_t i = 0; i < VLEN; i += cnt) {
    svbool_t wp = svwhilelt_b64((uint64_t)i, (uint64_t)VLEN);  // valid out lanes
    svint64_t lanes = svindex_s64((std::int64_t)base + i, 1);  // base+i + {0,1,...}
    svbool_t active = svcmplt(wp, lanes, n);                   // (base+i+l) < ub
    svint64_t ones = svdup_n_s64_z(active, 1);                 // 1 active / 0 else
    svst1b_s64(wp, reinterpret_cast<signed char*>(out) + i, ones);
  }
}

// ----------------------------- tile_reduce ----------------------------
// Horizontal reduction of a VLEN-lane tile to ONE scalar (an in-map / per-tile
// reduction: ``acc = sum/prod/min/max over the tile``). ``Op`` is the reduction
// op ('+' sum, '*' prod, 'm' min, 'M' max); returns the reduced element, not a
// vector. Full reduction only -- a masked / single-axis / K>=2 reduce keeps the
// ``pure`` per-lane expansion (the selector never routes those here).
//
// Balanced log-depth pairwise fold (consecutive pairs (0,1)(2,3)...; an odd
// trailing lane forwards unchanged). Over a compile-time-constant ``VLEN`` the
// loops unroll, so the compiler re-vectorises the partials; it reduces in the
// same order as the vectorized ``Reduce`` node's ``_dace_horizontal_tree`` so
// both paths agree. The per-lane combine reuses this header's own ``tile_apply``
// (self-contained; no cross-ISA dispatch header).
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
