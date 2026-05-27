// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// AVX2 backend of the K=1 tile-op intrinsics. Mirrors the SAME function
// signatures as the portable scalar reference
// (``dace/runtime/include/dace/tile_ops/scalar.h``); the tile-op node's
// chosen-backend expansion pulls in exactly one of the sibling headers, so this
// header must NOT be included together with ``scalar.h`` (both define the same
// functions). Ops are selected by a one-char ``Op`` template parameter (see
// the scalar.h legend); operand broadcast is a ``bool`` parameter -- no enums.
//
// AVX2 specifics versus the AVX-512 sibling:
//   * No opmask registers and no masked-arithmetic forms. The masking idiom is
//     "compute the full vector, then BLEND-with-zero (producers) or VPMASKMOV
//     (array writers) to commit only active lanes". A lane mask is a vector
//     (``__m256``/``__m256d``/``__m256i``) whose per-lane HIGH BIT selects the
//     lane (all-ones = active, zero = inactive).
//   * Several integer ops have no AVX2 intrinsic (int64 mul/min/max, integer
//     div, scatter, strided store) -- those fall back to a correct per-lane
//     scalar loop, exactly like the AVX-512 sibling falls back for unsupported
//     dtypes. Correctness first, SIMD where the intrinsic exists.
//
// Masked semantics (load-bearing, two flavours, identical to scalar.h):
//   * tile PRODUCERS (binop / merge / load / gather) -- ZERO-FILL inactive lanes
//     and GUARD the read, so an inactive (e.g. out-of-bounds tail) lane never
//     dereferences ``src`` / ``src[idx]``. Implemented with maskload / mask-
//     gather (which do not touch inactive lanes) or a blend-with-zero of an
//     in-tile computation.
//   * array WRITERS (store / scatter) -- RMW skip-inactive: an inactive lane is
//     not written, so the destination array (and its OOB tail) is never touched.
//     Implemented with maskstore (load / store) or a guarded scalar loop
//     (scatter, strided store -- no AVX2 instruction).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>


#if !defined(__AVX2__)
#error Included the AVX2 tile-op header without AVX2 support
#endif

namespace dace {
namespace tileops {

// ===================================================================
// Scalar per-lane op (reference semantics; used by every scalar tail /
// fallback path so the AVX2 results are bit-for-bit the scalar contract).
// ===================================================================
template <typename T, char Op>
inline T tile_apply(T a, T b) {
  if constexpr (Op == '+') return a + b;
  else if constexpr (Op == '-') return a - b;
  else if constexpr (Op == '*') return a * b;
  else if constexpr (Op == '/') return a / b;
  else if constexpr (Op == 'm') return std::min(a, b);
  else if constexpr (Op == 'M') return std::max(a, b);
  else if constexpr (Op == '<') return (a < b) ? T(1) : T(0);
  else if constexpr (Op == 'l') return (a <= b) ? T(1) : T(0);
  else if constexpr (Op == '>') return (a > b) ? T(1) : T(0);
  else if constexpr (Op == 'g') return (a >= b) ? T(1) : T(0);
  else if constexpr (Op == '=') return (a == b) ? T(1) : T(0);
  else if constexpr (Op == '!') return (a != b) ? T(1) : T(0);
  else if constexpr (Op == '&') return (a && b) ? T(1) : T(0);
  else /* Or */ return (a || b) ? T(1) : T(0);
}

namespace detail {

// ---- lane-mask builders -------------------------------------------------
// AVX2 masks are vectors whose per-lane HIGH BIT selects the lane. Build from
// the ``const bool*`` tile mask: active lane -> all-ones (-1), inactive -> 0.
// The mask lane width must match the element width (32-bit for ps/epi32,
// 64-bit for pd/epi64).
inline __m256i mask32_from_bools(const bool* mask, int base, int n) {
  int32_t mbuf[8];
  for (int j = 0; j < 8; ++j) mbuf[j] = (j < n && mask[base + j]) ? -1 : 0;
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mbuf));
}
inline __m256i mask64_from_bools(const bool* mask, int base, int n) {
  int64_t mbuf[4];
  for (int j = 0; j < 4; ++j) mbuf[j] = (j < n && mask[base + j]) ? -1 : 0;
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mbuf));
}

// ---- FP arithmetic / compare on a full vector ---------------------------
// Returns the full (unmasked) op result; comparisons yield a 1.0/0.0 value
// tile (all-ones compare mask AND-ed with a splat of 1).
template <char Op>
inline __m256 binop_ps(__m256 a, __m256 b) {
  if constexpr (Op == '+') return _mm256_add_ps(a, b);
  else if constexpr (Op == '-') return _mm256_sub_ps(a, b);
  else if constexpr (Op == '*') return _mm256_mul_ps(a, b);
  else if constexpr (Op == '/') return _mm256_div_ps(a, b);
  else if constexpr (Op == 'm') return _mm256_min_ps(a, b);
  else if constexpr (Op == 'M') return _mm256_max_ps(a, b);
  else {
    const __m256 one = _mm256_set1_ps(1.0f);
    if constexpr (Op == '<') return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_LT_OQ), one);
    else if constexpr (Op == 'l') return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_LE_OQ), one);
    else if constexpr (Op == '>') return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_GT_OQ), one);
    else if constexpr (Op == 'g') return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_GE_OQ), one);
    else if constexpr (Op == '=') return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_EQ_OQ), one);
    else if constexpr (Op == '!') return _mm256_and_ps(_mm256_cmp_ps(a, b, _CMP_NEQ_UQ), one);
    else if constexpr (Op == '&') {
      const __m256 z = _mm256_setzero_ps();
      __m256 na = _mm256_cmp_ps(a, z, _CMP_NEQ_UQ);
      __m256 nb = _mm256_cmp_ps(b, z, _CMP_NEQ_UQ);
      return _mm256_and_ps(_mm256_and_ps(na, nb), one);
    } else {  // Or
      const __m256 z = _mm256_setzero_ps();
      __m256 na = _mm256_cmp_ps(a, z, _CMP_NEQ_UQ);
      __m256 nb = _mm256_cmp_ps(b, z, _CMP_NEQ_UQ);
      return _mm256_and_ps(_mm256_or_ps(na, nb), one);
    }
  }
}

template <char Op>
inline __m256d binop_pd(__m256d a, __m256d b) {
  if constexpr (Op == '+') return _mm256_add_pd(a, b);
  else if constexpr (Op == '-') return _mm256_sub_pd(a, b);
  else if constexpr (Op == '*') return _mm256_mul_pd(a, b);
  else if constexpr (Op == '/') return _mm256_div_pd(a, b);
  else if constexpr (Op == 'm') return _mm256_min_pd(a, b);
  else if constexpr (Op == 'M') return _mm256_max_pd(a, b);
  else {
    const __m256d one = _mm256_set1_pd(1.0);
    if constexpr (Op == '<') return _mm256_and_pd(_mm256_cmp_pd(a, b, _CMP_LT_OQ), one);
    else if constexpr (Op == 'l') return _mm256_and_pd(_mm256_cmp_pd(a, b, _CMP_LE_OQ), one);
    else if constexpr (Op == '>') return _mm256_and_pd(_mm256_cmp_pd(a, b, _CMP_GT_OQ), one);
    else if constexpr (Op == 'g') return _mm256_and_pd(_mm256_cmp_pd(a, b, _CMP_GE_OQ), one);
    else if constexpr (Op == '=') return _mm256_and_pd(_mm256_cmp_pd(a, b, _CMP_EQ_OQ), one);
    else if constexpr (Op == '!') return _mm256_and_pd(_mm256_cmp_pd(a, b, _CMP_NEQ_UQ), one);
    else if constexpr (Op == '&') {
      const __m256d z = _mm256_setzero_pd();
      __m256d na = _mm256_cmp_pd(a, z, _CMP_NEQ_UQ);
      __m256d nb = _mm256_cmp_pd(b, z, _CMP_NEQ_UQ);
      return _mm256_and_pd(_mm256_and_pd(na, nb), one);
    } else {  // Or
      const __m256d z = _mm256_setzero_pd();
      __m256d na = _mm256_cmp_pd(a, z, _CMP_NEQ_UQ);
      __m256d nb = _mm256_cmp_pd(b, z, _CMP_NEQ_UQ);
      return _mm256_and_pd(_mm256_or_pd(na, nb), one);
    }
  }
}

// ---- int32 op on a full vector ------------------------------------------
// ``handled32<Op>()`` is false for the ops with no AVX2 int32 intrinsic
// (Div); the caller routes those to the scalar fallback instead.
template <char Op>
constexpr bool handled32() {
  return Op != '/';
}
template <char Op>
inline __m256i binop_epi32(__m256i a, __m256i b) {
  const __m256i one = _mm256_set1_epi32(1);
  const __m256i ones = _mm256_set1_epi32(-1);
  if constexpr (Op == '+') return _mm256_add_epi32(a, b);
  else if constexpr (Op == '-') return _mm256_sub_epi32(a, b);
  else if constexpr (Op == '*') return _mm256_mullo_epi32(a, b);
  else if constexpr (Op == 'm') return _mm256_min_epi32(a, b);
  else if constexpr (Op == 'M') return _mm256_max_epi32(a, b);
  else if constexpr (Op == '=') return _mm256_and_si256(_mm256_cmpeq_epi32(a, b), one);
  else if constexpr (Op == '>') return _mm256_and_si256(_mm256_cmpgt_epi32(a, b), one);
  else if constexpr (Op == '<') return _mm256_and_si256(_mm256_cmpgt_epi32(b, a), one);
  else if constexpr (Op == 'g')  // ~(a<b) = ~cmpgt(b,a)
    return _mm256_and_si256(_mm256_andnot_si256(_mm256_cmpgt_epi32(b, a), ones), one);
  else if constexpr (Op == 'l')  // ~(a>b) = ~cmpgt(a,b)
    return _mm256_and_si256(_mm256_andnot_si256(_mm256_cmpgt_epi32(a, b), ones), one);
  else if constexpr (Op == '!')  // ~cmpeq(a,b)
    return _mm256_and_si256(_mm256_andnot_si256(_mm256_cmpeq_epi32(a, b), ones), one);
  else if constexpr (Op == '&' || Op == '|') {
    const __m256i z = _mm256_setzero_si256();
    __m256i na = _mm256_andnot_si256(_mm256_cmpeq_epi32(a, z), ones);  // nonzero(a)
    __m256i nb = _mm256_andnot_si256(_mm256_cmpeq_epi32(b, z), ones);  // nonzero(b)
    __m256i comb = (Op == '&') ? _mm256_and_si256(na, nb) : _mm256_or_si256(na, nb);
    return _mm256_and_si256(comb, one);
  } else {  // Div -- unreachable (handled32<Div>() == false)
    return a;
  }
}

}  // namespace detail

// ===================================================================
// tile_binop : out[i] = a-operand <op> b-operand ; ZERO-FILL inactive.
// Operand reads are in-tile (always safe to evaluate), so masking only
// affects which lanes survive into ``out`` (inactive -> T(0)).
// ===================================================================
template <typename T, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask, int vlen) {
  auto scalar_tail = [&](int i) {
    const T av = (!BroadcastA) ? a[i] : a[0];
    const T bv = (!BroadcastB) ? b[i] : b[0];
    if constexpr (Masked) out[i] = mask[i] ? tile_apply<T, Op>(av, bv) : T(0);
    else out[i] = tile_apply<T, Op>(av, bv);
  };

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 8;
    const __m256 avb = (BroadcastA) ? _mm256_set1_ps(a[0]) : _mm256_setzero_ps();
    const __m256 bvb = (BroadcastB) ? _mm256_set1_ps(b[0]) : _mm256_setzero_ps();
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256 va = (!BroadcastA) ? _mm256_loadu_ps(a + i) : avb;
      __m256 vb = (!BroadcastB) ? _mm256_loadu_ps(b + i) : bvb;
      __m256 res = detail::binop_ps<Op>(va, vb);
      if constexpr (Masked)
        res = _mm256_blendv_ps(_mm256_setzero_ps(), res, _mm256_castsi256_ps(detail::mask32_from_bools(mask, i, W)));
      _mm256_storeu_ps(out + i, res);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 4;
    const __m256d avb = (BroadcastA) ? _mm256_set1_pd(a[0]) : _mm256_setzero_pd();
    const __m256d bvb = (BroadcastB) ? _mm256_set1_pd(b[0]) : _mm256_setzero_pd();
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256d va = (!BroadcastA) ? _mm256_loadu_pd(a + i) : avb;
      __m256d vb = (!BroadcastB) ? _mm256_loadu_pd(b + i) : bvb;
      __m256d res = detail::binop_pd<Op>(va, vb);
      if constexpr (Masked)
        res = _mm256_blendv_pd(_mm256_setzero_pd(), res, _mm256_castsi256_pd(detail::mask64_from_bools(mask, i, W)));
      _mm256_storeu_pd(out + i, res);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int32_t>::value) {
    if constexpr (detail::handled32<Op>()) {
      constexpr int W = 8;
      const __m256i avb = (BroadcastA) ? _mm256_set1_epi32(a[0]) : _mm256_setzero_si256();
      const __m256i bvb = (BroadcastB) ? _mm256_set1_epi32(b[0]) : _mm256_setzero_si256();
      int i = 0;
      for (; i + W <= vlen; i += W) {
        __m256i va = (!BroadcastA) ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)) : avb;
        __m256i vb = (!BroadcastB) ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i)) : bvb;
        __m256i res = detail::binop_epi32<Op>(va, vb);
        if constexpr (Masked)
          res = _mm256_blendv_epi8(_mm256_setzero_si256(), res, detail::mask32_from_bools(mask, i, W));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), res);
      }
      for (; i < vlen; ++i) scalar_tail(i);
      return;
    }
    // int32 Div: no AVX2 integer-division intrinsic -> scalar fallback.
    for (int i = 0; i < vlen; ++i) scalar_tail(i);
    return;
  }
  // int64 and any other dtype: int64 Mult/Min/Max/Div have no AVX2 intrinsic;
  // keep the whole int64 path scalar for a single correct implementation
  // (Add/Sub could vectorise but uniformity beats partial coverage here).
  for (int i = 0; i < vlen; ++i) scalar_tail(i);
}

template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask) {
  tile_binop<T, Op, BroadcastA, BroadcastB, Masked>(out, a, b, mask, VLEN);
}

// ===================================================================
// tile_merge : out[i] = cond[i] ? t : e ; ZERO-FILL inactive.
// The select is via blendv (per-lane high-bit), with cond built as an
// all-ones/zero vector mask from the cond tile.
// ===================================================================
template <typename T, typename CondT, bool BroadcastThen, bool BroadcastElse, bool Masked>
inline void tile_merge(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                       const T* __restrict__ e, const bool* __restrict__ mask, int vlen) {
  auto scalar_tail = [&](int i) {
    const T tv = (!BroadcastThen) ? t[i] : t[0];
    const T ev = (!BroadcastElse) ? e[i] : e[0];
    if constexpr (Masked) out[i] = mask[i] ? (cond[i] ? tv : ev) : T(0);
    else out[i] = cond[i] ? tv : ev;
  };

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 8;
    const __m256 tvb = (BroadcastThen) ? _mm256_set1_ps(t[0]) : _mm256_setzero_ps();
    const __m256 evb = (BroadcastElse) ? _mm256_set1_ps(e[0]) : _mm256_setzero_ps();
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256 vt = (!BroadcastThen) ? _mm256_loadu_ps(t + i) : tvb;
      __m256 ve = (!BroadcastElse) ? _mm256_loadu_ps(e + i) : evb;
      int32_t cbuf[8];
      for (int j = 0; j < W; ++j) cbuf[j] = cond[i + j] ? -1 : 0;
      __m256 cmask = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cbuf)));
      __m256 res = _mm256_blendv_ps(ve, vt, cmask);  // high-bit(cmask)? vt : ve
      if constexpr (Masked)
        res = _mm256_blendv_ps(_mm256_setzero_ps(), res, _mm256_castsi256_ps(detail::mask32_from_bools(mask, i, W)));
      _mm256_storeu_ps(out + i, res);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 4;
    const __m256d tvb = (BroadcastThen) ? _mm256_set1_pd(t[0]) : _mm256_setzero_pd();
    const __m256d evb = (BroadcastElse) ? _mm256_set1_pd(e[0]) : _mm256_setzero_pd();
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256d vt = (!BroadcastThen) ? _mm256_loadu_pd(t + i) : tvb;
      __m256d ve = (!BroadcastElse) ? _mm256_loadu_pd(e + i) : evb;
      int64_t cbuf[4];
      for (int j = 0; j < W; ++j) cbuf[j] = cond[i + j] ? -1 : 0;
      __m256d cmask = _mm256_castsi256_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cbuf)));
      __m256d res = _mm256_blendv_pd(ve, vt, cmask);
      if constexpr (Masked)
        res = _mm256_blendv_pd(_mm256_setzero_pd(), res, _mm256_castsi256_pd(detail::mask64_from_bools(mask, i, W)));
      _mm256_storeu_pd(out + i, res);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int32_t>::value) {
    constexpr int W = 8;
    const __m256i tvb = (BroadcastThen) ? _mm256_set1_epi32(t[0]) : _mm256_setzero_si256();
    const __m256i evb = (BroadcastElse) ? _mm256_set1_epi32(e[0]) : _mm256_setzero_si256();
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256i vt = (!BroadcastThen) ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(t + i)) : tvb;
      __m256i ve = (!BroadcastElse) ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(e + i)) : evb;
      int32_t cbuf[8];
      for (int j = 0; j < W; ++j) cbuf[j] = cond[i + j] ? -1 : 0;
      __m256i cmask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cbuf));
      __m256i res = _mm256_blendv_epi8(ve, vt, cmask);  // byte-granular ok (cond lanes uniform)
      if constexpr (Masked)
        res = _mm256_blendv_epi8(_mm256_setzero_si256(), res, detail::mask32_from_bools(mask, i, W));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), res);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int64_t>::value) {
    constexpr int W = 4;
    const __m256i tvb = (BroadcastThen) ? _mm256_set1_epi64x(t[0]) : _mm256_setzero_si256();
    const __m256i evb = (BroadcastElse) ? _mm256_set1_epi64x(e[0]) : _mm256_setzero_si256();
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256i vt = (!BroadcastThen) ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(t + i)) : tvb;
      __m256i ve = (!BroadcastElse) ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(e + i)) : evb;
      int64_t cbuf[4];
      for (int j = 0; j < W; ++j) cbuf[j] = cond[i + j] ? -1 : 0;
      __m256i cmask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cbuf));
      __m256i res = _mm256_blendv_epi8(ve, vt, cmask);
      if constexpr (Masked)
        res = _mm256_blendv_epi8(_mm256_setzero_si256(), res, detail::mask64_from_bools(mask, i, W));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), res);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  for (int i = 0; i < vlen; ++i) scalar_tail(i);
}

// Per-lane unary op (op codes: n neg, a abs, e exp, l log, s sqrt, S sin,
// C cos, f floor, c ceil, t tanh). The transcendentals have no portable SIMD
// intrinsic, so this shares scalar.h's lane-loop form in every backend.
template <typename T, char Op>
inline T tile_unop_apply(T a) {
  if constexpr (Op == 'n') return -a;
  else if constexpr (Op == 'a') return std::abs(a);
  else if constexpr (Op == 'e') return std::exp(a);
  else if constexpr (Op == 'l') return std::log(a);
  else if constexpr (Op == 's') return std::sqrt(a);
  else if constexpr (Op == 'S') return std::sin(a);
  else if constexpr (Op == 'C') return std::cos(a);
  else if constexpr (Op == 'f') return std::floor(a);
  else if constexpr (Op == 'c') return std::ceil(a);
  else /* 't' */ return std::tanh(a);
}

// out[i] = <op> a-operand ; ZERO-FILL inactive (operand read is in-tile).
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked>
inline void tile_unop(T* __restrict__ out, const T* __restrict__ a, const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; ++i) {
    const T av = Broadcast ? a[0] : a[i];
    if constexpr (Masked) out[i] = mask[i] ? tile_unop_apply<T, Op>(av) : T(0);
    else out[i] = tile_unop_apply<T, Op>(av);
  }
}

template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked>
inline void tile_merge(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                       const T* __restrict__ e, const bool* __restrict__ mask) {
  tile_merge<T, CondT, BroadcastThen, BroadcastElse, Masked>(out, cond, t, e, mask, VLEN);
}

// ===================================================================
// tile_load : dst[i] = src[i * stride] ; ZERO-FILL inactive + GUARDED read.
// Contiguous (stride == 1): vector loadu (unmasked) / maskload (masked,
// inactive lanes are NOT read -> OOB tail safe). Strided: scalar guarded loop.
// ===================================================================
template <typename T, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                      int vlen, std::int64_t stride = 1) {
  auto scalar_tail = [&](int i) {
    if constexpr (Masked) dst[i] = mask[i] ? src[i * stride] : T(0);
    else dst[i] = src[i * stride];
  };

  if (stride != 1) {  // no native strided load -> scalar guarded loop
    for (int i = 0; i < vlen; ++i) scalar_tail(i);
    return;
  }

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256 v;
      if constexpr (Masked)
        v = _mm256_maskload_ps(src + i, detail::mask32_from_bools(mask, i, W));
      else
        v = _mm256_loadu_ps(src + i);
      _mm256_storeu_ps(dst + i, v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256d v;
      if constexpr (Masked)
        v = _mm256_maskload_pd(src + i, detail::mask64_from_bools(mask, i, W));
      else
        v = _mm256_loadu_pd(src + i);
      _mm256_storeu_pd(dst + i, v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int32_t>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256i v;
      if constexpr (Masked)
        v = _mm256_maskload_epi32(reinterpret_cast<const int*>(src + i), detail::mask32_from_bools(mask, i, W));
      else
        v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int64_t>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256i v;
      if constexpr (Masked)
        v = _mm256_maskload_epi64(reinterpret_cast<const long long*>(src + i), detail::mask64_from_bools(mask, i, W));
      else
        v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  for (int i = 0; i < vlen; ++i) scalar_tail(i);
}

template <typename T, int VLEN, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                      std::int64_t stride = 1) {
  tile_load<T, Masked>(dst, src, mask, VLEN, stride);
}

// ===================================================================
// tile_store : dst[i * stride] = src[i] ; RMW skip-inactive.
// Contiguous unmasked: vector storeu. Masked: maskstore (only active lanes
// touch memory -> OOB tail never written). Strided: scalar guarded loop
// (no native strided/scatter store in AVX2).
// ===================================================================
template <typename T, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                       int vlen, std::int64_t stride = 1) {
  auto scalar_tail = [&](int i) {
    if constexpr (Masked) { if (mask[i]) dst[i * stride] = src[i]; }
    else dst[i * stride] = src[i];
  };

  if (stride != 1) {  // no native strided/scatter store -> scalar guarded loop
    for (int i = 0; i < vlen; ++i) scalar_tail(i);
    return;
  }

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256 v = _mm256_loadu_ps(src + i);
      if constexpr (Masked)
        _mm256_maskstore_ps(dst + i, detail::mask32_from_bools(mask, i, W), v);
      else
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256d v = _mm256_loadu_pd(src + i);
      if constexpr (Masked)
        _mm256_maskstore_pd(dst + i, detail::mask64_from_bools(mask, i, W), v);
      else
        _mm256_storeu_pd(dst + i, v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int32_t>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
      if constexpr (Masked)
        _mm256_maskstore_epi32(reinterpret_cast<int*>(dst + i), detail::mask32_from_bools(mask, i, W), v);
      else
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int64_t>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
      if constexpr (Masked)
        _mm256_maskstore_epi64(reinterpret_cast<long long*>(dst + i), detail::mask64_from_bools(mask, i, W), v);
      else
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  for (int i = 0; i < vlen; ++i) scalar_tail(i);
}

template <typename T, int VLEN, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                       std::int64_t stride = 1) {
  tile_store<T, Masked>(dst, src, mask, VLEN, stride);
}

// ===================================================================
// tile_gather : dst[i] = src[idx[i]] ; ZERO-FILL inactive + GUARDED read.
// Uses the AVX2 vgather family with an index vector whose lane width matches
// the element lane count (8 lanes for fp32/int32, 4 lanes for fp64/int64).
// The tile-op IdxT may be a different width than the element, so the index
// is copied into the matching-width buffer first. Masked: mask-gather, whose
// inactive lanes are NOT dereferenced (garbage / OOB index is safe).
// ===================================================================
template <typename T, typename IdxT, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask, int vlen) {
  auto scalar_tail = [&](int i) {
    if constexpr (Masked) dst[i] = mask[i] ? src[idx[i]] : T(0);
    else dst[i] = src[idx[i]];
  };

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 8;  // 8 fp32 lanes <-> 8x i32 index vector
    int i = 0;
    for (; i + W <= vlen; i += W) {
      int32_t ibuf[8];
      for (int j = 0; j < W; ++j) ibuf[j] = static_cast<int32_t>(idx[i + j]);
      __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ibuf));
      __m256 v;
      if constexpr (Masked) {
        __m256 m = _mm256_castsi256_ps(detail::mask32_from_bools(mask, i, W));
        v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), src, vindex, m, 4);
      } else {
        v = _mm256_i32gather_ps(src, vindex, 4);
      }
      _mm256_storeu_ps(dst + i, v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 4;  // 4 fp64 lanes <-> 4x i32 index (__m128i)
    int i = 0;
    for (; i + W <= vlen; i += W) {
      int32_t ibuf[4];
      for (int j = 0; j < W; ++j) ibuf[j] = static_cast<int32_t>(idx[i + j]);
      __m128i vindex = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ibuf));
      __m256d v;
      if constexpr (Masked) {
        __m256d m = _mm256_castsi256_pd(detail::mask64_from_bools(mask, i, W));
        v = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), src, vindex, m, 8);
      } else {
        v = _mm256_i32gather_pd(src, vindex, 8);
      }
      _mm256_storeu_pd(dst + i, v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int32_t>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vlen; i += W) {
      int32_t ibuf[8];
      for (int j = 0; j < W; ++j) ibuf[j] = static_cast<int32_t>(idx[i + j]);
      __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ibuf));
      __m256i v;
      if constexpr (Masked) {
        __m256i m = detail::mask32_from_bools(mask, i, W);
        v = _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), reinterpret_cast<const int*>(src), vindex, m, 4);
      } else {
        v = _mm256_i32gather_epi32(reinterpret_cast<const int*>(src), vindex, 4);
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  if constexpr (std::is_same<T, int64_t>::value) {
    constexpr int W = 4;  // 4 int64 lanes <-> 4x i64 index vector
    int i = 0;
    for (; i + W <= vlen; i += W) {
      int64_t ibuf[4];
      for (int j = 0; j < W; ++j) ibuf[j] = static_cast<int64_t>(idx[i + j]);
      __m256i vindex = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ibuf));
      __m256i v;
      if constexpr (Masked) {
        __m256i m = detail::mask64_from_bools(mask, i, W);
        v = _mm256_mask_i64gather_epi64(_mm256_setzero_si256(), reinterpret_cast<const long long*>(src), vindex, m, 8);
      } else {
        v = _mm256_i64gather_epi64(reinterpret_cast<const long long*>(src), vindex, 8);
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    for (; i < vlen; ++i) scalar_tail(i);
    return;
  }
  for (int i = 0; i < vlen; ++i) scalar_tail(i);
}

template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask) {
  tile_gather<T, IdxT, Masked>(dst, src, idx, mask, VLEN);
}

// ===================================================================
// tile_scatter : dst[idx[i]] = src[i] ; RMW skip-inactive.
// AVX2 has NO scatter instruction -> per-lane scalar loop, with the mask
// guard so an inactive lane (e.g. garbage / OOB index) is never written.
// ===================================================================
template <typename T, typename IdxT, bool Masked>
inline void tile_scatter(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                         const bool* __restrict__ mask, int vlen) {
  for (int i = 0; i < vlen; ++i) {
    if constexpr (Masked) { if (mask[i]) dst[idx[i]] = src[i]; }
    else dst[idx[i]] = src[i];
  }
}

template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_scatter(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                         const bool* __restrict__ mask) {
  tile_scatter<T, IdxT, Masked>(dst, src, idx, mask, VLEN);
}

}  // namespace tileops
}  // namespace dace
