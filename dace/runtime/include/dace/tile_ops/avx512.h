// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// AVX-512 backend of the K=1 tile-op intrinsics. Same signatures as the scalar
// reference (dace/tile_ops/scalar.h); the tile-op node's AVX512 expansion pulls
// THIS header in via its environment. NEVER include together with another
// backend header (the env picks exactly one).
//
// Each op is a function template parameterised by element type ``T``, the
// constexpr tile width ``VLEN``, a one-char op code ``Op`` (binop only; see the
// scalar header for the legend), per-operand broadcast booleans, and a
// ``Masked`` boolean -- no enums / structs / functors. ``VLEN`` may exceed the
// 512-bit register width (e.g. 64) or not divide it (e.g. 50); every op runs a
// W-wide SIMD chunk loop (W=16 fp32, W=8 fp64) plus a scalar tail.
//
// Coverage: fp32 / fp64 arithmetic ``tile_binop`` (``+ - * / m M``) lowers to
// ``_mm512`` intrinsics with an ``__mmask`` built from the bool tile (zero-fill
// producer; RMW-safe writer). Comparisons / logical ops / integer types and the
// other ops (merge/load/store/gather/scatter) use the correct per-lane loop
// (matching scalar.h) and are SIMD-ized incrementally. Masked semantics match
// scalar.h exactly: producers ZERO-FILL inactive lanes with guarded reads;
// writers (store/scatter) RMW skip-inactive.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#if !defined(__AVX512F__)
#error Included the AVX-512 tile-op header without AVX-512 support (build with -mavx512f)
#endif

namespace dace {
namespace tileops {

// Per-lane binary op (scalar reference semantics; used for fallback paths and
// the comparison / logical / integer cases).
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
  else /* '|' */ return (a || b) ? T(1) : T(0);
}

// True iff ``Op`` has a direct ``_mm512_<op>_{ps,pd}`` form (the arithmetic ops).
template <char Op>
constexpr bool avx512_is_simd_arith() {
  return Op == '+' || Op == '-' || Op == '*' || Op == '/' || Op == 'm' || Op == 'M';
}

namespace detail {
// One AVX-512 fp lane-group: out = op(av, bv) for the active lanes (k), zero
// elsewhere (ZERO-FILL producer semantics via maskz). float (W=16).
template <char Op>
inline void avx512_arith_ps(float* out, __m512 va, __m512 vb, __mmask16 k) {
  __m512 r;
  if constexpr (Op == '+') r = _mm512_add_ps(va, vb);
  else if constexpr (Op == '-') r = _mm512_sub_ps(va, vb);
  else if constexpr (Op == '*') r = _mm512_mul_ps(va, vb);
  else if constexpr (Op == '/') r = _mm512_div_ps(va, vb);
  else if constexpr (Op == 'm') r = _mm512_min_ps(va, vb);
  else r = _mm512_max_ps(va, vb);
  _mm512_storeu_ps(out, _mm512_maskz_mov_ps(k, r));  // zero-fill inactive
}
template <char Op>
inline void avx512_arith_pd(double* out, __m512d va, __m512d vb, __mmask8 k) {
  __m512d r;
  if constexpr (Op == '+') r = _mm512_add_pd(va, vb);
  else if constexpr (Op == '-') r = _mm512_sub_pd(va, vb);
  else if constexpr (Op == '*') r = _mm512_mul_pd(va, vb);
  else if constexpr (Op == '/') r = _mm512_div_pd(va, vb);
  else if constexpr (Op == 'm') r = _mm512_min_pd(va, vb);
  else r = _mm512_max_pd(va, vb);
  _mm512_storeu_pd(out, _mm512_maskz_mov_pd(k, r));
}
}  // namespace detail

// ----------------------------- tile_binop -----------------------------
// fp32/fp64 arithmetic -> AVX-512 W-chunk loop + scalar tail; else scalar loop.
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask) {
#if defined(__AVX512F__)
  if constexpr (avx512_is_simd_arith<Op>() && std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= VLEN; i += W) {
      __mmask16 k = 0xFFFF;
      if constexpr (Masked) { k = 0; for (int j = 0; j < W; ++j) if (mask[i + j]) k |= __mmask16(1) << j; }
      __m512 va = BroadcastA ? _mm512_set1_ps(a[0]) : _mm512_loadu_ps(a + i);
      __m512 vb = BroadcastB ? _mm512_set1_ps(b[0]) : _mm512_loadu_ps(b + i);
      detail::avx512_arith_ps<Op>(out + i, va, vb, k);
    }
    for (; i < VLEN; ++i) {
      const float av = BroadcastA ? a[0] : a[i];
      const float bv = BroadcastB ? b[0] : b[i];
      if constexpr (Masked) out[i] = mask[i] ? tile_apply<float, Op>(av, bv) : 0.0f;
      else out[i] = tile_apply<float, Op>(av, bv);
    }
    return;
  }
  if constexpr (avx512_is_simd_arith<Op>() && std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= VLEN; i += W) {
      __mmask8 k = 0xFF;
      if constexpr (Masked) { k = 0; for (int j = 0; j < W; ++j) if (mask[i + j]) k |= __mmask8(1) << j; }
      __m512d va = BroadcastA ? _mm512_set1_pd(a[0]) : _mm512_loadu_pd(a + i);
      __m512d vb = BroadcastB ? _mm512_set1_pd(b[0]) : _mm512_loadu_pd(b + i);
      detail::avx512_arith_pd<Op>(out + i, va, vb, k);
    }
    for (; i < VLEN; ++i) {
      const double av = BroadcastA ? a[0] : a[i];
      const double bv = BroadcastB ? b[0] : b[i];
      if constexpr (Masked) out[i] = mask[i] ? tile_apply<double, Op>(av, bv) : 0.0;
      else out[i] = tile_apply<double, Op>(av, bv);
    }
    return;
  }
#endif
  // Scalar path: comparisons / logical / integer types / no-AVX512 build.
  for (int i = 0; i < VLEN; ++i) {
    const T av = BroadcastA ? a[0] : a[i];
    const T bv = BroadcastB ? b[0] : b[i];
    if constexpr (Masked) out[i] = mask[i] ? tile_apply<T, Op>(av, bv) : T(0);
    else out[i] = tile_apply<T, Op>(av, bv);
  }
}

// ----- merge / load / store / gather / scatter -----
// Correct per-lane forms (matching scalar.h); SIMD-ized incrementally. Producers
// ZERO-FILL inactive + guard the read; writers RMW skip-inactive.
// Per-lane unary op (op codes: n neg, ! not, a abs, e exp, l log, s sqrt,
// S sin, C cos, f floor, c ceil, t tanh). The transcendentals have no portable
// SIMD intrinsic, so this shares scalar.h's lane-loop form in every backend.
template <typename T, char Op>
inline T tile_unop_apply(T a) {
  if constexpr (Op == 'n') return -a;
  else if constexpr (Op == '!') return T(!a);
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
inline void tile_ite(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                       const T* __restrict__ e, const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; ++i) {
    const T tv = BroadcastThen ? t[0] : t[i];
    const T ev = BroadcastElse ? e[0] : e[i];
    if constexpr (Masked) out[i] = mask[i] ? (cond[i] ? tv : ev) : T(0);
    else out[i] = cond[i] ? tv : ev;
  }
}

template <typename T, int VLEN, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                      std::int64_t stride = 1) {
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) dst[i] = mask[i] ? src[i * stride] : T(0);
    else dst[i] = src[i * stride];
  }
}

template <typename T, int VLEN, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                       std::int64_t stride = 1) {
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) { if (mask[i]) dst[i * stride] = src[i]; }
    else dst[i * stride] = src[i];
  }
}

template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) dst[i] = mask[i] ? src[idx[i]] : T(0);
    else dst[i] = src[idx[i]];
  }
}

template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_scatter(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                         const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) { if (mask[i]) dst[idx[i]] = src[i]; }
    else dst[idx[i]] = src[i];
  }
}

}  // namespace tileops
}  // namespace dace
