// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Scalar (portable) backend of the K=1 tile-op intrinsics. This is the
// reference implementation + the always-available fallback; the avx512 / avx2 /
// arm_neon / arm_sve sibling headers expose the SAME function signatures with
// per-ISA intrinsics. The tile-op library node's chosen-backend expansion pulls
// in exactly one of these headers via its DaCe environment (there is no joint
// dispatch header).
//
// Every op is a function template parameterised by element type ``T``, the
// compile-time tile width ``VLEN`` (always a constexpr -- the K=1 emitter knows
// it at expansion time), a one-char op code ``Op`` (binop only), per-operand
// broadcast booleans (``Broadcast`` splats lane 0 -- the scalar / symbol operand
// case; otherwise per-lane ``a[i]``), and a ``Masked`` boolean. No enums / tag
// structs / functors -- just template parameters on free functions.
//
// ``VLEN`` may exceed the hardware vector width (e.g. 64) or not divide it
// (e.g. 50); each op is written as a lane loop so the SIMD backends can chunk it
// into vector-width steps plus a scalar tail. Here in the scalar reference the
// loop is just a (vectorizer-hinted) per-lane loop.
//
// Op codes (single char, binop only):
//   ``+`` add  ``-`` sub  ``*`` mul  ``/`` div  ``m`` min  ``M`` max
//   ``<`` lt   ``l`` le    ``>`` gt   ``g`` ge   ``=`` eq   ``!`` ne
//   ``&`` and  ``|`` or
// Comparisons / logicals yield ``T(1)`` / ``T(0)`` (the tile stores a condition
// as the element type, matching the legacy vector_<op>).
//
// Masked semantics (load-bearing, two flavours):
//   * tile PRODUCERS (binop / merge / load / gather) -- ZERO-FILL inactive
//     lanes and GUARD the read, so an inactive (e.g. out-of-bounds tail) lane
//     never dereferences ``src`` / ``src[idx]``.
//   * array WRITERS (store / scatter) -- RMW skip-inactive: an inactive lane is
//     not written, so the destination array (and its OOB tail) is never touched.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

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

// ----------------------------- tile_binop -----------------------------
// out[i] = a-operand <op> b-operand ; ZERO-FILL inactive (operand reads are
// in-tile, always safe to evaluate).
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    const T av = BroadcastA ? a[0] : a[i];
    const T bv = BroadcastB ? b[0] : b[i];
    if constexpr (Masked) out[i] = mask[i] ? tile_apply<T, Op>(av, bv) : T(0);
    else out[i] = tile_apply<T, Op>(av, bv);
  }
}

// Per-lane unary op. Op codes (single char):
//   ``n`` neg(-a)  ``a`` abs  ``e`` exp  ``l`` log  ``s`` sqrt
//   ``S`` sin      ``C`` cos  ``f`` floor ``c`` ceil ``t`` tanh
// Transcendentals have no portable SIMD intrinsic, so every backend shares this
// vectorize-hinted lane loop (the compiler auto-vectorises neg/abs/sqrt and
// calls a vector libm for exp/log where available).
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

// ----------------------------- tile_unop ------------------------------
// out[i] = <op> a-operand ; ZERO-FILL inactive (operand read is in-tile, safe
// to evaluate even on an inactive lane).
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked>
inline void tile_unop(T* __restrict__ out, const T* __restrict__ a, const bool* __restrict__ mask) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    const T av = Broadcast ? a[0] : a[i];
    if constexpr (Masked) out[i] = mask[i] ? tile_unop_apply<T, Op>(av) : T(0);
    else out[i] = tile_unop_apply<T, Op>(av);
  }
}

// ----------------------------- tile_merge -----------------------------
// out[i] = cond[i] ? t : e ; ZERO-FILL inactive.
template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked>
inline void tile_merge(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                       const T* __restrict__ e, const bool* __restrict__ mask) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    const T tv = BroadcastThen ? t[0] : t[i];
    const T ev = BroadcastElse ? e[0] : e[i];
    if constexpr (Masked) out[i] = mask[i] ? (cond[i] ? tv : ev) : T(0);
    else out[i] = cond[i] ? tv : ev;
  }
}

// ----------------------------- tile_load ------------------------------
// dst[i] = src[i * stride] ; ZERO-FILL inactive + GUARDED read (inactive lane
// never dereferences src, so an OOB tail lane is safe).
template <typename T, int VLEN, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                      std::int64_t stride = 1) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) dst[i] = mask[i] ? src[i * stride] : T(0);
    else dst[i] = src[i * stride];
  }
}

// ----------------------------- tile_store -----------------------------
// dst[i * stride] = src[i] ; RMW skip-inactive (inactive lane not written, so
// the destination array / OOB tail is never touched).
template <typename T, int VLEN, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                       std::int64_t stride = 1) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) { if (mask[i]) dst[i * stride] = src[i]; }
    else dst[i * stride] = src[i];
  }
}

// ---------------------------- tile_gather -----------------------------
// dst[i] = src[idx[i]] ; ZERO-FILL inactive + GUARDED read (inactive lane never
// dereferences src[idx], so a garbage / OOB index is safe).
template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) dst[i] = mask[i] ? src[idx[i]] : T(0);
    else dst[i] = src[idx[i]];
  }
}

// ---------------------------- tile_scatter ----------------------------
// dst[idx[i]] = src[i] ; RMW skip-inactive (inactive lane never written, so a
// garbage / OOB index is safe).
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
