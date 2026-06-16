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
//   ``n`` neg(-a)  ``!`` not(!a)  ``a`` abs  ``e`` exp  ``l`` log  ``s`` sqrt
//   ``S`` sin      ``C`` cos  ``f`` floor ``c`` ceil ``t`` tanh
// Transcendentals have no portable SIMD intrinsic, so every backend shares this
// vectorize-hinted lane loop (the compiler auto-vectorises neg/abs/sqrt and
// calls a vector libm for exp/log where available).
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

// ----------------------------- tile_ite -----------------------------
// out[i] = cond[i] ? t : e ; ZERO-FILL inactive.
template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked>
inline void tile_ite(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
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

// Forward-declare ``tile_load_value`` (defined further down with the
// VLEN==1 polymorphism block) so the by-value ``Src`` overload below can
// reference it. Definitions are visible at instantiation time.
template <typename T> inline T tile_load_value(const T& x) noexcept;
template <typename T> inline T tile_load_value(const T* x) noexcept;
template <typename T, std::size_t N> inline T tile_load_value(const T (&x)[N]) noexcept;

// VLEN>1 ``tile_load`` with a by-value ``src`` (Scalar / Symbol operand
// codegen materialises as ``T _src = expr;``). SFINAE keeps this binding
// off the contiguous ``T* src`` overload above; ``Src&&`` accepts any of
// ``T``, ``T&``, ``T[N]``. ``stride`` is unused for a broadcast but kept
// in the signature for call-site uniformity with the pointer form -- the
// caller emits one ``tile_load<T, VLEN, Masked>(_dst, _src, mask, stride)``
// for every tile load and the runtime picks pointer-strided vs by-value
// broadcast through overload resolution.
template <typename T, int VLEN, bool Masked, typename Src>
inline std::enable_if_t<(VLEN > 1) && !std::is_pointer_v<std::remove_reference_t<Src>>, void>
tile_load(T* __restrict__ dst, Src&& src, const bool* __restrict__ mask, std::int64_t /*stride*/ = 1) {
  const T sv = tile_load_value<T>(src);
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) dst[i] = mask[i] ? sv : T(0);
    else dst[i] = sv;
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

// ---------------------------- tile_mask_gen ----------------------------
// Iteration mask: out[l] = (base + l) < ub for l in 0..VLEN-1. The K=1 form of
// the per-dim conjunction TileMaskGen lowers (K>=2 stays 'pure'); ``base`` is the
// surrounding map iter-var and ``ub`` the dim's exclusive upper bound. The
// predicate is a monotonic whilelt-style prefix; consumers rebuild their own
// hardware mask from this bool tile.
template <typename IdxT, int VLEN>
inline void tile_mask_gen(bool* __restrict__ out, IdxT base, IdxT ub) {
  for (int i = 0; i < VLEN; ++i) out[i] = (base + IdxT(i)) < ub;
}

// =========================== VLEN=1 overloads ============================
// DaCe codegen collapses a ``Register Array(shape=(1,))`` transient to a
// plain ``T`` variable for the K=0 / W=1 postamble — but other operands
// at the SAME call site may stay as ``T[1]`` arrays (which decay to
// ``T*``). The reference-only overloads above don't bind to that mix.
// Each VLEN=1 wrapper below uses ``tile_addr`` to accept any combination
// of ``T``, ``T&``, ``T*``, or ``T[N]`` arguments and normalise them to
// ``T*`` before forwarding to the canonical pointer-shape body. Mask
// always stays ``const bool*`` (DaCe routes the mask through a tile
// transient even at VLEN=1).

// ``tile_load_value`` extracts the single element from any kind of
// VLEN=1 tile operand:
//   * ``T``       — scalar value: return as-is.
//   * ``T&``      — reference: return the referenced value.
//   * ``T*``      — pointer to a 1-element buffer: return ``*p``.
//   * ``T[N]``    — array of N (always 1 here): return ``arr[0]``.
// Const-correctness is handled by the by-value return type. The output
// (``dst``) side uses ``tile_store_value`` which writes back through
// either a scalar reference or a pointer.

template <typename T>
inline T tile_load_value(const T& x) noexcept { return x; }
template <typename T>
inline T tile_load_value(const T* x) noexcept { return *x; }
template <typename T, std::size_t N>
inline T tile_load_value(const T (&x)[N]) noexcept { return x[0]; }

template <typename T, typename V>
inline void tile_store_value(T& dst, V v) noexcept { dst = static_cast<T>(v); }
template <typename T, typename V>
inline void tile_store_value(T* dst, V v) noexcept { *dst = static_cast<T>(v); }
template <typename T, std::size_t N, typename V>
inline void tile_store_value(T (&dst)[N], V v) noexcept { dst[0] = static_cast<T>(v); }

// VLEN=1 tile_binop.
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked,
          typename Out, typename A, typename B>
inline std::enable_if_t<VLEN == 1, void>
tile_binop(Out&& out, A&& a, B&& b, const bool* __restrict__ mask) {
  const T av = tile_load_value<T>(a);
  const T bv = tile_load_value<T>(b);
  T rv = tile_apply<T, Op>(av, bv);
  if constexpr (Masked) tile_store_value<T>(out, mask[0] ? rv : T(0));
  else tile_store_value<T>(out, rv);
}

// VLEN=1 tile_unop.
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked,
          typename Out, typename A>
inline std::enable_if_t<VLEN == 1, void>
tile_unop(Out&& out, A&& a, const bool* __restrict__ mask) {
  const T av = tile_load_value<T>(a);
  T rv = tile_unop_apply<T, Op>(av);
  if constexpr (Masked) tile_store_value<T>(out, mask[0] ? rv : T(0));
  else tile_store_value<T>(out, rv);
}

// VLEN=1 tile_ite.
template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked,
          typename Out, typename C, typename TThen, typename EElse>
inline std::enable_if_t<VLEN == 1, void>
tile_ite(Out&& out, C&& cond, TThen&& t, EElse&& e, const bool* __restrict__ mask) {
  const CondT cv = tile_load_value<CondT>(cond);
  const T tv = tile_load_value<T>(t);
  const T ev = tile_load_value<T>(e);
  T rv = cv ? tv : ev;
  if constexpr (Masked) tile_store_value<T>(out, mask[0] ? rv : T(0));
  else tile_store_value<T>(out, rv);
}

// VLEN=1 tile_load. ``stride`` is irrelevant (one lane).
template <typename T, int VLEN, bool Masked, typename Dst, typename Src>
inline std::enable_if_t<VLEN == 1, void>
tile_load(Dst&& dst, Src&& src, const bool* __restrict__ mask, std::int64_t /*stride*/ = 1) {
  const T sv = tile_load_value<T>(src);
  if constexpr (Masked) tile_store_value<T>(dst, mask[0] ? sv : T(0));
  else tile_store_value<T>(dst, sv);
}

// VLEN=1 tile_store.
template <typename T, int VLEN, bool Masked, typename Dst, typename Src>
inline std::enable_if_t<VLEN == 1, void>
tile_store(Dst&& dst, Src&& src, const bool* __restrict__ mask, std::int64_t /*stride*/ = 1) {
  const T sv = tile_load_value<T>(src);
  if constexpr (Masked) { if (mask[0]) tile_store_value<T>(dst, sv); }
  else tile_store_value<T>(dst, sv);
}

// VLEN=1 tile_gather: ``src`` array stays a pointer (the gather indexes
// into the outer buffer); ``dst`` and the index value may be scalars.
template <typename T, typename IdxT, int VLEN, bool Masked, typename Dst, typename Idx>
inline std::enable_if_t<VLEN == 1, void>
tile_gather(Dst&& dst, const T* __restrict__ src, Idx&& idx, const bool* __restrict__ mask) {
  const IdxT iv = tile_load_value<IdxT>(idx);
  if constexpr (Masked) tile_store_value<T>(dst, mask[0] ? src[iv] : T(0));
  else tile_store_value<T>(dst, src[iv]);
}

// VLEN=1 tile_scatter: ``dst`` stays a pointer; ``src``/``idx`` may be
// scalars.
template <typename T, typename IdxT, int VLEN, bool Masked, typename Src, typename Idx>
inline std::enable_if_t<VLEN == 1, void>
tile_scatter(T* __restrict__ dst, Src&& src, Idx&& idx, const bool* __restrict__ mask) {
  const T sv = tile_load_value<T>(src);
  const IdxT iv = tile_load_value<IdxT>(idx);
  if constexpr (Masked) { if (mask[0]) dst[iv] = sv; }
  else dst[iv] = sv;
}

}  // namespace tileops
}  // namespace dace
