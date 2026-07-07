// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// NVIDIA CUDA (device) backend of the K=1 tile-op intrinsics. Exposes the SAME
// ``dace::tileops::tile_<op>`` signatures as the scalar / avx512 / ... sibling
// headers, but every function is ``__device__`` (the tile ops run inside a GPU
// kernel) and the fp16 elementwise ops use the native ``half2`` (FP16x2) SIMD
// intrinsics from <cuda_fp16.h> -- ``__hadd2`` / ``__hsub2`` / ``__hmul2`` /
// ``__h2div`` / ``__hmin2`` / ``__hmax2`` / ``__hneg2``. The GPU vectorizer only
// targets fp16 (and fp8) -- see the design note in vectorize_cpu_multi_dim.
//
// FP8 (``__nv_fp8_e4m3`` / ``__nv_fp8_e5m2``) has **no native arithmetic**
// (cuda_fp8.h is conversion + data-movement only): the fp8 path converts each
// element to ``float``, computes, and converts back. fp8x4 packing rides on the
// same per-lane loop -- the storage type is 1 byte so a contiguous tile is a
// byte vector; the compiler coalesces the float<->fp8 conversions.
//
// Op-code legend and masked semantics are identical to scalar.h (the reference).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890) || !defined(__CUDA_ARCH__)
#include <cuda_fp8.h>
#endif
#endif

#ifndef DACE_DFI
#if defined(__CUDACC__)
#define DACE_DFI __device__ __forceinline__
#else
#define DACE_DFI inline
#endif
#endif

namespace dace {
namespace tileops {

// ----------------------------- compute-type apply ----------------------------
// A scalar binary apply that is correct for every element type the GPU path
// sees. Types without native arithmetic operators (fp8) are computed through
// ``float``; ``__half`` and the builtin types use their device operators.
template <typename T>
DACE_DFI float _cuda_to_compute(T a) {
  return static_cast<float>(a);
}
template <typename T>
DACE_DFI T _cuda_from_compute(float a) {
  return static_cast<T>(a);
}

template <typename T, char Op>
DACE_DFI T tile_apply(T a, T b) {
  const float af = _cuda_to_compute<T>(a);
  const float bf = _cuda_to_compute<T>(b);
  float r;
  if constexpr (Op == '+')
    r = af + bf;
  else if constexpr (Op == '-')
    r = af - bf;
  else if constexpr (Op == '*')
    r = af * bf;
  else if constexpr (Op == '/')
    r = af / bf;
  else if constexpr (Op == 'm')
    r = fminf(af, bf);
  else if constexpr (Op == 'M')
    r = fmaxf(af, bf);
  else if constexpr (Op == '<')
    r = (af < bf) ? 1.0f : 0.0f;
  else if constexpr (Op == 'l')
    r = (af <= bf) ? 1.0f : 0.0f;
  else if constexpr (Op == '>')
    r = (af > bf) ? 1.0f : 0.0f;
  else if constexpr (Op == 'g')
    r = (af >= bf) ? 1.0f : 0.0f;
  else if constexpr (Op == '=')
    r = (af == bf) ? 1.0f : 0.0f;
  else if constexpr (Op == '!')
    r = (af != bf) ? 1.0f : 0.0f;
  else if constexpr (Op == '&')
    r = (af && bf) ? 1.0f : 0.0f;
  else /* '|' */
    r = (af || bf) ? 1.0f : 0.0f;
  return _cuda_from_compute<T>(r);
}

template <typename T, char Op>
DACE_DFI T tile_unop_apply(T a) {
  const float af = _cuda_to_compute<T>(a);
  float r;
  if constexpr (Op == 'n')
    r = -af;
  else if constexpr (Op == '!')
    r = af ? 0.0f : 1.0f;
  else if constexpr (Op == 'a')
    r = fabsf(af);
  else if constexpr (Op == 'e')
    r = expf(af);
  else if constexpr (Op == 'l')
    r = logf(af);
  else if constexpr (Op == 's')
    r = sqrtf(af);
  else if constexpr (Op == 'S')
    r = sinf(af);
  else if constexpr (Op == 'C')
    r = cosf(af);
  else if constexpr (Op == 'f')
    r = floorf(af);
  else if constexpr (Op == 'c')
    r = ceilf(af);
  else /* 't' */
    r = tanhf(af);
  return _cuda_from_compute<T>(r);
}

#if defined(__CUDACC__)
// half2 (FP16x2) fast path for the arithmetic binops on ``__half`` tiles. The
// comparison / logical ops keep the scalar path (their half2 forms return a
// mask, not a 1.0/0.0 element). ``Op`` is restricted at the call site below.
template <char Op>
DACE_DFI constexpr bool _is_half2_binop() {
  // Arithmetic + min/max + the six comparisons. The half2 comparison intrinsics
  // (``__hlt2`` / ...) set each lane to 1.0 (true) / 0.0 (false), matching the
  // scalar path's ``(af < bf) ? 1.0f : 0.0f`` element semantics exactly (1.0 / 0.0
  // are representable in fp16). Only the logical ``&`` / ``|`` stay scalar (no
  // half2 boolean-combine intrinsic).
  return Op == '+' || Op == '-' || Op == '*' || Op == '/' || Op == 'm' || Op == 'M' || Op == '<' || Op == 'l' ||
         Op == '>' || Op == 'g' || Op == '=' || Op == '!';
}

template <char Op>
DACE_DFI __half2 _half2_apply(__half2 a, __half2 b) {
  if constexpr (Op == '+')
    return __hadd2(a, b);
  else if constexpr (Op == '-')
    return __hsub2(a, b);
  else if constexpr (Op == '*')
    return __hmul2(a, b);
  else if constexpr (Op == '/')
    return __h2div(a, b);
  else if constexpr (Op == 'm')
    return __hmin2(a, b);
  else if constexpr (Op == 'M')
    return __hmax2(a, b);
  else if constexpr (Op == '<')
    return __hlt2(a, b);
  else if constexpr (Op == 'l')
    return __hle2(a, b);
  else if constexpr (Op == '>')
    return __hgt2(a, b);
  else if constexpr (Op == 'g')
    return __hge2(a, b);
  else if constexpr (Op == '=')
    return __heq2(a, b);
  else /* '!' (not-equal) */
    return __hne2(a, b);
}

// half2 fast path for the unary ops with a native FP16x2 intrinsic: negate / abs
// / the transcendentals exp / log / sqrt / sin / cos and floor / ceil. ``!``
// (logical not) and ``t`` (tanh -- CUDA has no ``h2tanh``) keep the scalar path.
template <char Op>
DACE_DFI constexpr bool _is_half2_unop() {
  return Op == 'n' || Op == 'a' || Op == 'e' || Op == 'l' || Op == 's' || Op == 'S' || Op == 'C' || Op == 'f' ||
         Op == 'c';
}

template <char Op>
DACE_DFI __half2 _half2_unop_apply(__half2 a) {
  if constexpr (Op == 'n')
    return __hneg2(a);
  else if constexpr (Op == 'a')
    return __habs2(a);
  else if constexpr (Op == 'e')
    return h2exp(a);
  else if constexpr (Op == 'l')
    return h2log(a);
  else if constexpr (Op == 's')
    return h2sqrt(a);
  else if constexpr (Op == 'S')
    return h2sin(a);
  else if constexpr (Op == 'C')
    return h2cos(a);
  else if constexpr (Op == 'f')
    return h2floor(a);
  else /* 'c' */
    return h2ceil(a);
}

// Reduction combine on two SCALAR halves (``__hadd`` / ``__hmul`` / ``__hmax`` /
// ``__hmin``). Used to fold the two lanes of an accumulated half2 into one half.
template <char Op>
DACE_DFI __half _half_combine(__half a, __half b) {
  if constexpr (Op == '+')
    return __hadd(a, b);
  else if constexpr (Op == '*')
    return __hmul(a, b);
  else if constexpr (Op == 'm')
    return __hmin(a, b);
  else /* 'M' */
    return __hmax(a, b);
}

template <char Op>
DACE_DFI constexpr bool _is_half2_reduce() {
  return Op == '+' || Op == '*' || Op == 'm' || Op == 'M';
}
#endif

// ----------------------------- tile_binop -----------------------------
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
DACE_DFI void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                         const bool* __restrict__ mask) {
#if defined(__CUDACC__)
  // FP16x2 path: __half tile, even width, a half2-capable op, unmasked. Two lanes
  // per half2 intrinsic (arithmetic, min/max, and the six comparisons).
  if constexpr (__is_same(T, __half) && (VLEN % 2 == 0) && !Masked && _is_half2_binop<Op>()) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 2) {
      const __half2 av = BroadcastA ? __half2half2(a[0]) : __halves2half2(a[i], a[i + 1]);
      const __half2 bv = BroadcastB ? __half2half2(b[0]) : __halves2half2(b[i], b[i + 1]);
      const __half2 rv = _half2_apply<Op>(av, bv);
      out[i] = __low2half(rv);
      out[i + 1] = __high2half(rv);
    }
    return;
  }
#endif
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    const T av = BroadcastA ? a[0] : a[i];
    const T bv = BroadcastB ? b[0] : b[i];
    if constexpr (Masked)
      out[i] = mask[i] ? tile_apply<T, Op>(av, bv) : T(0);
    else
      out[i] = tile_apply<T, Op>(av, bv);
  }
}

// ----------------------------- tile_unop ------------------------------
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked>
DACE_DFI void tile_unop(T* __restrict__ out, const T* __restrict__ a, const bool* __restrict__ mask) {
#if defined(__CUDACC__)
  // FP16x2 path for every unop with a native half2 intrinsic (negate / abs /
  // exp / log / sqrt / sin / cos / floor / ceil). Two lanes per intrinsic.
  if constexpr (__is_same(T, __half) && (VLEN % 2 == 0) && !Masked && _is_half2_unop<Op>()) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 2) {
      const __half2 av = Broadcast ? __half2half2(a[0]) : __halves2half2(a[i], a[i + 1]);
      const __half2 rv = _half2_unop_apply<Op>(av);
      out[i] = __low2half(rv);
      out[i + 1] = __high2half(rv);
    }
    return;
  }
#endif
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    const T av = Broadcast ? a[0] : a[i];
    if constexpr (Masked)
      out[i] = mask[i] ? tile_unop_apply<T, Op>(av) : T(0);
    else
      out[i] = tile_unop_apply<T, Op>(av);
  }
}

// ----------------------------- tile_ite -----------------------------
template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked>
DACE_DFI void tile_ite(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                       const T* __restrict__ e, const bool* __restrict__ mask) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    const T tv = BroadcastThen ? t[0] : t[i];
    const T ev = BroadcastElse ? e[0] : e[i];
    const bool c = _cuda_to_compute<CondT>(cond[i]) != 0.0f;
    if constexpr (Masked)
      out[i] = mask[i] ? (c ? tv : ev) : T(0);
    else
      out[i] = c ? tv : ev;
  }
}

// ----------------------------- tile_load ------------------------------
template <typename T, int VLEN, bool Masked>
DACE_DFI void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                        std::int64_t stride = 1) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked)
      dst[i] = mask[i] ? src[i * stride] : T(0);
    else
      dst[i] = src[i * stride];
  }
}

// ----------------------------- tile_store -----------------------------
template <typename T, int VLEN, bool Masked>
DACE_DFI void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                         std::int64_t stride = 1) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) {
      if (mask[i]) dst[i * stride] = src[i];
    } else
      dst[i * stride] = src[i];
  }
}

// ---------------------------- tile_gather -----------------------------
template <typename T, typename IdxT, int VLEN, bool Masked>
DACE_DFI void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                          const bool* __restrict__ mask) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked)
      dst[i] = mask[i] ? src[idx[i]] : T(0);
    else
      dst[i] = src[idx[i]];
  }
}

// ---------------------------- tile_scatter ----------------------------
template <typename T, typename IdxT, int VLEN, bool Masked>
DACE_DFI void tile_scatter(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                           const bool* __restrict__ mask) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) {
      if (mask[i]) dst[idx[i]] = src[i];
    } else
      dst[idx[i]] = src[i];
  }
}

// ---------------------------- tile_mask_gen ----------------------------
template <typename IdxT, int VLEN>
DACE_DFI void tile_mask_gen(bool* __restrict__ out, IdxT base, IdxT ub) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) out[i] = (base + IdxT(i)) < ub;
}

// ----------------------------- tile_reduce ----------------------------
// Horizontal reduction of a VLEN-lane tile to a SINGLE scalar (an in-map / per-tile
// reduction: ``acc = sum/max/min/prod over the tile``). ``Op`` is the reduction op
// ('+' sum, '*' prod, 'm' min, 'M' max). Returns the reduced ELEMENT (a ``__half`` for
// an fp16 tile), not a vector.
//
// CUDA has NO single "reduce half2 -> half" intrinsic, so for an fp16 tile we compose
// one: accumulate the lanes pairwise with the half2 arithmetic intrinsic (``__hadd2`` /
// ... two lanes per op), then fold the surviving half2's two lanes into one ``__half``
// with the scalar combine (``__hadd`` / ...). The generic template keeps the portable
// per-lane scalar fold for every other element type.
template <typename T, int VLEN, char Op>
DACE_DFI T tile_reduce(const T* __restrict__ src) {
#if defined(__CUDACC__)
  if constexpr (__is_same(T, __half) && VLEN >= 2 && (VLEN % 2 == 0) && _is_half2_reduce<Op>()) {
    __half2 acc = __halves2half2(src[0], src[1]);
#pragma unroll
    for (int i = 2; i < VLEN; i += 2) acc = _half2_apply<Op>(acc, __halves2half2(src[i], src[i + 1]));
    return _half_combine<Op>(__low2half(acc), __high2half(acc));
  }
#endif
  T acc = src[0];
#pragma unroll
  for (int i = 1; i < VLEN; ++i) acc = tile_apply<T, Op>(acc, src[i]);
  return acc;
}

// =========================== VLEN=1 overloads ============================
// Mirror scalar.h: DaCe collapses a Register Array(shape=(1,)) transient to a
// plain ``T``, so the VLEN=1 call site can mix ``T`` / ``T*`` / ``T[1]``
// operands. ``tile_load_value`` / ``tile_store_value`` normalise them.
template <typename T>
DACE_DFI T tile_load_value(const T& x) {
  return x;
}
template <typename T>
DACE_DFI T tile_load_value(const T* x) {
  return *x;
}
template <typename T, std::size_t N>
DACE_DFI T tile_load_value(const T (&x)[N]) {
  return x[0];
}

template <typename T, typename V>
DACE_DFI void tile_store_value(T& dst, V v) {
  dst = static_cast<T>(v);
}
template <typename T, typename V>
DACE_DFI void tile_store_value(T* dst, V v) {
  *dst = static_cast<T>(v);
}
template <typename T, std::size_t N, typename V>
DACE_DFI void tile_store_value(T (&dst)[N], V v) {
  dst[0] = static_cast<T>(v);
}

template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked, typename Out, typename A,
          typename B>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_binop(Out&& out, A&& a, B&& b,
                                                                   const bool* __restrict__ mask) {
  const T av = tile_load_value<T>(a);
  const T bv = tile_load_value<T>(b);
  T rv = tile_apply<T, Op>(av, bv);
  if constexpr (Masked)
    tile_store_value<T>(out, mask[0] ? rv : T(0));
  else
    tile_store_value<T>(out, rv);
}

template <typename T, int VLEN, char Op, bool Broadcast, bool Masked, typename Out, typename A>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_unop(Out&& out, A&& a, const bool* __restrict__ mask) {
  const T av = tile_load_value<T>(a);
  T rv = tile_unop_apply<T, Op>(av);
  if constexpr (Masked)
    tile_store_value<T>(out, mask[0] ? rv : T(0));
  else
    tile_store_value<T>(out, rv);
}

template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked, typename Out,
          typename C, typename TThen, typename EElse>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_ite(Out&& out, C&& cond, TThen&& t, EElse&& e,
                                                                 const bool* __restrict__ mask) {
  const bool cv = _cuda_to_compute<CondT>(tile_load_value<CondT>(cond)) != 0.0f;
  const T tv = tile_load_value<T>(t);
  const T ev = tile_load_value<T>(e);
  T rv = cv ? tv : ev;
  if constexpr (Masked)
    tile_store_value<T>(out, mask[0] ? rv : T(0));
  else
    tile_store_value<T>(out, rv);
}

template <typename T, int VLEN, bool Masked, typename Dst, typename Src>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_load(Dst&& dst, Src&& src, const bool* __restrict__ mask,
                                                                  std::int64_t /*stride*/ = 1) {
  const T sv = tile_load_value<T>(src);
  if constexpr (Masked)
    tile_store_value<T>(dst, mask[0] ? sv : T(0));
  else
    tile_store_value<T>(dst, sv);
}

template <typename T, int VLEN, bool Masked, typename Dst, typename Src>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_store(Dst&& dst, Src&& src, const bool* __restrict__ mask,
                                                                   std::int64_t /*stride*/ = 1) {
  const T sv = tile_load_value<T>(src);
  if constexpr (Masked) {
    if (mask[0]) tile_store_value<T>(dst, sv);
  } else
    tile_store_value<T>(dst, sv);
}

template <typename T, typename IdxT, int VLEN, bool Masked, typename Dst, typename Idx>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_gather(Dst&& dst, const T* __restrict__ src, Idx&& idx,
                                                                    const bool* __restrict__ mask) {
  const IdxT iv = tile_load_value<IdxT>(idx);
  if constexpr (Masked)
    tile_store_value<T>(dst, mask[0] ? src[iv] : T(0));
  else
    tile_store_value<T>(dst, src[iv]);
}

template <typename T, typename IdxT, int VLEN, bool Masked, typename Src, typename Idx>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_scatter(T* __restrict__ dst, Src&& src, Idx&& idx,
                                                                     const bool* __restrict__ mask) {
  const T sv = tile_load_value<T>(src);
  const IdxT iv = tile_load_value<IdxT>(idx);
  if constexpr (Masked) {
    if (mask[0]) dst[iv] = sv;
  } else
    dst[iv] = sv;
}

}  // namespace tileops
}  // namespace dace
