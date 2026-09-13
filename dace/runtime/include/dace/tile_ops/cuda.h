// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// CUDA/HIP device backend of the K=1 tile-op intrinsics; same dace::tileops::tile_<op>
// signatures as scalar.h, but every function is __device__. fp16 uses native half2 SIMD
// intrinsics; fp8 has no native arithmetic, so it round-trips through float.
#pragma once

// Guarded on the device compiler, not the vendor: __CUDACC__ alone would make every tile op
// vanish (fall back to scalar) under HIP.

#include <cmath>
#include <cstdint>
#include <type_traits>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#elif defined(__CUDACC__)
#include <cuda_fp16.h>
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890) || !defined(__CUDA_ARCH__)
#include <cuda_fp8.h>
#endif
#endif

#ifndef DACE_DFI
#if defined(__CUDACC__) || defined(__HIPCC__)
#define DACE_DFI __device__ __forceinline__
#else
#define DACE_DFI inline
#endif
#endif

namespace dace {
namespace tileops {

// compute-type apply: the type a scalar tile op computes in. Anything with native device
// arithmetic computes in itself; only fp8 and __half (no arithmetic operators of their own)
// convert through float. Routing everything through float would silently narrow double and
// drop 64-bit integers past 2^24.
template <typename T>
using tile_compute_t = std::conditional_t<std::is_arithmetic_v<T>, T, float>;

// The type the transcendental unops evaluate in: an integral tile has no ``exp`` of its own, and
// only ``double`` earns the double-precision libm entry points.
template <typename T>
using tile_math_t = std::conditional_t<std::is_same_v<tile_compute_t<T>, double>, double, float>;

// ``std::min`` / ``std::max`` are host-only, and ``fminf`` picks the non-NaN operand where
// scalar.h's ``std::min`` propagates it. These match scalar.h, which is the reference the CPU
// tiles are checked against.
template <typename C>
DACE_DFI C _tile_min(C a, C b) {
  return (b < a) ? b : a;
}

template <typename C>
DACE_DFI C _tile_max(C a, C b) {
  return (a < b) ? b : a;
}

template <typename C>
DACE_DFI C _tile_abs(C a) {
  if constexpr (std::is_unsigned_v<C>)
    return a;
  else
    return (a < C(0)) ? static_cast<C>(-a) : a;
}

template <typename C>
DACE_DFI C _tile_fma(C a, C b, C c) {
  if constexpr (std::is_same_v<C, double>)
    return fma(a, b, c);
  else if constexpr (std::is_same_v<C, float>)
    return fmaf(a, b, c);
  else
    return static_cast<C>(a * b + c);
}

template <typename T>
DACE_DFI bool _tile_truthy(T a) {
  return static_cast<tile_compute_t<T>>(a) != tile_compute_t<T>(0);
}

template <typename T, char Op>
DACE_DFI T tile_apply(T a, T b) {
  using C = tile_compute_t<T>;
  const C af = static_cast<C>(a);
  const C bf = static_cast<C>(b);
  C r;
  if constexpr (Op == '+')
    r = af + bf;
  else if constexpr (Op == '-')
    r = af - bf;
  else if constexpr (Op == '*')
    r = af * bf;
  else if constexpr (Op == '/')
    r = af / bf;
  else if constexpr (Op == 'm')
    r = _tile_min(af, bf);
  else if constexpr (Op == 'M')
    r = _tile_max(af, bf);
  else if constexpr (Op == '<')
    r = (af < bf) ? C(1) : C(0);
  else if constexpr (Op == 'l')
    r = (af <= bf) ? C(1) : C(0);
  else if constexpr (Op == '>')
    r = (af > bf) ? C(1) : C(0);
  else if constexpr (Op == 'g')
    r = (af >= bf) ? C(1) : C(0);
  else if constexpr (Op == '=')
    r = (af == bf) ? C(1) : C(0);
  else if constexpr (Op == '!')
    r = (af != bf) ? C(1) : C(0);
  else if constexpr (Op == '&')
    r = (af && bf) ? C(1) : C(0);
  else /* '|' */
    r = (af || bf) ? C(1) : C(0);
  return static_cast<T>(r);
}

template <typename T, char Op>
DACE_DFI T tile_unop_apply(T a) {
  using C = tile_compute_t<T>;
  using M = tile_math_t<T>;
  if constexpr (Op == 'n') {
    return static_cast<T>(static_cast<C>(-static_cast<C>(a)));
  } else if constexpr (Op == '!') {
    return static_cast<T>(_tile_truthy<T>(a) ? C(0) : C(1));
  } else if constexpr (Op == 'a') {
    return static_cast<T>(_tile_abs<C>(static_cast<C>(a)));
  } else {
    const M af = static_cast<M>(a);
    M r;
    if constexpr (Op == 'e')
      r = exp(af);
    else if constexpr (Op == 'l')
      r = log(af);
    else if constexpr (Op == 's')
      r = sqrt(af);
    else if constexpr (Op == 'S')
      r = sin(af);
    else if constexpr (Op == 'C')
      r = cos(af);
    else if constexpr (Op == 'f')
      r = floor(af);
    else if constexpr (Op == 'c')
      r = ceil(af);
    else /* 't' */
      r = tanh(af);
    return static_cast<T>(r);
  }
}

#if defined(__CUDACC__) || defined(__HIPCC__)
// half2 (FP16x2) fast path for the arithmetic binops on ``__half`` tiles. The
// comparison / logical ops keep the scalar path (their half2 forms return a
// mask, not a 1.0/0.0 element). ``Op`` is restricted at the call site below.
template <char Op>
DACE_DFI constexpr bool _is_half2_binop() {
  // Arithmetic, min/max and the six comparisons; & / | stay scalar (no half2 boolean-combine).
  return Op == '+' || Op == '-' || Op == '*' || Op == '/' || Op == 'm' || Op == 'M' || Op == '<' || Op == 'l' ||
         Op == '>' || Op == 'g' || Op == '=' || Op == '!';
}

// ROCm 7.2.3 ships the scalar ``__hmin`` / ``__hmax`` but no FP16x2 pair, so the two lanes fold
// through the scalar ones there; CUDA keeps its single native instruction.
DACE_DFI __half2 _half2_min(__half2 a, __half2 b) {
#if defined(__HIPCC__)
  return __halves2half2(__hmin(__low2half(a), __low2half(b)), __hmin(__high2half(a), __high2half(b)));
#else
  return __hmin2(a, b);
#endif
}

DACE_DFI __half2 _half2_max(__half2 a, __half2 b) {
#if defined(__HIPCC__)
  return __halves2half2(__hmax(__low2half(a), __low2half(b)), __hmax(__high2half(a), __high2half(b)));
#else
  return __hmax2(a, b);
#endif
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
    return _half2_min(a, b);
  else if constexpr (Op == 'M')
    return _half2_max(a, b);
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

// Aligned 32-bit pair load/store for the FP16x2 fast paths. Tile buffers are DACE_ALIGN(64)
// and every non-broadcast half2 access starts at an even lane, so &p[i] is 4-byte aligned.
DACE_DFI __half2 _load_half2(const __half* __restrict__ p) { return *reinterpret_cast<const __half2*>(p); }
DACE_DFI void _store_half2(__half* __restrict__ p, __half2 v) { *reinterpret_cast<__half2*>(p) = v; }

// One chunk of a contiguous __half copy: 16 B, 8 B or 4 B (half2). Both pointers must
// already be BYTES-aligned -- the ladder below is what guarantees that.
template <int BYTES>
DACE_DFI void _half_copy_chunk(__half* __restrict__ dst, const __half* __restrict__ src) {
  if constexpr (BYTES == 16)
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
  else if constexpr (BYTES == 8)
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
  else /* 4 */
    _store_half2(dst, _load_half2(src));
}

// Contiguous __half block copy between a tile and an array, widened to the largest chunk both
// ends support (16 B, 8 B, 4 B half2, else per element). ALIGN is the byte alignment of the
// array-side pointer, proven by the caller from the memlet; the tile side is always at least
// as aligned as the chunk. The ladder descends one rung at a time (all if constexpr, no
// runtime test) rather than picking one width for the whole copy, so a width the chunk does
// not evenly divide demotes only the tail, not the whole copy. Caller must have already
// excluded a mask and a non-unit stride: this copies VLEN contiguous elements unconditionally.
template <int VLEN, int ALIGN>
DACE_DFI void half_copy_aligned(__half* __restrict__ dst, const __half* __restrict__ src) {
  // Widest chunk BOTH the alignment promise and the remaining length allow; 0 = per element.
  constexpr int C = (ALIGN >= 16 && VLEN >= 8) ? 16 : (ALIGN >= 8 && VLEN >= 4) ? 8 : (ALIGN >= 4 && VLEN >= 2) ? 4 : 0;
  if constexpr (C == 0) {
#pragma unroll
    for (int i = 0; i < VLEN; ++i) dst[i] = src[i];
  } else {
    constexpr int E = C / 2;  // elements per chunk
#pragma unroll
    for (int i = 0; i + E <= VLEN; i += E) _half_copy_chunk<C>(dst + i, src + i);
    constexpr int done = (VLEN / E) * E;
    if constexpr (done < VLEN) half_copy_aligned<VLEN - done, C>(dst + done, src + done);
  }
}

// Contiguous __half block read whose first element sits SHIFT elements above an aligned 32-bit
// word (a +-1 stencil neighbour). A 32-bit load must start on a 4-byte boundary, so the aligned
// window below the access is read instead and cut out with PRMT. Kept per-word (not widened via
// half_copy_aligned) on purpose: neighbouring stencil accesses round down to the same words and
// CSE away, which a widened per-access window would break. Read-only: the window covers elements
// the caller does not own, which a store would clobber; the caller has proven it stays in bounds.
template <int VLEN, int SHIFT>
DACE_DFI void half_read_shifted(__half* __restrict__ dst, const __half* __restrict__ src) {
  static_assert(VLEN % 2 == 0 && SHIFT > 0 && SHIFT * sizeof(__half) < 4, "shifted half read needs 0 < SHIFT < 2");
  // Byte selector picking bytes [SHIFT*2, SHIFT*2+4) out of the {hi,lo} 8-byte pair.
  constexpr unsigned sel = 0x3210u + 0x1111u * (SHIFT * sizeof(__half));
  const unsigned* base = reinterpret_cast<const unsigned*>(src - SHIFT);
#pragma unroll
  for (int i = 0; i < VLEN; i += 2)
    *reinterpret_cast<unsigned*>(dst + i) = __byte_perm(base[i / 2], base[i / 2 + 1], sel);
}
#endif

// tile_binop
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
DACE_DFI void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                         const bool* __restrict__ mask) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // FP16x2 path: __half tile, even width, a half2-capable op, unmasked. Two lanes
  // per half2 intrinsic (arithmetic, min/max, and the six comparisons).
  // ``else`` rather than an early ``return``: the scalar loop has to be DISCARDED, not merely
  // left unreachable, or nvcc raises #128-D (loop is not reachable) and -Werror fails the build.
  if constexpr (__is_same(T, __half) && (VLEN % 2 == 0) && !Masked && _is_half2_binop<Op>()) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 2) {
      const __half2 av = BroadcastA ? __half2half2(a[0]) : _load_half2(&a[i]);
      const __half2 bv = BroadcastB ? __half2half2(b[0]) : _load_half2(&b[i]);
      const __half2 rv = _half2_apply<Op>(av, bv);
      _store_half2(&out[i], rv);
    }
  } else
#endif
  {
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
}

// tile_fma: out[i] = fma(a, b, c) = a*b + c, single rounding. A __half tile at even width
// uses the native FP16x2 __hfma2; everything else computes through tile_compute_t.
template <typename T, int VLEN, bool BroadcastA, bool BroadcastB, bool BroadcastC, bool Masked>
DACE_DFI void tile_fma(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c,
                       const bool* __restrict__ mask) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // FP16x2 path: __half tile, even width, unmasked. Two lanes per __hfma2.
  // ``else`` and not an early ``return`` -- see the note in tile_binop (nvcc #128-D).
  if constexpr (__is_same(T, __half) && (VLEN % 2 == 0) && !Masked) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 2) {
      const __half2 av = BroadcastA ? __half2half2(a[0]) : _load_half2(&a[i]);
      const __half2 bv = BroadcastB ? __half2half2(b[0]) : _load_half2(&b[i]);
      const __half2 cv = BroadcastC ? __half2half2(c[0]) : _load_half2(&c[i]);
      const __half2 rv = __hfma2(av, bv, cv);  // av*bv + cv
      _store_half2(&out[i], rv);
    }
  } else
#endif
  {
#pragma unroll
    for (int i = 0; i < VLEN; ++i) {
      using C = tile_compute_t<T>;
      const C af = static_cast<C>(BroadcastA ? a[0] : a[i]);
      const C bf = static_cast<C>(BroadcastB ? b[0] : b[i]);
      const C cf = static_cast<C>(BroadcastC ? c[0] : c[i]);
      const T rv = static_cast<T>(_tile_fma<C>(af, bf, cf));  // single-rounded a*b + c
      if constexpr (Masked)
        out[i] = mask[i] ? rv : T(0);
      else
        out[i] = rv;
    }
  }
}

// tile_unop
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked>
DACE_DFI void tile_unop(T* __restrict__ out, const T* __restrict__ a, const bool* __restrict__ mask) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // FP16x2 path for every unop with a native half2 intrinsic (negate / abs /
  // exp / log / sqrt / sin / cos / floor / ceil). Two lanes per intrinsic.
  // ``else`` and not an early ``return`` -- see the note in tile_binop (nvcc #128-D).
  if constexpr (__is_same(T, __half) && (VLEN % 2 == 0) && !Masked && _is_half2_unop<Op>()) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 2) {
      const __half2 av = Broadcast ? __half2half2(a[0]) : _load_half2(&a[i]);
      const __half2 rv = _half2_unop_apply<Op>(av);
      _store_half2(&out[i], rv);
    }
  } else
#endif
  {
#pragma unroll
    for (int i = 0; i < VLEN; ++i) {
      const T av = Broadcast ? a[0] : a[i];
      if constexpr (Masked)
        out[i] = mask[i] ? tile_unop_apply<T, Op>(av) : T(0);
      else
        out[i] = tile_unop_apply<T, Op>(av);
    }
  }
}

// tile_ite
template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked>
DACE_DFI void tile_ite(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                       const T* __restrict__ e, const bool* __restrict__ mask) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    const T tv = BroadcastThen ? t[0] : t[i];
    const T ev = BroadcastElse ? e[0] : e[i];
    const bool c = _tile_truthy<CondT>(cond[i]);
    if constexpr (Masked)
      out[i] = mask[i] ? (c ? tv : ev) : T(0);
    else
      out[i] = c ? tv : ev;
  }
}

// tile_load: Align is the byte alignment of the array side the caller can prove (defaults to
// the element alignment, i.e. the per-element loop). Shift is how many elements src sits above
// that aligned address; nonzero routes to half_read_shifted.
template <typename T, int VLEN, bool Masked, int Align = alignof(T), int Shift = 0>
DACE_DFI void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                        std::int64_t stride = 1) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // A mask makes the lanes divergent and a non-unit stride is not contiguous; both keep the loop.
  if constexpr (__is_same(T, __half) && !Masked && Align >= 4 && VLEN % 2 == 0) {
    if (stride == 1) {
      if constexpr (Shift == 0)
        half_copy_aligned<VLEN, Align>(dst, src);
      else
        half_read_shifted<VLEN, Shift>(dst, src);
      return;
    }
  }
#endif
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked)
      dst[i] = mask[i] ? src[i * stride] : T(0);
    else
      dst[i] = src[i * stride];
  }
}

// tile_store
// Mirror of ``tile_load``: here the ARRAY side ``Align`` describes is ``dst``.
template <typename T, int VLEN, bool Masked, int Align = alignof(T)>
DACE_DFI void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                         std::int64_t stride = 1) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // A masked store must skip its off lanes one at a time, and a non-unit stride is not contiguous.
  if constexpr (__is_same(T, __half) && !Masked && Align >= 4 && VLEN % 2 == 0) {
    if (stride == 1) {
      half_copy_aligned<VLEN, Align>(dst, src);
      return;
    }
  }
#endif
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) {
      if (mask[i]) dst[i * stride] = src[i];
    } else
      dst[i * stride] = src[i];
  }
}

// tile_gather
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

// tile_scatter
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

// tile_mask_gen
// ``stride`` mirrors tile_load's: 1 per thread, the lane count under CUDA_WARP.
template <typename IdxT, int VLEN>
DACE_DFI void tile_mask_gen(bool* __restrict__ out, IdxT base, IdxT ub, IdxT stride = 1) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) out[i] = (base + IdxT(i) * stride) < ub;
}

// tile_reduce: horizontal reduction of a VLEN-lane tile to one scalar (Op: '+' sum, '*' prod,
// 'm' min, 'M' max). An fp16 tile folds as a balanced tree of half2 ops (CUDA has no single
// "reduce half2 -> half" intrinsic), then combines the surviving pair with a scalar combine;
// every other element type folds through a plain per-lane scalar accumulate.
template <typename T, int VLEN, char Op>
DACE_DFI T tile_reduce(const T* __restrict__ src) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  if constexpr (__is_same(T, __half) && VLEN >= 2 && (VLEN % 2 == 0) && _is_half2_reduce<Op>()) {
    constexpr int H = VLEN / 2;
    __half2 buf[H];
#pragma unroll
    for (int i = 0; i < H; ++i) buf[i] = _load_half2(&src[2 * i]);
    int n = H;
    while (n > 1) {
      int half = n / 2;
#pragma unroll
      for (int i = 0; i < half; ++i) buf[i] = _half2_apply<Op>(buf[2 * i], buf[2 * i + 1]);
      if (n & 1) buf[half] = buf[n - 1];
      n = half + (n & 1);
    }
    return _half_combine<Op>(__low2half(buf[0]), __high2half(buf[0]));
  }
#endif
  // Every other element type (fp32 / fp64) folds through the plain per-lane scalar
  // accumulate (unrolled over the compile-time-constant VLEN).
  T acc = src[0];
#pragma unroll
  for (int i = 1; i < VLEN; ++i) acc = tile_apply<T, Op>(acc, src[i]);
  return acc;
}

// VLEN=1 overloads
// Mirror scalar.h: DaCe collapses a Register Array(shape=(1,)) transient to a
// plain ``T``, so the VLEN=1 call site can mix ``T`` / ``T*`` / ``T[1]``
// operands. ``tile_load_value`` / ``tile_store_value`` normalise them.
template <typename T>
DACE_DFI T tile_load_value(const T& x) {
  return x;
}
template <typename T>
DACE_DFI T tile_load_value(const T* __restrict__ x) {
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
DACE_DFI void tile_store_value(T* __restrict__ dst, V v) {
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

template <typename T, int VLEN, bool BroadcastA, bool BroadcastB, bool BroadcastC, bool Masked, typename Out,
          typename A, typename B, typename C>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_fma(Out&& out, A&& a, B&& b, C&& c,
                                                                 const bool* __restrict__ mask) {
  using CT = tile_compute_t<T>;
  const CT af = static_cast<CT>(tile_load_value<T>(a));
  const CT bf = static_cast<CT>(tile_load_value<T>(b));
  const CT cf = static_cast<CT>(tile_load_value<T>(c));
  T rv = static_cast<T>(_tile_fma<CT>(af, bf, cf));
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
  const bool cv = _tile_truthy<CondT>(tile_load_value<CondT>(cond));
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
