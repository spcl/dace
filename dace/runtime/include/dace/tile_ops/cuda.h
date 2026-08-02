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

// Aligned 32-bit pair load/store for the FP16x2 fast paths. The tile buffers the
// vectorizer stages are DACE_ALIGN(64) and every non-broadcast half2 access starts
// at an even lane, so ``&p[i]`` is 4-byte aligned. A single LD.U32 / ST.U32 then
// replaces the two-element ``__halves2half2`` pack (two 16-bit loads + a pack) and
// the ``__low2half`` / ``__high2half`` unpack (two extracts + two 16-bit stores).
DACE_DFI __half2 _load_half2(const __half* __restrict__ p) { return *reinterpret_cast<const __half2*>(p); }
DACE_DFI void _store_half2(__half* __restrict__ p, __half2 v) { *reinterpret_cast<__half2*>(p) = v; }

// Contiguous ``__half`` block copy between a tile and an array, widened to the largest chunk
// both ends are known to support: 16 B (LDG/STG.E.128), 8 B (.64), 4 B (half2), else per element.
//
// ALIGN is the byte alignment of the ARRAY-side pointer, which only the CALLER can know -- the
// codegen passes what it can prove from the memlet (dace/libraries/tileops/_isa_codegen.py). The
// tile side needs no test: the vectorizer stages every tile as DACE_ALIGN(64), and a chunk starts
// at element i*C of it, so its address is 64 B aligned at C <= 8.
//
// The choice is entirely ``if constexpr``: exactly ONE chunk width is emitted, with no runtime
// address test and no scalar fallback left behind. Testing alignment at runtime instead would
// keep every branch alive in the SASS -- measured at 3.5x the instruction count of the scalar
// loop, with the 16-bit loads still there -- which is the opposite of the point.
//
// Callers must have already excluded a mask and a non-unit stride: this copies VLEN CONTIGUOUS
// elements unconditionally.
template <int VLEN, int ALIGN>
DACE_DFI void half_copy_aligned(__half* __restrict__ dst, const __half* __restrict__ src) {
  if constexpr (ALIGN >= 16 && VLEN % 8 == 0) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 8) *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
  } else if constexpr (ALIGN >= 8 && VLEN % 4 == 0) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 4) *reinterpret_cast<uint2*>(dst + i) = *reinterpret_cast<const uint2*>(src + i);
  } else if constexpr (ALIGN >= 4 && VLEN % 2 == 0) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 2) _store_half2(&dst[i], _load_half2(&src[i]));
  } else {
#pragma unroll
    for (int i = 0; i < VLEN; ++i) dst[i] = src[i];
  }
}

// Contiguous ``__half`` block read whose FIRST element sits SHIFT elements above an aligned
// 32-bit word -- a +-1 stencil neighbour is the whole motivation. A 32-bit load must start on a
// 4-byte boundary (an unaligned LDG.E is invalid on NVIDIA, not merely slow), so the aligned
// window BELOW the access is read instead and the wanted elements are cut out of the register
// pair with PRMT. Two aligned words cover any SHIFT < 4/sizeof(__half) plus two elements.
//
// PRMT here is not the cost -- the per-element LDG.E.U16 swarm it replaces is. Neighbouring
// accesses in the same row round DOWN to the same words, so the loads CSE away and a 3-point
// stencil ends up reading each word once.
//
// Read-only: the window covers elements the caller does not own, which a store would clobber.
// The caller has proven the window stays inside the allocation; below element 0 is impossible
// because the base is ``src - SHIFT`` with the access offset >= SHIFT by construction.
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
      const __half2 av = BroadcastA ? __half2half2(a[0]) : _load_half2(&a[i]);
      const __half2 bv = BroadcastB ? __half2half2(b[0]) : _load_half2(&b[i]);
      const __half2 rv = _half2_apply<Op>(av, bv);
      _store_half2(&out[i], rv);
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

// ----------------------------- tile_fma -------------------------------
// out[i] = fma(a, b, c) = a*b + c (single rounding). A ``__half`` tile at even
// width uses the native FP16x2 fused multiply-add ``__hfma2(av, bv, cv)`` (=
// av*bv + cv, single-rounded per lane); every other element type computes
// through ``float`` with ``fmaf`` (matching the sibling ``tile_binop`` compute
// path -- a double tile degrades through float exactly as tile_binop does).
// ``fmaf`` / ``__hfma2`` are fused single-rounded, so the GPU and the CPU pure
// lowerings agree.
template <typename T, int VLEN, bool BroadcastA, bool BroadcastB, bool BroadcastC, bool Masked>
DACE_DFI void tile_fma(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c,
                       const bool* __restrict__ mask) {
#if defined(__CUDACC__)
  // FP16x2 path: __half tile, even width, unmasked. Two lanes per __hfma2.
  if constexpr (__is_same(T, __half) && (VLEN % 2 == 0) && !Masked) {
#pragma unroll
    for (int i = 0; i < VLEN; i += 2) {
      const __half2 av = BroadcastA ? __half2half2(a[0]) : _load_half2(&a[i]);
      const __half2 bv = BroadcastB ? __half2half2(b[0]) : _load_half2(&b[i]);
      const __half2 cv = BroadcastC ? __half2half2(c[0]) : _load_half2(&c[i]);
      const __half2 rv = __hfma2(av, bv, cv);  // av*bv + cv
      _store_half2(&out[i], rv);
    }
    return;
  }
#endif
#pragma unroll
  for (int i = 0; i < VLEN; ++i) {
    const float af = _cuda_to_compute<T>(BroadcastA ? a[0] : a[i]);
    const float bf = _cuda_to_compute<T>(BroadcastB ? b[0] : b[i]);
    const float cf = _cuda_to_compute<T>(BroadcastC ? c[0] : c[i]);
    const T rv = _cuda_from_compute<T>(fmaf(af, bf, cf));  // single-rounded a*b + c
    if constexpr (Masked)
      out[i] = mask[i] ? rv : T(0);
    else
      out[i] = rv;
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
      const __half2 av = Broadcast ? __half2half2(a[0]) : _load_half2(&a[i]);
      const __half2 rv = _half2_unop_apply<Op>(av);
      _store_half2(&out[i], rv);
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
// ``Align`` is the byte alignment of the ARRAY side that the caller can prove; it defaults to the
// element alignment, which selects the per-element loop below -- so a caller that does not supply
// it gets exactly the code this template emitted before the parameter existed. ``Shift`` is how
// many elements ``src`` sits ABOVE that aligned address: 0 means ``src`` itself is aligned,
// nonzero routes to the aligned-window read (``half_read_shifted``).
template <typename T, int VLEN, bool Masked, int Align = alignof(T), int Shift = 0>
DACE_DFI void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                        std::int64_t stride = 1) {
#if defined(__CUDACC__)
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

// ----------------------------- tile_store -----------------------------
// Mirror of ``tile_load``: here the ARRAY side ``Align`` describes is ``dst``.
template <typename T, int VLEN, bool Masked, int Align = alignof(T)>
DACE_DFI void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                         std::int64_t stride = 1) {
#if defined(__CUDACC__)
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
// ``stride`` mirrors tile_load's: 1 per thread, the lane count under CUDA_WARP.
template <typename IdxT, int VLEN>
DACE_DFI void tile_mask_gen(bool* __restrict__ out, IdxT base, IdxT ub, IdxT stride = 1) {
#pragma unroll
  for (int i = 0; i < VLEN; ++i) out[i] = (base + IdxT(i) * stride) < ub;
}

// ----------------------------- tile_reduce ----------------------------
// Horizontal reduction of a VLEN-lane tile to a SINGLE scalar (an in-map / per-tile
// reduction: ``acc = sum/max/min/prod over the tile``). ``Op`` is the reduction op
// ('+' sum, '*' prod, 'm' min, 'M' max). Returns the reduced ELEMENT (a ``__half`` for
// an fp16 tile), not a vector.
//
// An fp16 tile folds as a balanced tree of half2 ops (consecutive pairs (0,1)(2,3)...;
// an odd trailing element forwards unchanged). CUDA has NO single "reduce half2 -> half"
// intrinsic, so we compose one: pack the VLEN lanes into VLEN/2 ``half2`` values and fold
// that array as a balanced tree of the half2 intrinsic (``__hadd2`` / ..., two lanes per op
// -- a sequence of half2 trees), then combine the surviving half2's two lanes into one
// ``__half`` with the scalar combine (``__hadd`` / ...). The O(log VLEN) critical path lets
// the compiler re-vectorise the partials. Every other element type (fp32 / fp64) folds
// through the plain per-lane scalar accumulate. Over a compile-time-constant ``VLEN`` every
// loop unrolls.
template <typename T, int VLEN, char Op>
DACE_DFI T tile_reduce(const T* __restrict__ src) {
#if defined(__CUDACC__)
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

template <typename T, int VLEN, bool BroadcastA, bool BroadcastB, bool BroadcastC, bool Masked, typename Out,
          typename A, typename B, typename C>
DACE_DFI typename std::enable_if<VLEN == 1, void>::type tile_fma(Out&& out, A&& a, B&& b, C&& c,
                                                                const bool* __restrict__ mask) {
  const float af = _cuda_to_compute<T>(tile_load_value<T>(a));
  const float bf = _cuda_to_compute<T>(tile_load_value<T>(b));
  const float cf = _cuda_to_compute<T>(tile_load_value<T>(c));
  T rv = _cuda_from_compute<T>(fmaf(af, bf, cf));
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
