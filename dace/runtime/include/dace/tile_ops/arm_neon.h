// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// ARM NEON (AArch64 Advanced SIMD, 128-bit ``q`` registers) backend of the
// K=1 tile-op intrinsics. Exposes the SAME ``dace::tileops`` function
// signatures as ``scalar.h`` (the portable reference) but lowers the
// vectorisable paths to ``<arm_neon.h>`` intrinsics; the chosen-backend
// expansion pulls in exactly one backend header.
//
// Lane widths: float -> float32x4_t (W=4), double -> float64x2_t (W=2),
// int32_t -> int32x4_t (W=4), int64_t -> int64x2_t (W=2).
//
// Masked semantics (carried over unchanged from the reference):
//   * tile PRODUCERS (binop / merge / load / gather) ZERO-FILL inactive lanes
//     and GUARD the read. Vector producers compute the full register then blend
//     against zero with ``vbslq`` using a lane bitmask; tile_load / tile_gather
//     stay scalar-guarded because an inactive lane could index past the user
//     array (vector-loading it would dereference OOB).
//   * array WRITERS (store / scatter) RMW skip-inactive. NEON has NO masked or
//     partial store -- every ``vst1q`` writes the full 128-bit register / W
//     addresses -- so a full-width store at a masked tail writes past the array
//     end (the TSVC s2710 OOB segfault). The ONLY OOB-safe masked writer is a
//     scalar gated loop; the compiler still autovectorises the active in-bounds
//     run.
//
// Native gaps (-> correct scalar loop, see the per-function notes):
//   * integer divide (no ``vdivq_s32`` / ``vdivq_s64``; NEON has no integer
//     division at all),
//   * int64 multiply (no ``vmulq_s64``; SIMD integer multiply tops out at
//     32-bit element width),
//   * int64 min / max (no ``vminq_s64`` / ``vmaxq_s64``; integer min/max only
//     8/16/32-bit),
//   * ALL gather / scatter (no index-vector load/store intrinsics),
//   * ALL masked stores / scatters and arbitrary strided stores (no masked /
//     partial store -- OOB-unsafe otherwise),
//   * ``!=`` has no ``vcneq_*`` intrinsic; built from NOT / XOR of ``vceqq_*``.
#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>



#if !defined(__ARM_NEON)
#error Included the NEON tile-op header without NEON support (AArch64 Advanced SIMD)
#endif

namespace dace {
namespace tileops {

// ===========================================================================
// Scalar reference body (shared correctness path + gap fallback)
// ===========================================================================

// Per-lane binary op. Comparisons / logicals yield ``T(1)`` / ``T(0)`` (the
// tile stores the condition as the element type, matching the reference).
template <typename T, char Op>
inline T tile_apply(T a, T b) {
  if constexpr (Op == '+') return a + b;
  else if constexpr (Op == '-') return a - b;
  else if constexpr (Op == '*') return a * b;
  else if constexpr (Op == '/') return a / b;
  else if constexpr (Op == '%') return py_mod(a, b);  // Python/NumPy modulo (not C's); via the scalar path
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

// ===========================================================================
// NEON lane-mask helpers
// ===========================================================================

// Build a per-lane bitmask (all-ones / all-zeros) from a ``const bool*`` mask
// (1 byte per lane). Each active lane becomes the all-ones value; the vbslq
// blend below then selects computed-vs-zero per lane. Scalar-built (NEON has
// no byte->lane widen-to-mask of arbitrary width); the surrounding op already
// strides W lanes, so this is W scalar reads per tile chunk.
inline uint32x4_t neon_mask4_u32(const bool* __restrict__ mask) {
  uint32x4_t m = vdupq_n_u32(0);
  m = vsetq_lane_u32(mask[0] ? ~0u : 0u, m, 0);
  m = vsetq_lane_u32(mask[1] ? ~0u : 0u, m, 1);
  m = vsetq_lane_u32(mask[2] ? ~0u : 0u, m, 2);
  m = vsetq_lane_u32(mask[3] ? ~0u : 0u, m, 3);
  return m;
}
inline uint64x2_t neon_mask2_u64(const bool* __restrict__ mask) {
  uint64x2_t m = vdupq_n_u64(0);
  m = vsetq_lane_u64(mask[0] ? ~0ull : 0ull, m, 0);
  m = vsetq_lane_u64(mask[1] ? ~0ull : 0ull, m, 1);
  return m;
}

// Comparison lane-mask -> element-typed ``1`` / ``0`` tile values.
inline float32x4_t neon_select_0_1_f32(uint32x4_t mask) {
  return vbslq_f32(mask, vdupq_n_f32(1.0f), vdupq_n_f32(0.0f));
}
inline float64x2_t neon_select_0_1_f64(uint64x2_t mask) {
  return vbslq_f64(mask, vdupq_n_f64(1.0), vdupq_n_f64(0.0));
}
// All-ones lane mask is 2's-complement ``-1``; AND with a splatted ``1`` to get
// integer ``1`` / ``0`` (uniform across both int widths, no reinterpret subtlety).
inline int32x4_t neon_select_0_1_s32(uint32x4_t mask) {
  return vandq_s32(vreinterpretq_s32_u32(mask), vdupq_n_s32(1));
}
inline int64x2_t neon_select_0_1_s64(uint64x2_t mask) {
  return vandq_s64(vreinterpretq_s64_u64(mask), vdupq_n_s64(1));
}

// ===========================================================================
// Vector op kernels (one register's worth of lanes); op selected at compile
// time. Returns the result register; producers blend against zero afterwards.
// ===========================================================================

// Floating point: every op except the logical And / Or (which are C++
// truthiness producing ``T(1)`` / ``T(0)`` and have no NEON intrinsic) maps to
// a native intrinsic. And / Or -> scalar.
template <char Op>
inline constexpr bool neon_float_has_vector =
    (Op != '&' && Op != '|');

template <char Op>
inline float32x4_t neon_binop_f32(float32x4_t a, float32x4_t b) {
  if constexpr (Op == '+') return vaddq_f32(a, b);
  else if constexpr (Op == '-') return vsubq_f32(a, b);
  else if constexpr (Op == '*') return vmulq_f32(a, b);
  else if constexpr (Op == '/') return vdivq_f32(a, b);
  else if constexpr (Op == 'm') return vminq_f32(a, b);
  else if constexpr (Op == 'M') return vmaxq_f32(a, b);
  else if constexpr (Op == '<') return neon_select_0_1_f32(vcltq_f32(a, b));
  else if constexpr (Op == 'l') return neon_select_0_1_f32(vcleq_f32(a, b));
  else if constexpr (Op == '>') return neon_select_0_1_f32(vcgtq_f32(a, b));
  else if constexpr (Op == 'g') return neon_select_0_1_f32(vcgeq_f32(a, b));
  else if constexpr (Op == '=') return neon_select_0_1_f32(vceqq_f32(a, b));
  else /* Ne */ return neon_select_0_1_f32(vmvnq_u32(vceqq_f32(a, b)));
}

template <char Op>
inline float64x2_t neon_binop_f64(float64x2_t a, float64x2_t b) {
  if constexpr (Op == '+') return vaddq_f64(a, b);
  else if constexpr (Op == '-') return vsubq_f64(a, b);
  else if constexpr (Op == '*') return vmulq_f64(a, b);
  else if constexpr (Op == '/') return vdivq_f64(a, b);
  else if constexpr (Op == 'm') return vminq_f64(a, b);
  else if constexpr (Op == 'M') return vmaxq_f64(a, b);
  else if constexpr (Op == '<') return neon_select_0_1_f64(vcltq_f64(a, b));
  else if constexpr (Op == 'l') return neon_select_0_1_f64(vcleq_f64(a, b));
  else if constexpr (Op == '>') return neon_select_0_1_f64(vcgtq_f64(a, b));
  else if constexpr (Op == 'g') return neon_select_0_1_f64(vcgeq_f64(a, b));
  else if constexpr (Op == '=') return neon_select_0_1_f64(vceqq_f64(a, b));
  // No vmvnq_u64: XOR the 64-bit eq mask with all-ones to get ``!=``.
  else /* Ne */ return neon_select_0_1_f64(veorq_u64(vceqq_f64(a, b), vdupq_n_u64(~0ull)));
}

// int32: Add/Sub/Mult/Min/Max and all compares are native. Div has no NEON
// integer-divide instruction; And/Or are C++ truthiness -> handled scalar.
template <char Op>
inline constexpr bool neon_s32_has_vector =
    (Op == '+' || Op == '-' || Op == '*' ||
     Op == 'm' || Op == 'M' || Op == '<' ||
     Op == 'l' || Op == '>' || Op == 'g' ||
     Op == '=' || Op == '!');

template <char Op>
inline int32x4_t neon_binop_s32(int32x4_t a, int32x4_t b) {
  if constexpr (Op == '+') return vaddq_s32(a, b);
  else if constexpr (Op == '-') return vsubq_s32(a, b);
  else if constexpr (Op == '*') return vmulq_s32(a, b);
  else if constexpr (Op == 'm') return vminq_s32(a, b);
  else if constexpr (Op == 'M') return vmaxq_s32(a, b);
  else if constexpr (Op == '<') return neon_select_0_1_s32(vcltq_s32(a, b));
  else if constexpr (Op == 'l') return neon_select_0_1_s32(vcleq_s32(a, b));
  else if constexpr (Op == '>') return neon_select_0_1_s32(vcgtq_s32(a, b));
  else if constexpr (Op == 'g') return neon_select_0_1_s32(vcgeq_s32(a, b));
  else if constexpr (Op == '=') return neon_select_0_1_s32(vceqq_s32(a, b));
  else /* Ne */ return neon_select_0_1_s32(vmvnq_u32(vceqq_s32(a, b)));
}

// int64: only Add/Sub and the (AArch64) 64-bit compares are native. Mult / Min /
// Max / Div have no NEON intrinsic; And/Or are truthiness -> handled scalar.
template <char Op>
inline constexpr bool neon_s64_has_vector =
    (Op == '+' || Op == '-' || Op == '<' ||
     Op == 'l' || Op == '>' || Op == 'g' ||
     Op == '=' || Op == '!');

template <char Op>
inline int64x2_t neon_binop_s64(int64x2_t a, int64x2_t b) {
  if constexpr (Op == '+') return vaddq_s64(a, b);
  else if constexpr (Op == '-') return vsubq_s64(a, b);
  else if constexpr (Op == '<') return neon_select_0_1_s64(vcltq_s64(a, b));
  else if constexpr (Op == 'l') return neon_select_0_1_s64(vcleq_s64(a, b));
  else if constexpr (Op == '>') return neon_select_0_1_s64(vcgtq_s64(a, b));
  else if constexpr (Op == 'g') return neon_select_0_1_s64(vcgeq_s64(a, b));
  else if constexpr (Op == '=') return neon_select_0_1_s64(vceqq_s64(a, b));
  // No vmvnq_u64: XOR the 64-bit eq mask with all-ones to get ``!=``.
  else /* Ne */ return neon_select_0_1_s64(veorq_u64(vceqq_s64(a, b), vdupq_n_u64(~0ull)));
}

// ----------------------------- tile_binop -----------------------------
// out[i] = a-operand <op> b-operand ; ZERO-FILL inactive (operand reads are
// in-tile, always safe to evaluate). Vector body where a native intrinsic
// exists for (T, Op); scalar tail + all gap (T, Op) pairs via tile_apply.
template <typename T, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask, int vlen) {
  int i = 0;
  if constexpr (std::is_same<T, float>::value) {
    if constexpr (neon_float_has_vector<Op>) {
      constexpr int W = 4;
      const float32x4_t va_b = (BroadcastA) ? vdupq_n_f32(a[0]) : vdupq_n_f32(0.0f);
      const float32x4_t vb_b = (BroadcastB) ? vdupq_n_f32(b[0]) : vdupq_n_f32(0.0f);
      for (; i + W <= vlen; i += W) {
        float32x4_t va = (!BroadcastA) ? vld1q_f32(a + i) : va_b;
        float32x4_t vb = (!BroadcastB) ? vld1q_f32(b + i) : vb_b;
        float32x4_t vc = neon_binop_f32<Op>(va, vb);
        if constexpr (Masked) vc = vbslq_f32(neon_mask4_u32(mask + i), vc, vdupq_n_f32(0.0f));
        vst1q_f32(out + i, vc);
      }
    }
  } else if constexpr (std::is_same<T, double>::value) {
    if constexpr (neon_float_has_vector<Op>) {
      constexpr int W = 2;
      const float64x2_t va_b = (BroadcastA) ? vdupq_n_f64(a[0]) : vdupq_n_f64(0.0);
      const float64x2_t vb_b = (BroadcastB) ? vdupq_n_f64(b[0]) : vdupq_n_f64(0.0);
      for (; i + W <= vlen; i += W) {
        float64x2_t va = (!BroadcastA) ? vld1q_f64(a + i) : va_b;
        float64x2_t vb = (!BroadcastB) ? vld1q_f64(b + i) : vb_b;
        float64x2_t vc = neon_binop_f64<Op>(va, vb);
        if constexpr (Masked) vc = vbslq_f64(neon_mask2_u64(mask + i), vc, vdupq_n_f64(0.0));
        vst1q_f64(out + i, vc);
      }
    }
  } else if constexpr (std::is_same<T, std::int32_t>::value) {
    if constexpr (neon_s32_has_vector<Op>) {
      constexpr int W = 4;
      const int32x4_t va_b = (BroadcastA) ? vdupq_n_s32(a[0]) : vdupq_n_s32(0);
      const int32x4_t vb_b = (BroadcastB) ? vdupq_n_s32(b[0]) : vdupq_n_s32(0);
      for (; i + W <= vlen; i += W) {
        int32x4_t va = (!BroadcastA) ? vld1q_s32(a + i) : va_b;
        int32x4_t vb = (!BroadcastB) ? vld1q_s32(b + i) : vb_b;
        int32x4_t vc = neon_binop_s32<Op>(va, vb);
        if constexpr (Masked)
          vc = vbslq_s32(neon_mask4_u32(mask + i), vc, vdupq_n_s32(0));
        vst1q_s32(out + i, vc);
      }
    }
  } else if constexpr (std::is_same<T, std::int64_t>::value) {
    if constexpr (neon_s64_has_vector<Op>) {
      constexpr int W = 2;
      const int64x2_t va_b = (BroadcastA) ? vdupq_n_s64(a[0]) : vdupq_n_s64(0);
      const int64x2_t vb_b = (BroadcastB) ? vdupq_n_s64(b[0]) : vdupq_n_s64(0);
      for (; i + W <= vlen; i += W) {
        int64x2_t va = (!BroadcastA) ? vld1q_s64(a + i) : va_b;
        int64x2_t vb = (!BroadcastB) ? vld1q_s64(b + i) : vb_b;
        int64x2_t vc = neon_binop_s64<Op>(va, vb);
        if constexpr (Masked)
          vc = vbslq_s64(neon_mask2_u64(mask + i), vc, vdupq_n_s64(0));
        vst1q_s64(out + i, vc);
      }
    }
  }
  // Scalar tail + every gap (int div, i64 mul/min/max, And/Or, non-NEON T).
  for (; i < vlen; ++i) {
    const T av = (!BroadcastA) ? a[i] : a[0];
    const T bv = (!BroadcastB) ? b[i] : b[0];
    if constexpr (Masked) out[i] = mask[i] ? tile_apply<T, Op>(av, bv) : T(0);
    else out[i] = tile_apply<T, Op>(av, bv);
  }
}
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask) {
  tile_binop<T, Op, BroadcastA, BroadcastB, Masked>(out, a, b, mask, VLEN);
}

// ----------------------------- tile_ite -----------------------------
// out[i] = cond[i] ? t : e ; ZERO-FILL inactive. Vector blend when CondT == T
// and T is a NEON type AND both operands are full tiles (matching lane widths);
// every other shape (broadcast operands, mismatched cond width, non-NEON T)
// falls to the scalar ternary.
template <typename T, typename CondT, bool BroadcastThen, bool BroadcastElse, bool Masked>
inline void tile_ite(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                       const T* __restrict__ e, const bool* __restrict__ mask, int vlen) {
  int i = 0;
  constexpr bool kSameWidth = std::is_same<CondT, T>::value;
  constexpr bool kTileTile = (!BroadcastThen) && (!BroadcastElse);
  if constexpr (kSameWidth && kTileTile && std::is_same<T, float>::value) {
    constexpr int W = 4;
    for (; i + W <= vlen; i += W) {
      // cond stored as element type 1/0; nonzero -> all-ones lane mask.
      uint32x4_t cm = vmvnq_u32(vceqq_f32(vld1q_f32(cond + i), vdupq_n_f32(0.0f)));
      float32x4_t vc = vbslq_f32(cm, vld1q_f32(t + i), vld1q_f32(e + i));
      if constexpr (Masked) vc = vbslq_f32(neon_mask4_u32(mask + i), vc, vdupq_n_f32(0.0f));
      vst1q_f32(out + i, vc);
    }
  } else if constexpr (kSameWidth && kTileTile && std::is_same<T, double>::value) {
    constexpr int W = 2;
    for (; i + W <= vlen; i += W) {
      uint64x2_t cm =
          veorq_u64(vceqq_f64(vld1q_f64(cond + i), vdupq_n_f64(0.0)), vdupq_n_u64(~0ull));
      float64x2_t vc = vbslq_f64(cm, vld1q_f64(t + i), vld1q_f64(e + i));
      if constexpr (Masked) vc = vbslq_f64(neon_mask2_u64(mask + i), vc, vdupq_n_f64(0.0));
      vst1q_f64(out + i, vc);
    }
  } else if constexpr (kSameWidth && kTileTile && std::is_same<T, std::int32_t>::value) {
    constexpr int W = 4;
    for (; i + W <= vlen; i += W) {
      uint32x4_t cm = vmvnq_u32(vceqq_s32(vld1q_s32(cond + i), vdupq_n_s32(0)));
      int32x4_t vc = vbslq_s32(cm, vld1q_s32(t + i), vld1q_s32(e + i));
      if constexpr (Masked) vc = vbslq_s32(neon_mask4_u32(mask + i), vc, vdupq_n_s32(0));
      vst1q_s32(out + i, vc);
    }
  } else if constexpr (kSameWidth && kTileTile && std::is_same<T, std::int64_t>::value) {
    constexpr int W = 2;
    for (; i + W <= vlen; i += W) {
      uint64x2_t cm =
          veorq_u64(vceqq_s64(vld1q_s64(cond + i), vdupq_n_s64(0)), vdupq_n_u64(~0ull));
      int64x2_t vc = vbslq_s64(cm, vld1q_s64(t + i), vld1q_s64(e + i));
      if constexpr (Masked) vc = vbslq_s64(neon_mask2_u64(mask + i), vc, vdupq_n_s64(0));
      vst1q_s64(out + i, vc);
    }
  }
  // Scalar tail + every mismatched-width / broadcast / non-NEON shape.
  for (; i < vlen; ++i) {
    const T tv = (!BroadcastThen) ? t[i] : t[0];
    const T ev = (!BroadcastElse) ? e[i] : e[0];
    if constexpr (Masked) out[i] = mask[i] ? (cond[i] ? tv : ev) : T(0);
    else out[i] = cond[i] ? tv : ev;
  }
}
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
  tile_ite<T, CondT, BroadcastThen, BroadcastElse, Masked>(out, cond, t, e, mask, VLEN);
}

// ----------------------------- tile_load ------------------------------
// dst[i] = src[i * stride] ; ZERO-FILL inactive + GUARDED read.
//
// Unmasked, stride == 1: contiguous vld1q -> vst1q (in-tile, full vectors).
// Masked OR strided: scalar guarded loop. The masked form MUST be scalar
// (an inactive tail lane could index past the user array; a vector vld1q
// would dereference OOB). Arbitrary runtime stride has no NEON gather-load.
template <typename T, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                      int vlen, std::int64_t stride = 1) {
  if constexpr (!Masked) {
    if (stride == 1) {
      int i = 0;
      if constexpr (std::is_same<T, float>::value) {
        for (; i + 4 <= vlen; i += 4) vst1q_f32(dst + i, vld1q_f32(src + i));
      } else if constexpr (std::is_same<T, double>::value) {
        for (; i + 2 <= vlen; i += 2) vst1q_f64(dst + i, vld1q_f64(src + i));
      } else if constexpr (std::is_same<T, std::int32_t>::value) {
        for (; i + 4 <= vlen; i += 4) vst1q_s32(dst + i, vld1q_s32(src + i));
      } else if constexpr (std::is_same<T, std::int64_t>::value) {
        for (; i + 2 <= vlen; i += 2) vst1q_s64(dst + i, vld1q_s64(src + i));
      }
      for (; i < vlen; ++i) dst[i] = src[i];
      return;
    }
  }
  // Masked (guarded read) and arbitrary strided: scalar.
  for (int i = 0; i < vlen; ++i) {
    if constexpr (Masked) dst[i] = mask[i] ? src[i * stride] : T(0);
    else dst[i] = src[i * stride];
  }
}
template <typename T, int VLEN, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                      std::int64_t stride = 1) {
  tile_load<T, Masked>(dst, src, mask, VLEN, stride);
}

// ----------------------------- tile_store -----------------------------
// dst[i * stride] = src[i] ; RMW skip-inactive.
//
// Unmasked, stride == 1: contiguous vld1q -> vst1q. Masked OR strided: scalar
// gated loop. The masked / strided store MUST be scalar -- NEON has no masked
// or partial store, so a full-width vst1q at a masked tail writes past the
// array end (the s2710 OOB segfault).
template <typename T, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                       int vlen, std::int64_t stride = 1) {
  if constexpr (!Masked) {
    if (stride == 1) {
      int i = 0;
      if constexpr (std::is_same<T, float>::value) {
        for (; i + 4 <= vlen; i += 4) vst1q_f32(dst + i, vld1q_f32(src + i));
      } else if constexpr (std::is_same<T, double>::value) {
        for (; i + 2 <= vlen; i += 2) vst1q_f64(dst + i, vld1q_f64(src + i));
      } else if constexpr (std::is_same<T, std::int32_t>::value) {
        for (; i + 4 <= vlen; i += 4) vst1q_s32(dst + i, vld1q_s32(src + i));
      } else if constexpr (std::is_same<T, std::int64_t>::value) {
        for (; i + 2 <= vlen; i += 2) vst1q_s64(dst + i, vld1q_s64(src + i));
      }
      for (; i < vlen; ++i) dst[i] = src[i];
      return;
    }
  }
  // Masked (gated, OOB-safe) and arbitrary strided: scalar.
  for (int i = 0; i < vlen; ++i) {
    if constexpr (Masked) { if (mask[i]) dst[i * stride] = src[i]; }
    else dst[i * stride] = src[i];
  }
}
template <typename T, int VLEN, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                       std::int64_t stride = 1) {
  tile_store<T, Masked>(dst, src, mask, VLEN, stride);
}

// ---------------------------- tile_gather -----------------------------
// dst[i] = src[idx[i]] ; ZERO-FILL inactive + GUARDED read.
//
// NEON has no index-vector load (gather) intrinsic -> scalar (guarded when
// masked, so a garbage / OOB index on an inactive lane is never dereferenced).
template <typename T, typename IdxT, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask, int vlen) {
  for (int i = 0; i < vlen; ++i) {
    if constexpr (Masked) dst[i] = mask[i] ? src[idx[i]] : T(0);
    else dst[i] = src[idx[i]];
  }
}
template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask) {
  tile_gather<T, IdxT, Masked>(dst, src, idx, mask, VLEN);
}

// ---------------------------- tile_scatter ----------------------------
// dst[idx[i]] = src[i] ; RMW skip-inactive.
//
// NEON has no index-vector store (scatter) intrinsic -> scalar (gated when
// masked; an inactive lane never writes, so a garbage / OOB index is safe).
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

// ---------------------------- tile_mask_gen ----------------------------
// out[l] = (base + l) < ub. NEON (AArch64): 64-bit-lane compare (vcltq_s64,
// W=2) extracted to bool bytes; scalar tail.
template <typename IdxT, int VLEN>
inline void tile_mask_gen(bool* __restrict__ out, IdxT base, IdxT ub) {
  constexpr int W = 2;
  const int64x2_t ubv = vdupq_n_s64((std::int64_t)ub);
  int i = 0;
  for (; i + W <= VLEN; i += W) {
    const std::int64_t lb = (std::int64_t)base + i;
    const std::int64_t seed[W] = {lb, lb + 1};
    int64x2_t lanes = vld1q_s64(seed);
    uint64x2_t cmp = vcltq_s64(lanes, ubv);  // base+l < ub
    out[i + 0] = vgetq_lane_u64(cmp, 0) != 0;
    out[i + 1] = vgetq_lane_u64(cmp, 1) != 0;
  }
  for (; i < VLEN; ++i) out[i] = (base + IdxT(i)) < ub;
}

}  // namespace tileops
}  // namespace dace
