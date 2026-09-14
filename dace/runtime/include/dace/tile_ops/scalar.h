// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Scalar (portable) backend of the K=1 tile-op intrinsics: the reference implementation and
// always-available fallback; avx512/avx2/arm_neon/arm_sve mirror the same signatures.
//
// Op codes (binop only): + - * / % (C modulo) p (Python modulo) m/M min/max, < l > g = !
// comparisons (yield T(1)/T(0)), & | logical. Producers zero-fill inactive lanes and guard
// the read; array writers (store/scatter) skip inactive lanes instead (RMW).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

// std::is_pointer does not recognize a __restrict__-qualified pointer as a pointer, which
// would let a restrict-qualified tile-load source silently fall into the by-value broadcast
// overload below instead of the strided per-lane load. Strip restrict before the test.
namespace dace_tileops_detail {
template <typename T>
struct _strip_restrict {
  using type = T;
};
template <typename T>
struct _strip_restrict<T* __restrict__> {
  using type = T*;
};
template <typename Src>
inline constexpr bool _is_pointer_like =
    std::is_pointer_v<typename _strip_restrict<std::remove_reference_t<Src>>::type>;
}  // namespace dace_tileops_detail

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
  if constexpr (Op == '+')
    return a + b;
  else if constexpr (Op == '-')
    return a - b;
  else if constexpr (Op == '*')
    return a * b;
  else if constexpr (Op == '/')
    return a / b;
  else if constexpr (Op == '%')
    return c_mod(a, b);
  else if constexpr (Op == 'p')
    return py_mod(a, b);
  else if constexpr (Op == 'm')
    return std::min(a, b);
  else if constexpr (Op == 'M')
    return std::max(a, b);
  else if constexpr (Op == '<')
    return (a < b) ? T(1) : T(0);
  else if constexpr (Op == 'l')
    return (a <= b) ? T(1) : T(0);
  else if constexpr (Op == '>')
    return (a > b) ? T(1) : T(0);
  else if constexpr (Op == 'g')
    return (a >= b) ? T(1) : T(0);
  else if constexpr (Op == '=')
    return (a == b) ? T(1) : T(0);
  else if constexpr (Op == '!')
    return (a != b) ? T(1) : T(0);
  else if constexpr (Op == '&')
    return (a && b) ? T(1) : T(0);
  else /* '|' */
    return (a || b) ? T(1) : T(0);
}

// tile_binop
// out[i] = a-operand <op> b-operand ; ZERO-FILL inactive (operand reads are
// in-tile, always safe to evaluate).
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked>
inline void tile_binop(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                       const bool* __restrict__ mask) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    const T av = BroadcastA ? a[0] : a[i];
    const T bv = BroadcastB ? b[0] : b[i];
    if constexpr (Masked)
      out[i] = mask[i] ? tile_apply<T, Op>(av, bv) : T(0);
    else
      out[i] = tile_apply<T, Op>(av, bv);
  }
}

// tile_fma: out[i] = fma(a, b, c) = a*b + c, single rounding. Zero-fill inactive.
template <typename T, int VLEN, bool BroadcastA, bool BroadcastB, bool BroadcastC, bool Masked>
inline void tile_fma(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c,
                     const bool* __restrict__ mask) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    const T av = BroadcastA ? a[0] : a[i];
    const T bv = BroadcastB ? b[0] : b[i];
    const T cv = BroadcastC ? c[0] : c[i];
    if constexpr (Masked)
      out[i] = mask[i] ? T(std::fma(av, bv, cv)) : T(0);
    else
      out[i] = T(std::fma(av, bv, cv));
  }
}

// Per-lane unary op. Op codes: n neg, ! not, a abs, e exp, l log, s sqrt, S sin, C cos,
// f floor, c ceil, t tanh.
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

// tile_unop
// out[i] = <op> a-operand ; ZERO-FILL inactive (operand read is in-tile, safe
// to evaluate even on an inactive lane).
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked>
inline void tile_unop(T* __restrict__ out, const T* __restrict__ a, const bool* __restrict__ mask) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    const T av = Broadcast ? a[0] : a[i];
    if constexpr (Masked)
      out[i] = mask[i] ? tile_unop_apply<T, Op>(av) : T(0);
    else
      out[i] = tile_unop_apply<T, Op>(av);
  }
}

// tile_ite
// out[i] = cond[i] ? t : e ; ZERO-FILL inactive.
template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked>
inline void tile_ite(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                     const T* __restrict__ e, const bool* __restrict__ mask) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    const T tv = BroadcastThen ? t[0] : t[i];
    const T ev = BroadcastElse ? e[0] : e[i];
    if constexpr (Masked)
      out[i] = mask[i] ? (cond[i] ? tv : ev) : T(0);
    else
      out[i] = cond[i] ? tv : ev;
  }
}

// tile_load
// dst[i] = src[i * stride] ; ZERO-FILL inactive + GUARDED read (inactive lane
// never dereferences src, so an OOB tail lane is safe).
template <typename T, int VLEN, bool Masked>
inline void tile_load(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                      std::int64_t stride = 1) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked)
      dst[i] = mask[i] ? src[i * stride] : T(0);
    else
      dst[i] = src[i * stride];
  }
}

// Forward-declared: defined below with the VLEN==1 polymorphism block.
template <typename T>
inline T tile_load_value(const T& x) noexcept;
template <typename T>
inline T tile_load_value(const T* __restrict__ x) noexcept;
template <typename T, std::size_t N>
inline T tile_load_value(const T (&x)[N]) noexcept;

// VLEN>1 tile_load with a by-value src (Scalar/Symbol operand codegen materializes as
// `T _src = expr;`). SFINAE keeps this off the pointer overload above.
template <typename T, int VLEN, bool Masked, typename Src>
inline std::enable_if_t<(VLEN > 1) && !dace_tileops_detail::_is_pointer_like<Src>, void> tile_load(
    T* __restrict__ dst, Src&& src, const bool* __restrict__ mask, std::int64_t /*stride*/ = 1) {
  const T sv = tile_load_value<T>(src);
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked)
      dst[i] = mask[i] ? sv : T(0);
    else
      dst[i] = sv;
  }
}

// tile_store
// dst[i * stride] = src[i] ; RMW skip-inactive (inactive lane not written, so
// the destination array / OOB tail is never touched).
template <typename T, int VLEN, bool Masked>
inline void tile_store(T* __restrict__ dst, const T* __restrict__ src, const bool* __restrict__ mask,
                       std::int64_t stride = 1) {
  _dace_tile_vectorize(VLEN) for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) {
      if (mask[i]) dst[i * stride] = src[i];
    } else
      dst[i * stride] = src[i];
  }
}

// tile_gather
// dst[i] = src[idx[i]] ; ZERO-FILL inactive + GUARDED read (inactive lane never
// dereferences src[idx], so a garbage / OOB index is safe).
template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_gather(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                        const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked)
      dst[i] = mask[i] ? src[idx[i]] : T(0);
    else
      dst[i] = src[idx[i]];
  }
}

// tile_scatter
// dst[idx[i]] = src[i] ; RMW skip-inactive (inactive lane never written, so a
// garbage / OOB index is safe).
template <typename T, typename IdxT, int VLEN, bool Masked>
inline void tile_scatter(T* __restrict__ dst, const T* __restrict__ src, const IdxT* __restrict__ idx,
                         const bool* __restrict__ mask) {
  for (int i = 0; i < VLEN; ++i) {
    if constexpr (Masked) {
      if (mask[i]) dst[idx[i]] = src[i];
    } else
      dst[idx[i]] = src[i];
  }
}

// tile_mask_gen: iteration mask, out[l] = (base + l) < ub for l in 0..VLEN-1.
template <typename IdxT, int VLEN>
inline void tile_mask_gen(bool* __restrict__ out, IdxT base, IdxT ub) {
  for (int i = 0; i < VLEN; ++i) out[i] = (base + IdxT(i)) < ub;
}

// tile_reduce: horizontal reduction of a VLEN-lane tile to one scalar (Op: '+' sum, '*' prod,
// 'm' min, 'M' max). Balanced log-depth pairwise fold, matching the order the vectorized
// Reduce node's _dace_horizontal_tree uses so both paths agree numerically.
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

// VLEN=1 overloads. DaCe codegen may collapse a shape=(1,) transient to a plain T while
// other operands at the same call site stay T[1] (decays to T*); tile_addr below normalizes
// any of T, T&, T*, T[N] to T* before forwarding to the canonical pointer-shape body.

// tile_load_value extracts the single element from any kind of VLEN=1 tile operand.

template <typename T>
inline T tile_load_value(const T& x) noexcept {
  return x;
}
template <typename T>
inline T tile_load_value(const T* __restrict__ x) noexcept {
  return *x;
}
template <typename T, std::size_t N>
inline T tile_load_value(const T (&x)[N]) noexcept {
  return x[0];
}

template <typename T, typename V>
inline void tile_store_value(T& dst, V v) noexcept {
  dst = static_cast<T>(v);
}
template <typename T, typename V>
inline void tile_store_value(T* __restrict__ dst, V v) noexcept {
  *dst = static_cast<T>(v);
}
template <typename T, std::size_t N, typename V>
inline void tile_store_value(T (&dst)[N], V v) noexcept {
  dst[0] = static_cast<T>(v);
}

// VLEN=1 tile_binop.
template <typename T, int VLEN, char Op, bool BroadcastA, bool BroadcastB, bool Masked, typename Out, typename A,
          typename B>
inline std::enable_if_t<VLEN == 1, void> tile_binop(Out&& out, A&& a, B&& b, const bool* __restrict__ mask) {
  const T av = tile_load_value<T>(a);
  const T bv = tile_load_value<T>(b);
  T rv = tile_apply<T, Op>(av, bv);
  if constexpr (Masked)
    tile_store_value<T>(out, mask[0] ? rv : T(0));
  else
    tile_store_value<T>(out, rv);
}

// VLEN=1 tile_fma.
template <typename T, int VLEN, bool BroadcastA, bool BroadcastB, bool BroadcastC, bool Masked, typename Out,
          typename A, typename B, typename C>
inline std::enable_if_t<VLEN == 1, void> tile_fma(Out&& out, A&& a, B&& b, C&& c, const bool* __restrict__ mask) {
  const T av = tile_load_value<T>(a);
  const T bv = tile_load_value<T>(b);
  const T cv = tile_load_value<T>(c);
  T rv = T(std::fma(av, bv, cv));
  if constexpr (Masked)
    tile_store_value<T>(out, mask[0] ? rv : T(0));
  else
    tile_store_value<T>(out, rv);
}

// VLEN=1 tile_unop.
template <typename T, int VLEN, char Op, bool Broadcast, bool Masked, typename Out, typename A>
inline std::enable_if_t<VLEN == 1, void> tile_unop(Out&& out, A&& a, const bool* __restrict__ mask) {
  const T av = tile_load_value<T>(a);
  T rv = tile_unop_apply<T, Op>(av);
  if constexpr (Masked)
    tile_store_value<T>(out, mask[0] ? rv : T(0));
  else
    tile_store_value<T>(out, rv);
}

// VLEN=1 tile_ite.
template <typename T, typename CondT, int VLEN, bool BroadcastThen, bool BroadcastElse, bool Masked, typename Out,
          typename C, typename TThen, typename EElse>
inline std::enable_if_t<VLEN == 1, void> tile_ite(Out&& out, C&& cond, TThen&& t, EElse&& e,
                                                  const bool* __restrict__ mask) {
  const CondT cv = tile_load_value<CondT>(cond);
  const T tv = tile_load_value<T>(t);
  const T ev = tile_load_value<T>(e);
  T rv = cv ? tv : ev;
  if constexpr (Masked)
    tile_store_value<T>(out, mask[0] ? rv : T(0));
  else
    tile_store_value<T>(out, rv);
}

// VLEN=1 tile_load. ``stride`` is irrelevant (one lane).
template <typename T, int VLEN, bool Masked, typename Dst, typename Src>
inline std::enable_if_t<VLEN == 1, void> tile_load(Dst&& dst, Src&& src, const bool* __restrict__ mask,
                                                   std::int64_t /*stride*/ = 1) {
  const T sv = tile_load_value<T>(src);
  if constexpr (Masked)
    tile_store_value<T>(dst, mask[0] ? sv : T(0));
  else
    tile_store_value<T>(dst, sv);
}

// VLEN=1 tile_store.
template <typename T, int VLEN, bool Masked, typename Dst, typename Src>
inline std::enable_if_t<VLEN == 1, void> tile_store(Dst&& dst, Src&& src, const bool* __restrict__ mask,
                                                    std::int64_t /*stride*/ = 1) {
  const T sv = tile_load_value<T>(src);
  if constexpr (Masked) {
    if (mask[0]) tile_store_value<T>(dst, sv);
  } else
    tile_store_value<T>(dst, sv);
}

// VLEN=1 tile_gather: ``src`` array stays a pointer (the gather indexes
// into the outer buffer); ``dst`` and the index value may be scalars.
template <typename T, typename IdxT, int VLEN, bool Masked, typename Dst, typename Idx>
inline std::enable_if_t<VLEN == 1, void> tile_gather(Dst&& dst, const T* __restrict__ src, Idx&& idx,
                                                     const bool* __restrict__ mask) {
  const IdxT iv = tile_load_value<IdxT>(idx);
  if constexpr (Masked)
    tile_store_value<T>(dst, mask[0] ? src[iv] : T(0));
  else
    tile_store_value<T>(dst, src[iv]);
}

// VLEN=1 tile_scatter: ``dst`` stays a pointer; ``src``/``idx`` may be
// scalars.
template <typename T, typename IdxT, int VLEN, bool Masked, typename Src, typename Idx>
inline std::enable_if_t<VLEN == 1, void> tile_scatter(T* __restrict__ dst, Src&& src, Idx&& idx,
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
