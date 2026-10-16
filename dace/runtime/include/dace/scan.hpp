// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Header-only prefix-scan routines for the DaCe Scan library node's parallel CPU
// expansion (+, *, min, max; inclusive and exclusive, plus a seeded inclusive form).
// Floating-point association is block-wise and NOT reproducible across thread
// counts; use the sequential expansion instead when reproducibility matters.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>

#include "types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__has_include)
#if __has_include(<unistd.h>)
#include <unistd.h>
#endif
#endif

// Assumes a non-nested context (not checked at runtime); calling from inside an
// existing omp parallel region still gives correct values, just no speedup.

namespace dace {
namespace scan {

namespace detail {

#ifdef _OPENMP
// OpenMP has no built-in reduction for a class type; complex needs a user-defined one
// (declared here, not reused from reduction.h, since a UDR is found by unqualified
// lookup from the point of use). Only + and *; see is_ordered below for min/max.
#pragma omp declare reduction(+ : complex64 : omp_out += omp_in) initializer(omp_priv = complex64(0))
#pragma omp declare reduction(+ : complex128 : omp_out += omp_in) initializer(omp_priv = complex128(0))
#pragma omp declare reduction(* : complex64 : omp_out *= omp_in) initializer(omp_priv = complex64(1))
#pragma omp declare reduction(* : complex128 : omp_out *= omp_in) initializer(omp_priv = complex128(1))
#endif

// Largest team the blocked scan will ask for; bounds the stack array of block totals.
constexpr int MAX_TEAM = 256;

// One thread's block total, padded to a cache line. K > 1 carries one total per scan chain.
template <typename T, int K = 1>
struct alignas(64) TeamSlot {
  T v[K];
};

// Fallback tile size when the cache hierarchy cannot be queried.
constexpr long TILE_BYTES = 1L << 17;

// One cache level's size in bytes, or 0 if unknown. Queried once and cached.
inline long cache_bytes(int level) {
#if defined(_SC_LEVEL2_CACHE_SIZE) && defined(_SC_LEVEL1_DCACHE_SIZE) && defined(_SC_LEVEL3_CACHE_SIZE)
  static const long sizes[3] = {sysconf(_SC_LEVEL1_DCACHE_SIZE), sysconf(_SC_LEVEL2_CACHE_SIZE),
                                sysconf(_SC_LEVEL3_CACHE_SIZE)};
  const long v = (level >= 1 && level <= 3) ? sizes[level - 1] : 0;
  return v > 0 ? v : 0;
#else
  (void)level;
  return 0;
#endif
}

// Bytes per thread per tile, derived from this machine's L2 (clamped to [32 KB, 512 KB]) so the
// scan pass re-reads a block that is still in cache after the reduce pass wrote it.
inline long tile_bytes() {
  const long l2 = cache_bytes(2);
  if (l2 <= 0) return TILE_BYTES;
  const long derived = l2 / 8;
  if (derived < (1L << 15)) return 1L << 15;
  if (derived > (1L << 19)) return 1L << 19;
  return derived;
}

// Elements one thread takes per tile. A one-thread team takes the whole range in a single
// tile, since there are no block totals to exchange.
inline long block_span(long n, long team, long elem_bytes) {
  if (team <= 1) return n;
  const long budget = tile_bytes();
  const long cap = (elem_bytes > 0 && budget / elem_bytes > 0) ? budget / elem_bytes : 1;
  const long even = (n + team - 1) / team;
  return (cap < even) ? cap : even;
}

/// Team the blocked scan asks for, clamped to the block-total array.
inline int team_size() {
#ifdef _OPENMP
  const int m = omp_get_max_threads();
  return (m < MAX_TEAM) ? m : MAX_TEAM;
#else
  return 1;
#endif
}

// Size of, and this thread's rank in, the enclosing region; compiles to 1 / 0 without OpenMP.
inline long team_count() {
#ifdef _OPENMP
  return static_cast<long>(omp_get_num_threads());
#else
  return 1;
#endif
}

inline long team_rank() {
#ifdef _OPENMP
  return static_cast<long>(omp_get_thread_num());
#else
  return 0;
#endif
}

// Complex has no ordering: min/max are refused by name here, not by a missing operator<.
template <typename T>
struct is_ordered
    : std::integral_constant<bool, !std::is_same<T, complex64>::value && !std::is_same<T, complex128>::value> {};

// The min/max neutral elements, matching OpenMP's reduction(min:)/reduction(max:) init.
template <typename T>
constexpr T min_identity() {
  static_assert(is_ordered<T>::value,
                "dace::scan: min/max needs an ordered element type; "
                "complex64 / complex128 have none -- only sum and product are defined for them.");
  return std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
}

template <typename T>
constexpr T max_identity() {
  static_assert(is_ordered<T>::value,
                "dace::scan: min/max needs an ordered element type; "
                "complex64 / complex128 have none -- only sum and product are defined for them.");
  // T(...), not a bare -infinity(): keeps both ternary arms the same type.
  return std::numeric_limits<T>::has_infinity ? T(-std::numeric_limits<T>::infinity())
                                              : std::numeric_limits<T>::lowest();
}

// Per-block passes: one function per op, since an OpenMP reduction identifier is not a
// template parameter. `A` is named explicitly (not deduced from `It`) because a widening
// scan folds a block into a value the input type cannot hold.
template <typename A, typename It>
inline A fold_sum(It f, long lo, long hi) {
  A s = 0;
#pragma omp simd reduction(+ : s)
  for (long i = lo; i < hi; ++i) s = s + f[i];
  return s;
}

template <typename A, typename It>
inline A fold_product(It f, long lo, long hi) {
  A s = 1;
#pragma omp simd reduction(* : s)
  for (long i = lo; i < hi; ++i) s = s * f[i];
  return s;
}

template <typename A, typename It>
inline A fold_min(It f, long lo, long hi) {
  A s = min_identity<A>();
#pragma omp simd reduction(min : s)
  for (long i = lo; i < hi; ++i) s = std::min<A>(s, static_cast<A>(f[i]));
  return s;
}

template <typename A, typename It>
inline A fold_max(It f, long lo, long hi) {
  A s = max_identity<A>();
#pragma omp simd reduction(max : s)
  for (long i = lo; i < hi; ++i) s = std::max<A>(s, static_cast<A>(f[i]));
  return s;
}

template <typename It, typename OutIt, typename T>
inline void scan_incl_sum(It f, OutIt o, long lo, long hi, T seed) {
  T acc = seed;
#pragma omp simd reduction(inscan, + : acc)
  for (long i = lo; i < hi; ++i) {
    acc = acc + f[i];
#pragma omp scan inclusive(acc)
    o[i] = acc;
  }
}

template <typename It, typename OutIt, typename T>
inline void scan_incl_product(It f, OutIt o, long lo, long hi, T seed) {
  T acc = seed;
#pragma omp simd reduction(inscan, * : acc)
  for (long i = lo; i < hi; ++i) {
    acc = acc * f[i];
#pragma omp scan inclusive(acc)
    o[i] = acc;
  }
}

template <typename It, typename OutIt, typename T>
inline void scan_incl_min(It f, OutIt o, long lo, long hi, T seed) {
  T acc = seed;
#pragma omp simd reduction(inscan, min : acc)
  for (long i = lo; i < hi; ++i) {
    acc = std::min<T>(acc, f[i]);
#pragma omp scan inclusive(acc)
    o[i] = acc;
  }
}

template <typename It, typename OutIt, typename T>
inline void scan_incl_max(It f, OutIt o, long lo, long hi, T seed) {
  T acc = seed;
#pragma omp simd reduction(inscan, max : acc)
  for (long i = lo; i < hi; ++i) {
    acc = std::max<T>(acc, f[i]);
#pragma omp scan inclusive(acc)
    o[i] = acc;
  }
}

template <typename It, typename OutIt, typename T>
inline void scan_excl_sum(It f, OutIt o, long lo, long hi, T seed) {
  T acc = seed;
#pragma omp simd reduction(inscan, + : acc)
  for (long i = lo; i < hi; ++i) {
    o[i] = acc;
#pragma omp scan exclusive(acc)
    acc = acc + f[i];
  }
}

template <typename It, typename OutIt, typename T>
inline void scan_excl_product(It f, OutIt o, long lo, long hi, T seed) {
  T acc = seed;
#pragma omp simd reduction(inscan, * : acc)
  for (long i = lo; i < hi; ++i) {
    o[i] = acc;
#pragma omp scan exclusive(acc)
    acc = acc * f[i];
  }
}

template <typename It, typename OutIt, typename T>
inline void scan_excl_min(It f, OutIt o, long lo, long hi, T seed) {
  T acc = seed;
#pragma omp simd reduction(inscan, min : acc)
  for (long i = lo; i < hi; ++i) {
    o[i] = acc;
#pragma omp scan exclusive(acc)
    acc = std::min<T>(acc, f[i]);
  }
}

template <typename It, typename OutIt, typename T>
inline void scan_excl_max(It f, OutIt o, long lo, long hi, T seed) {
  T acc = seed;
#pragma omp simd reduction(inscan, max : acc)
  for (long i = lo; i < hi; ++i) {
    o[i] = acc;
#pragma omp scan exclusive(acc)
    acc = std::max<T>(acc, f[i]);
  }
}

// Three-phase blocked scan over one parallel region. `reduce(lo, hi)` folds a block,
// `scan(lo, hi, off)` writes out[lo:hi] seeded with off, `combine` folds two block totals.
template <typename T, typename Reduce, typename Scan, typename Combine>
inline void blocked_scan(long n, long elem_bytes, T seed, Reduce reduce, Scan scan, Combine combine) {
  if (n <= 0) return;
#ifdef _OPENMP
  const int want = team_size();
  // No size test here: a scan too small for a team is expanded to the sequential shape
  // instead, before it ever reaches this function.
  if (want > 1) {
    TeamSlot<T> totals[MAX_TEAM];
#pragma omp parallel num_threads(want)
    {
      const long team = static_cast<long>(omp_get_num_threads());
      const long me = static_cast<long>(omp_get_thread_num());
      const long per = block_span(n, team, elem_bytes);
      const long tile = per * team;
      T carry = seed;
      for (long base = 0; base < n; base += tile) {
        const long end = (base + tile < n) ? base + tile : n;
        const long lo = (base + me * per < end) ? base + me * per : end;
        const long hi = (lo + per < end) ? lo + per : end;
        if (team > 1) {
          totals[me].v[0] = reduce(lo, hi);
#pragma omp barrier
          T off = carry, all = carry;
          for (long q = 0; q < team; ++q) {
            if (q == me) off = all;
            all = combine(all, totals[q].v[0]);
          }
          scan(lo, hi, off);
          carry = all;
#pragma omp barrier
        } else {
          scan(lo, hi, carry);
        }
      }
    }
    return;
  }
#endif
  scan(0, n, seed);
}

// The affine map x -> a*x + b: the carry of a first-order linear recurrence. A plain scan
// carries a value; this carries a function, which is what makes out[k] = c[k]*out[k-1] + d[k]
// block at all -- composition is associative, so the three-phase shape above still applies.
template <typename E>
struct affine_map {
  E a;
  E b;
};

// Compose two affine maps: y applied after x.
template <typename E>
inline affine_map<E> affine_compose(const affine_map<E>& x, const affine_map<E>& y) {
  return affine_map<E>{y.a * x.a, y.a * x.b + y.b};
}

// Fold one block's coefficients and deltas into a single affine map. The seed carries a == 0,
// so the running product m.a never spans more than one block and cannot overflow.
template <typename E, typename CIt, typename DIt>
inline affine_map<E> fold_affine(CIt c, DIt d, long lo, long hi) {
  affine_map<E> m{static_cast<E>(1), static_cast<E>(0)};
  for (long k = lo; k < hi; ++k) {
    const E ck = static_cast<E>(c[k]);
    m.b = ck * m.b + static_cast<E>(d[k]);
    m.a = ck * m.a;
  }
  return m;
}

// Write out[lo:hi] for the recurrence entered with the composed map off. No simd inscan here,
// unlike the four scalar ops: each element's coefficient is only known at that element, so the
// chain is a genuine dependent multiply-add; the parallelism is the blocking alone.
template <typename E, typename CIt, typename DIt, typename OutIt>
inline void scan_incl_affine(CIt c, DIt d, OutIt o, long lo, long hi, affine_map<E> off) {
  E acc = off.b;
  for (long k = lo; k < hi; ++k) {
    acc = static_cast<E>(c[k]) * acc + static_cast<E>(d[k]);
    o[k] = acc;
  }
}

}  // namespace detail

// INCLUSIVE
// The seeded overload is ``out[k] = seed OP in[0] OP ... OP in[k]``; the plain one
// seeds with the op's identity, which for min/max is the neutral element rather
// than ``in[0]`` (same result, and it keeps an empty range from reading ``in``).

template <typename It, typename OutIt, typename T>
inline void inclusive_sum(It first, It last, OutIt out_first, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  detail::blocked_scan<E>(
      static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
      [first](long lo, long hi) { return detail::fold_sum<E>(first, lo, hi); },
      [first, out_first](long lo, long hi, E off) { detail::scan_incl_sum(first, out_first, lo, hi, off); },
      [](E a, E b) { return a + b; });
}

template <typename It, typename OutIt>
inline void inclusive_sum(It first, It last, OutIt out_first) {
  inclusive_sum(first, last, out_first, typename std::iterator_traits<OutIt>::value_type(0));
}

template <typename It, typename OutIt, typename T>
inline void inclusive_product(It first, It last, OutIt out_first, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  detail::blocked_scan<E>(
      static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
      [first](long lo, long hi) { return detail::fold_product<E>(first, lo, hi); },
      [first, out_first](long lo, long hi, E off) { detail::scan_incl_product(first, out_first, lo, hi, off); },
      [](E a, E b) { return a * b; });
}

template <typename It, typename OutIt>
inline void inclusive_product(It first, It last, OutIt out_first) {
  inclusive_product(first, last, out_first, typename std::iterator_traits<OutIt>::value_type(1));
}

template <typename It, typename OutIt, typename T>
inline void inclusive_min(It first, It last, OutIt out_first, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  detail::blocked_scan<E>(
      static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
      [first](long lo, long hi) { return detail::fold_min<E>(first, lo, hi); },
      [first, out_first](long lo, long hi, E off) { detail::scan_incl_min(first, out_first, lo, hi, off); },
      [](E a, E b) { return std::min<E>(a, b); });
}

template <typename It, typename OutIt>
inline void inclusive_min(It first, It last, OutIt out_first) {
  inclusive_min(first, last, out_first, detail::min_identity<typename std::iterator_traits<OutIt>::value_type>());
}

template <typename It, typename OutIt, typename T>
inline void inclusive_max(It first, It last, OutIt out_first, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  detail::blocked_scan<E>(
      static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
      [first](long lo, long hi) { return detail::fold_max<E>(first, lo, hi); },
      [first, out_first](long lo, long hi, E off) { detail::scan_incl_max(first, out_first, lo, hi, off); },
      [](E a, E b) { return std::max<E>(a, b); });
}

template <typename It, typename OutIt>
inline void inclusive_max(It first, It last, OutIt out_first) {
  inclusive_max(first, last, out_first, detail::max_identity<typename std::iterator_traits<OutIt>::value_type>());
}

// EXCLUSIVE
// out[0] = seed; out[i] = seed OP in[0] OP ... OP in[i-1]

template <typename It, typename OutIt, typename T>
inline void exclusive_sum(It first, It last, OutIt out_first, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  detail::blocked_scan<E>(
      static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
      [first](long lo, long hi) { return detail::fold_sum<E>(first, lo, hi); },
      [first, out_first](long lo, long hi, E off) { detail::scan_excl_sum(first, out_first, lo, hi, off); },
      [](E a, E b) { return a + b; });
}

template <typename It, typename OutIt, typename T>
inline void exclusive_product(It first, It last, OutIt out_first, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  detail::blocked_scan<E>(
      static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
      [first](long lo, long hi) { return detail::fold_product<E>(first, lo, hi); },
      [first, out_first](long lo, long hi, E off) { detail::scan_excl_product(first, out_first, lo, hi, off); },
      [](E a, E b) { return a * b; });
}

template <typename It, typename OutIt, typename T>
inline void exclusive_min(It first, It last, OutIt out_first, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  detail::blocked_scan<E>(
      static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
      [first](long lo, long hi) { return detail::fold_min<E>(first, lo, hi); },
      [first, out_first](long lo, long hi, E off) { detail::scan_excl_min(first, out_first, lo, hi, off); },
      [](E a, E b) { return std::min<E>(a, b); });
}

template <typename It, typename OutIt, typename T>
inline void exclusive_max(It first, It last, OutIt out_first, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  detail::blocked_scan<E>(
      static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
      [first](long lo, long hi) { return detail::fold_max<E>(first, lo, hi); },
      [first, out_first](long lo, long hi, E off) { detail::scan_excl_max(first, out_first, lo, hi, off); },
      [](E a, E b) { return std::max<E>(a, b); });
}

// Strided inclusive: out[i + s] = out[i] OP in[i] for stride s > 0. Each residue class
// k in [0, s) is an independent inclusive scan; s == 1 matches the contiguous routines
// above. One parallel region splits residue classes across the team (never one region
// per class). Threads split whole classes, so results do not move with thread count --
// unlike the unit-stride inscan routines above.

namespace detail {

// Row-major sweep carrying S accumulators, S a compile-time constant so it unrolls into registers.
template <long S, typename It, typename OutIt, typename Seed, typename Op>
inline void strided_scan_unrolled(It first, OutIt out, long n, Seed seed, Op op) {
  using T = typename std::iterator_traits<It>::value_type;
  T acc[S] = {};
  const long head = (S < n) ? S : n;
  for (long k = 0; k < head; ++k) {
    acc[k] = seed(first[k]);
    out[k] = acc[k];
  }
  long j = S;
  for (; j + S <= n; j += S) {
    for (long k = 0; k < S; ++k) {
      acc[k] = op(acc[k], first[j + k]);
      out[j + k] = acc[k];
    }
  }
  for (long k = 0; j + k < n; ++k) {
    acc[k] = op(acc[k], first[j + k]);
    out[j + k] = acc[k];
  }
}

// Wide strides: the carry lives in out itself, s elements back. Restricted to residue
// classes [k0, k1) so a thread can own a slice of the classes.
template <typename It, typename OutIt, typename Seed, typename Op>
inline void strided_scan_block(It first, OutIt out, long n, long s, long k0, long k1, Seed seed, Op op) {
  if (k0 >= k1) return;
  const long head = (k1 < n) ? k1 : n;
  for (long j = k0; j < head; ++j) out[j] = seed(first[j]);
  for (long base = s; base < n; base += s) {
    const long hi = (base + k1 < n) ? base + k1 : n;
    for (long j = base + k0; j < hi; ++j) out[j] = op(out[j - s], first[j]);
  }
}

// Residue classes per cache line: a thread block must be a multiple of this or two
// threads write the same line on every row.
template <typename T>
constexpr long classes_per_line() {
  return (64 / static_cast<long>(sizeof(T))) > 1 ? (64 / static_cast<long>(sizeof(T))) : 1;
}

// Below this element count the fork costs more than the team saves.
constexpr long PARALLEL_MIN_ELEMENTS = 1L << 16;

// Minimum bytes of contiguous classes per thread.
constexpr long PARALLEL_MIN_BLOCK_BYTES = 4096;

// seed opens a residue class (identity OP first element); op extends it.
template <typename It, typename OutIt, typename Seed, typename Op>
inline void strided_scan(It first, OutIt out, long n, long s, Seed seed, Op op) {
  if (s <= 0) std::abort();
#ifdef _OPENMP
  using T = typename std::iterator_traits<It>::value_type;
  constexpr long line = classes_per_line<T>();
  constexpr long min_block = PARALLEL_MIN_BLOCK_BYTES / static_cast<long>(sizeof(T));
  const long threads = static_cast<long>(omp_get_max_threads());
  if (threads > 1 && n >= PARALLEL_MIN_ELEMENTS && s >= min_block * threads) {
#pragma omp parallel num_threads(static_cast<int>(threads))
    {
      const long team = static_cast<long>(omp_get_num_threads());
      const long mine = static_cast<long>(omp_get_thread_num());
      const long lines = (s + line - 1) / line;
      const long per = ((lines + team - 1) / team) * line;
      const long k0 = (mine * per < s) ? mine * per : s;
      const long k1 = (k0 + per < s) ? k0 + per : s;
      strided_scan_block(first, out, n, s, k0, k1, seed, op);
    }
    return;
  }
#endif
  switch (s) {
    case 1:
      strided_scan_unrolled<1>(first, out, n, seed, op);
      return;
    case 2:
      strided_scan_unrolled<2>(first, out, n, seed, op);
      return;
    case 3:
      strided_scan_unrolled<3>(first, out, n, seed, op);
      return;
    case 4:
      strided_scan_unrolled<4>(first, out, n, seed, op);
      return;
    case 5:
      strided_scan_unrolled<5>(first, out, n, seed, op);
      return;
    case 6:
      strided_scan_unrolled<6>(first, out, n, seed, op);
      return;
    case 7:
      strided_scan_unrolled<7>(first, out, n, seed, op);
      return;
    case 8:
      strided_scan_unrolled<8>(first, out, n, seed, op);
      return;
    default:
      strided_scan_block(first, out, n, s, 0, s, seed, op);
      return;
  }
}

}  // namespace detail

template <typename It, typename OutIt>
inline void strided_inclusive_sum(It first, OutIt out, long n, long s) {
  using T = typename std::iterator_traits<It>::value_type;
  detail::strided_scan(
      first, out, n, s, [](const T& x) { return T(0) + x; }, [](const T& a, const T& b) { return a + b; });
}

template <typename It, typename OutIt>
inline void strided_inclusive_product(It first, OutIt out, long n, long s) {
  using T = typename std::iterator_traits<It>::value_type;
  detail::strided_scan(
      first, out, n, s, [](const T& x) { return T(1) * x; }, [](const T& a, const T& b) { return a * b; });
}

template <typename It, typename OutIt>
inline void strided_inclusive_min(It first, OutIt out, long n, long s) {
  using T = typename std::iterator_traits<It>::value_type;
  detail::strided_scan(
      first, out, n, s, [](const T& x) { return x; }, [](const T& a, const T& b) { return std::min<T>(a, b); });
}

template <typename It, typename OutIt>
inline void strided_inclusive_max(It first, OutIt out, long n, long s) {
  using T = typename std::iterator_traits<It>::value_type;
  detail::strided_scan(
      first, out, n, s, [](const T& x) { return x; }, [](const T& a, const T& b) { return std::max<T>(a, b); });
}

// First-order linear recurrence: out[k] = c[k]*out[k-1] + d[k] over k in [0, n), entered
// with out[-1] = seed. Reuses blocked_scan with T = affine_map<E>. Reproduces the sequential
// result exactly within a block; moves with OMP_NUM_THREADS only through the block carries.

template <typename CIt, typename DIt, typename OutIt, typename T>
inline void inclusive_affine(CIt coef, DIt delta, OutIt out_first, long n, T seed) {
  using E = typename std::iterator_traits<OutIt>::value_type;
  using M = detail::affine_map<E>;
  detail::blocked_scan<M>(
      n, static_cast<long>(3 * sizeof(E)), M{static_cast<E>(0), static_cast<E>(seed)},
      [coef, delta](long lo, long hi) { return detail::fold_affine<E>(coef, delta, lo, hi); },
      [coef, delta, out_first](long lo, long hi, M off) {
        detail::scan_incl_affine<E>(coef, delta, out_first, lo, hi, off);
      },
      [](M x, M y) { return detail::affine_compose<E>(x, y); });
}

// Strided affine: x[i] = c[i] * x[i - stride] + d[i] is `stride` independent unit-stride
// affine scans, one per residue class of the index mod stride. Reuses inclusive_affine
// unchanged; stride == 1 is the plain contiguous scan.

namespace detail {

// Random-access view of every stride-th element from origin, so a residue class looks
// contiguous to fold_affine / scan_incl_affine.
template <typename It>
struct strided_view {
  It base;
  long origin;
  long stride;

  inline auto operator[](long i) const -> decltype(base[0]) { return base[origin + i * stride]; }
};

template <typename It>
inline strided_view<It> strided(It base, long origin, long stride) {
  return strided_view<It>{base, origin, stride};
}

/// Seeds for a strided scan nothing seeded: every class enters at the monoid's identity.
///
/// A view rather than an allocated array of zeros, because the count is the stride and the
/// caller would otherwise allocate one element per residue class to say "nothing".
template <typename E>
struct zero_seed_view {
  inline E operator[](long) const { return static_cast<E>(0); }
};

template <typename E>
inline zero_seed_view<E> zero_seeds() {
  return zero_seed_view<E>{};
}

}  // namespace detail

/// ``out[k] = coef[k] * out[k - stride] + delta[k]`` over ``n`` elements, seeded per class.
///
/// :param seeds: one seed per residue class, ``seeds[r]`` entering class ``r``. The caller holds
///               them because they are the carrier's pre-loop values, which only it can read.
template <typename CIt, typename DIt, typename OutIt, typename SIt>
inline void inclusive_affine_strided(CIt coef, DIt delta, OutIt out_first, long n, long stride, SIt seeds) {
  if (stride <= 1) {
    inclusive_affine(coef, delta, out_first, n, seeds[0]);
    return;
  }
  const long classes = stride < n ? stride : n;
// One class per thread, each running its own sequential recurrence. Requesting the team here
// and not inside is deliberate: an inner region would be nested and serialised anyway, and
// this way the classes are what the schedule balances.
#pragma omp parallel for schedule(static)
  for (long r = 0; r < classes; ++r) {
    const long len = (n - r + stride - 1) / stride;
    detail::affine_map<typename std::iterator_traits<OutIt>::value_type> off{
        static_cast<typename std::iterator_traits<OutIt>::value_type>(0),
        static_cast<typename std::iterator_traits<OutIt>::value_type>(seeds[r])};
    detail::scan_incl_affine(detail::strided(coef, r, stride), detail::strided(delta, r, stride),
                             detail::strided(out_first, r, stride), 0L, len, off);
  }
}

}  // namespace scan
}  // namespace dace
