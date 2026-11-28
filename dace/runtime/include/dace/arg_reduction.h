// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// argmax / argmin as OpenMP reductions over a (value, index) pair. Ties keep the LOWER index (the HIGHER one for the
// ``_last`` forms), so a parallel answer is the sequential one however the range is split. A complex value orders
// lexicographically, real part first, as NumPy's argmax does. CPF carries the same definitions
// (dace.cpf_lowering ARG_REDUCTION_CPP), so a standalone unit reduces through the same combiners.
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "types.h"

namespace dace {

template <typename V, typename I>
struct arg_pair {
  V __ar_v;
  I __ar_i;
};

template <typename V, typename = void>
struct arg_is_complex : std::false_type {};
template <typename V>
struct arg_is_complex<V, std::void_t<decltype(std::declval<V>().imag())>> : std::true_type {};

template <typename V>
DACE_HDFI bool arg_greater(const V& a, const V& b) {
  if constexpr (arg_is_complex<V>::value) {
    return a.real() > b.real() || (a.real() == b.real() && a.imag() > b.imag());
  } else {
    return a > b;
  }
}

template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_pair_of(const V& v, I i) {
  return arg_pair<V, I>{v, i};
}

// One thread's scan from a held pair. ``max`` / ``min`` take a strictly better value, so the first extreme in index
// order stays (a strict ``>`` / ``<`` guard); ``max_last`` / ``min_last`` also take an equal one, so the last extreme
// stays (a non-strict ``>=`` / ``<=`` guard). A comparison with NaN is false, so a NaN is never taken and a held NaN
// is never replaced, as in the loop.
template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_max_step(const arg_pair<V, I>& held, const V& v, I i) {
  return arg_greater(v, held.__ar_v) ? arg_pair<V, I>{v, i} : held;
}
template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_min_step(const arg_pair<V, I>& held, const V& v, I i) {
  return arg_greater(held.__ar_v, v) ? arg_pair<V, I>{v, i} : held;
}
template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_max_last_step(const arg_pair<V, I>& held, const V& v, I i) {
  return arg_greater(v, held.__ar_v) || v == held.__ar_v ? arg_pair<V, I>{v, i} : held;
}
template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_min_last_step(const arg_pair<V, I>& held, const V& v, I i) {
  return arg_greater(held.__ar_v, v) || v == held.__ar_v ? arg_pair<V, I>{v, i} : held;
}

// Two threads' answers: the better value, or on a tie the lower index (the higher one when ``Last``). Every lowering
// starts a chunk from the neutral pair or from a non-NaN element, so a NaN pair is the seed at position 0, which the
// loop never replaces: it wins whichever side it is on.
template <bool Max, bool Last, typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_combine(const arg_pair<V, I>& out, const arg_pair<V, I>& in) {
  if (!(out.__ar_v == out.__ar_v) || !(in.__ar_v == in.__ar_v)) return out.__ar_v == out.__ar_v ? in : out;
  const bool better = Max ? arg_greater(in.__ar_v, out.__ar_v) : arg_greater(out.__ar_v, in.__ar_v);
  const bool wins_tie = Last ? in.__ar_i > out.__ar_i : in.__ar_i < out.__ar_i;
  return (better || (in.__ar_v == out.__ar_v && wins_tie)) ? in : out;
}
template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_max_combine(const arg_pair<V, I>& out, const arg_pair<V, I>& in) {
  return arg_combine<true, false>(out, in);
}
template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_min_combine(const arg_pair<V, I>& out, const arg_pair<V, I>& in) {
  return arg_combine<false, false>(out, in);
}
template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_max_last_combine(const arg_pair<V, I>& out, const arg_pair<V, I>& in) {
  return arg_combine<true, true>(out, in);
}
template <typename V, typename I>
DACE_HDFI arg_pair<V, I> arg_min_last_combine(const arg_pair<V, I>& out, const arg_pair<V, I>& in) {
  return arg_combine<false, true>(out, in);
}

// The value no element is worse than, or better than when ``lowest`` is false.
template <typename V>
DACE_HDFI V arg_bound(bool lowest) {
  if constexpr (arg_is_complex<V>::value) {
    const typename V::value_type part = arg_bound<typename V::value_type>(lowest);
    return V(part, part);
  } else if constexpr (std::numeric_limits<V>::has_infinity) {
    return lowest ? -std::numeric_limits<V>::infinity() : std::numeric_limits<V>::infinity();
  } else {
    return lowest ? std::numeric_limits<V>::lowest() : std::numeric_limits<V>::max();
  }
}

// A thread's start: the neutral pair, whose index loses every tie (the lowest index when ties go to the last). Never
// omp_orig: icx 2026.2 at -O2 reuses the value it read for the seed as omp_out, so threads overwrite each other's
// answers.
template <typename V, typename I>
DACE_HDFI void arg_seed(arg_pair<V, I>& priv, bool lowest, bool last = false) {
  priv = arg_pair<V, I>{arg_bound<V>(lowest), last ? std::numeric_limits<I>::lowest() : std::numeric_limits<I>::max()};
}

}  // namespace dace

// Every pair an ArgReduce lowering names: each scalar and complex value type, indexed by int32 or int64.
#define DACE_ARG_PAIRS(I)                                                                                           \
  dace::arg_pair<float, I>, dace::arg_pair<double, I>, dace::arg_pair<int8_t, I>, dace::arg_pair<int16_t, I>,      \
      dace::arg_pair<int32_t, I>, dace::arg_pair<int64_t, I>, dace::arg_pair<uint8_t, I>,                          \
      dace::arg_pair<uint16_t, I>, dace::arg_pair<uint32_t, I>, dace::arg_pair<uint64_t, I>,                       \
      dace::arg_pair<dace::complex64, I>, dace::arg_pair<dace::complex128, I>

#pragma omp declare reduction(arg_max_pair : DACE_ARG_PAIRS(int32_t), DACE_ARG_PAIRS(int64_t) : \
    omp_out = dace::arg_max_combine(omp_out, omp_in)) \
    initializer(dace::arg_seed(omp_priv, true))
#pragma omp declare reduction(arg_min_pair : DACE_ARG_PAIRS(int32_t), DACE_ARG_PAIRS(int64_t) : \
    omp_out = dace::arg_min_combine(omp_out, omp_in)) \
    initializer(dace::arg_seed(omp_priv, false))
#pragma omp declare reduction(arg_max_last_pair : DACE_ARG_PAIRS(int32_t), DACE_ARG_PAIRS(int64_t) : \
    omp_out = dace::arg_max_last_combine(omp_out, omp_in)) \
    initializer(dace::arg_seed(omp_priv, true, true))
#pragma omp declare reduction(arg_min_last_pair : DACE_ARG_PAIRS(int32_t), DACE_ARG_PAIRS(int64_t) : \
    omp_out = dace::arg_min_last_combine(omp_out, omp_in)) \
    initializer(dace::arg_seed(omp_priv, false, true))
