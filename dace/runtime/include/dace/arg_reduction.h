// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// argmax / argmin as OpenMP reductions over a (value, index) pair. Ties keep the LOWER index, so a parallel
// answer is the sequential first-occurrence one however the range is split. A complex value orders
// lexicographically, real part first, as NumPy's argmax does. CPF carries the same definitions
// (dace.cpf_lowering ARG_REDUCTION_CPP), so a standalone unit reduces through the same combiners.
#pragma once

#include <cstdint>
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
static inline bool arg_greater(const V& a, const V& b) {
  if constexpr (arg_is_complex<V>::value) {
    return a.real() > b.real() || (a.real() == b.real() && a.imag() > b.imag());
  } else {
    return a > b;
  }
}

template <typename V, typename I>
static inline arg_pair<V, I> arg_pair_of(const V& v, I i) {
  return arg_pair<V, I>{v, i};
}

// One thread's scan: a strictly better value replaces the held one, so the first extreme in index order stays.
template <typename V, typename I>
static inline arg_pair<V, I> arg_max_step(const arg_pair<V, I>& held, const V& v, I i) {
  return arg_greater(v, held.__ar_v) ? arg_pair<V, I>{v, i} : held;
}
template <typename V, typename I>
static inline arg_pair<V, I> arg_min_step(const arg_pair<V, I>& held, const V& v, I i) {
  return arg_greater(held.__ar_v, v) ? arg_pair<V, I>{v, i} : held;
}

// Two threads' answers: the better value, or on a tie the lower index.
template <typename V, typename I>
static inline arg_pair<V, I> arg_max_combine(const arg_pair<V, I>& out, const arg_pair<V, I>& in) {
  return (arg_greater(in.__ar_v, out.__ar_v) || (in.__ar_v == out.__ar_v && in.__ar_i < out.__ar_i)) ? in : out;
}
template <typename V, typename I>
static inline arg_pair<V, I> arg_min_combine(const arg_pair<V, I>& out, const arg_pair<V, I>& in) {
  return (arg_greater(out.__ar_v, in.__ar_v) || (in.__ar_v == out.__ar_v && in.__ar_i < out.__ar_i)) ? in : out;
}

}  // namespace dace

// Every pair an ArgReduce lowering names: each scalar and complex value type, indexed by int32 or int64.
#define DACE_ARG_PAIRS(I)                                                                                           \
  dace::arg_pair<float, I>, dace::arg_pair<double, I>, dace::arg_pair<int8_t, I>, dace::arg_pair<int16_t, I>,      \
      dace::arg_pair<int32_t, I>, dace::arg_pair<int64_t, I>, dace::arg_pair<uint8_t, I>,                          \
      dace::arg_pair<uint16_t, I>, dace::arg_pair<uint32_t, I>, dace::arg_pair<uint64_t, I>,                       \
      dace::arg_pair<dace::complex64, I>, dace::arg_pair<dace::complex128, I>

#pragma omp declare reduction(arg_max_pair : DACE_ARG_PAIRS(int32_t), DACE_ARG_PAIRS(int64_t) : \
    omp_out = dace::arg_max_combine(omp_out, omp_in)) initializer(omp_priv = omp_orig)
#pragma omp declare reduction(arg_min_pair : DACE_ARG_PAIRS(int32_t), DACE_ARG_PAIRS(int64_t) : \
    omp_out = dace::arg_min_combine(omp_out, omp_in)) initializer(omp_priv = omp_orig)
