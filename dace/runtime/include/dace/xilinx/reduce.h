// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "hlslib/xilinx/Operators.h"
#include "hlslib/xilinx/TreeReduce.h"

#include "dace/types.h"
#include "dace/xilinx/access.h"

namespace dace {

////////////////////////////////////////////////////////////////////////////////
// Conversion from DACE reduction types to hlslib types
////////////////////////////////////////////////////////////////////////////////

template <ReductionType rt>
struct ConvertReduction;

template <>
struct ConvertReduction<ReductionType::Min> {
  template <typename T>
  using Operator = hlslib::op::Min<T>;
};

template <>
struct ConvertReduction<ReductionType::Max> {
  template <typename T>
  using Operator = hlslib::op::Max<T>;
};

template <>
struct ConvertReduction<ReductionType::Sum> {
  template <typename T>
  using Operator = hlslib::op::Sum<T>;
};

template <>
struct ConvertReduction<ReductionType::Product> {
  template <typename T>
  using Operator = hlslib::op::Product<T>;
};

template <>
struct ConvertReduction<ReductionType::Logical_And> {
  template <typename T>
  using Operator = hlslib::op::And<T>;
};

////////////////////////////////////////////////////////////////////////////////
// Helper functions/template implementation
// (Actual implementation is at the bottom of the file.)
////////////////////////////////////////////////////////////////////////////////

namespace {

template <typename T, typename = void>
struct IsRandomAccess {
  static constexpr bool value = false;
};

template <typename T>
struct IsRandomAccess<
    T, typename std::enable_if<!std::is_same<
           typename std::iterator_traits<T>::value_type, void>::value>::type> {
  static constexpr bool value = true;
};

// Vector to a scalar, call tree reduction
template <typename T, unsigned W_in, unsigned W_out, class Functor,
          typename T_in, typename T_out>
typename std::enable_if<(IsRandomAccess<T_in>::value &&
                         !IsRandomAccess<T_out>::value && W_out - W_in < 0),
                        T_out>::type
ReduceImpl(T_in &&a, T_out &&b) {
#pragma HLS INLINE
  static_assert(W_out != 1,
                "Vector reduction only supported for output length 1.");
  const auto a_reduced = hlslib::TreeReduce<T, Functor, W_in>(a);
  return Functor::Apply(a_reduced, b[0]);
}

// Vector to a scalar wrapped in a vector type, call tree reduction
template <typename T, unsigned W_in, unsigned W_out, class Functor,
          typename T_in, typename T_out>
typename std::enable_if<(IsRandomAccess<T_in>::value &&
                         IsRandomAccess<T_out>::value && W_out - W_in < 0),
                        T_out>::type
ReduceImpl(T_in &&a, T_out &&b) {
#pragma HLS INLINE
  static_assert(W_out != 1,
                "Vector reduction only supported for output length 1.");
  const auto a_reduced = hlslib::TreeReduce<T, Functor, W_in>(a);
  typename std::remove_reference<T_out>::type result;
  result[0] = Functor::Apply(a_reduced, b[0]);
  return result;
}

// Between two scalars
template <typename T, unsigned W_in, unsigned W_out, class Functor,
          typename T_in, typename T_out>
typename std::enable_if<(!IsRandomAccess<T_in>::value &&
                         !IsRandomAccess<T_out>::value && W_in == 1 &&
                         W_out == 1),
                        typename std::remove_reference<T_out>::type>::type
ReduceImpl(T_in &&a, T_out &&b) {
  #pragma HLS INLINE
  return Functor::Apply(std::forward<T_in>(a), std::forward<T_out>(b));
}

// Between two scalars wrapped in vector types
template <typename T, unsigned W_in, unsigned W_out, class Functor,
          typename T_in, typename T_out>
typename std::enable_if<(IsRandomAccess<T_in>::value &&
                         IsRandomAccess<T_out>::value && W_in == 1 &&
                         W_out == 1),
                        T_out>::type
ReduceImpl(T_in &&a, T_out &&b) {
  #pragma HLS INLINE
  typename std::remove_reference<T_out>::type result;
  result[0] = Functor::Apply(a[0], b[0]);
  return result;
}

// Vector-to-vector, apply the reduction on every index
template <typename T, unsigned W_in, unsigned W_out, class Functor,
          typename T_in, typename T_out>
typename std::enable_if<(IsRandomAccess<T_in>() && IsRandomAccess<T_out>() &&
                         W_in > 1 && W_out > 1),
                        T_out>::type
ReduceImpl(T_in &&a, T_out &&b) {
  #pragma HLS INLINE
  return hlslib::op::Wide<Functor, T_out, W_out>(std::forward<T_in>(a),
                                                 std::forward<T_out>(b));
}

}  // End anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// Function exposed to DACE
////////////////////////////////////////////////////////////////////////////////

template <typename T, unsigned W_in, unsigned W_out, class Functor,
          typename T_in, typename T_out>
T Reduce(T_in &&a, T_out &&b) {
  #pragma HLS INLINE
  static_assert(W_out <= W_in,
                "Output vector length must be shorter or identical to input "
                "vector length.");
  return ReduceImpl<T, W_in, W_out, Functor>(Read<T, W_in>(a),
                                             Read<T, W_out>(b));
}

template <ReductionType reduction_type, typename T>
struct xilinx_wcr_fixed {
  static inline T reduce(T *ptr, const T &value) {
    #pragma HLS INLINE
    using Functor =
        typename ConvertReduction<reduction_type>::template Operator<T>;
    T old_val = *ptr;
    *ptr = Reduce<T, 1, 1, Functor>(old_val, value);
    return old_val;
  }
};

// Specialization for vector types
template <ReductionType reduction_type, typename T, unsigned int veclen>
struct xilinx_wcr_fixed_vec {
  static inline vec<T, veclen> reduce(vec<T, veclen> *ptr,
                                      const vec<T, veclen> &value) {
    #pragma HLS INLINE
    using Functor =
        typename ConvertReduction<reduction_type>::template Operator<T>;
    vec<T, veclen> old_val = *ptr;
    *ptr = Reduce<T, veclen, veclen, Functor>(old_val, value);
    return old_val;
  }
};

}  // End namespace dace
