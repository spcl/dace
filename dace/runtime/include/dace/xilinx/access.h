// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include "dace/xilinx/vec.h"
#include "dace/xilinx/stream.h"

namespace dace {

template <typename T, unsigned vector_length>
vec<T, vector_length> Read(vec<T, vector_length> const *ptr) {
  #pragma HLS INLINE
  return *ptr;
}

template <typename T, unsigned vector_length>
vec<T, vector_length> Read(vec<T, vector_length> const &ref) {
  #pragma HLS INLINE
  return ref;
}

template <typename T, unsigned vector_length>
void Write(vec<T, vector_length> *ptr, vec<T, vector_length> const &value) {
  #pragma HLS INLINE
  *ptr = value;
}

template <typename T, unsigned vector_length>
void Write(vec<T, vector_length> &ref, vec<T, vector_length> const &value) {
  #pragma HLS INLINE
  ref = value;
}

template <typename T, unsigned vector_length>
vec<T, vector_length> Pack(T const *const ptr) {
  #pragma HLS INLINE
  vec<T, vector_length> val;
  for (unsigned i = 0; i < vector_length; ++i) {
    #pragma HLS UNROLL
    val[i] = ptr[i];
  }
  return val;
}

template <typename T, unsigned vector_length>
void Unpack(vec<T, vector_length> const &val, T *const ptr) {
  #pragma HLS INLINE
  for (unsigned i = 0; i < vector_length; ++i) {
    #pragma HLS UNROLL
    ptr[i] = val[i];
  }
}

} // End namespace dace
