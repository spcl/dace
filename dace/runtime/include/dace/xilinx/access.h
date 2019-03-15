#pragma once

#include "dace/xilinx/array_interface.h"
#include "dace/xilinx/vec.h"
#include "dace/xilinx/stream.h"

namespace dace {

template <typename T, unsigned vector_length>
vec<T, vector_length> Read(ArrayInterface<T, vector_length> const &interface) {
  #pragma HLS INLINE
  return *interface.ptr_in();
}

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

template <typename T, unsigned vector_length, unsigned capacity>
vec<T, vector_length> Read(
    StreamView<T, vector_length, capacity> &stream_view) {
  #pragma HLS INLINE
  return stream_view.pop();
}

template <typename T, unsigned vector_length>
void Write(ArrayInterface<T, vector_length> &interface,
           vec<T, vector_length> const &value) {
  #pragma HLS INLINE
  *interface.ptr_out() = value;
}

template <typename T, unsigned vector_length>
void Write(ArrayInterface<T, vector_length> interface,
           vec<T, vector_length> const &value) {
  #pragma HLS INLINE
  *interface.ptr_out() = value;
}

template <typename T, unsigned vector_length>
void Write(vec<T, vector_length> *ptr, vec<T, vector_length> const &value) {
  #pragma HLS INLINE
  *ptr = value;
}

template <typename T, unsigned vector_length, unsigned capacity>
void Write(vec<T, vector_length> *ptr,
           StreamView<T, vector_length, capacity> &stream) {
  #pragma HLS INLINE
  *ptr = stream;
}

template <typename T, unsigned vector_length>
void Write(vec<T, vector_length> &ref, vec<T, vector_length> const &value) {
  #pragma HLS INLINE
  ref = value;
}

template <typename T, unsigned vector_length, unsigned capacity>
void Write(StreamView<T, vector_length, capacity> &stream_view,
           vec<T, vector_length> const &value) {
  #pragma HLS INLINE
  stream_view.push(value);
}

} // End namespace dace
