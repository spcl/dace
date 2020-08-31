// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include "dace/xilinx/vec.h"

namespace dace {

// Class that wraps both an input and an output pointer, as these are generated
// as separate interfaces in HLS, but should be seen as a single pointer from
// the point of view of dataflow modules
template <typename T>
class ArrayInterface {
 public:
  ArrayInterface(T const *ptr_in, T *ptr_out)
      : ptr_in_(ptr_in), ptr_out_(ptr_out){
    #pragma HLS INLINE
  }

  T const *ptr_in() const {
    #pragma HLS INLINE
#ifndef DACE_SYNTHESIS
    if (ptr_in_ == nullptr) {
      throw std::runtime_error("Accessed nullptr in ArrayInterface");
    }
#endif
    return ptr_in_;
  }

  T *ptr_out() const {
    #pragma HLS INLINE
#ifndef DACE_SYNTHESIS
    if (ptr_out_ == nullptr) {
      throw std::runtime_error("Accessed nullptr in ArrayInterface");
    }
#endif
    return ptr_out_;
  }

  T const &operator[](size_t i) const {
    #pragma HLS INLINE
    return ptr_in_[i];
  }

  friend ArrayInterface<T> operator+(ArrayInterface<T> const &arr,
                                     size_t offset) {
    #pragma HLS INLINE
    return ArrayInterface<T>(arr.ptr_in_ + offset, arr.ptr_out_ + offset);
  }

 private:
  T const *ptr_in_;
  T *ptr_out_;
};

}  // End namespace dace
