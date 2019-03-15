#pragma once

#include "dace/xilinx/vec.h"

namespace dace {

// Class that wraps both an input and an output pointer, as these are generated
// as separate interfaces in HLS, but should be seen as a single pointer from
// the point of view of dataflow modules
template <typename T, unsigned vector_length>
class ArrayInterface {

 public:
 
  ArrayInterface(vec<T, vector_length> const *ptr_in,
                 vec<T, vector_length> *ptr_out)
      : ptr_in_(ptr_in), ptr_out_(ptr_out) {
    #pragma HLS INLINE    
  }

  vec<T, vector_length> const *ptr_in() const {
    #pragma HLS INLINE
#ifndef DACE_SYNTHESIS
    if (ptr_in_ == nullptr) {
      throw std::runtime_error("Accessed nullptr in ArrayInterface");
    }
#endif
    return ptr_in_;
  }

  vec<T, vector_length> *ptr_out() const {
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

  friend ArrayInterface<T, vector_length> operator+(
      ArrayInterface<T, vector_length> const &arr, size_t offset) {
    #pragma HLS INLINE
    return ArrayInterface<T, vector_length>(arr.ptr_in_ + offset,
                                            arr.ptr_out_ + offset);
  }

 private:
  vec<T, vector_length> const *ptr_in_;
  vec<T, vector_length> *ptr_out_;
};

} // End namespace dace
