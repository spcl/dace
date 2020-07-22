#pragma once

#include <utility>
#ifndef DACE_SYNTHESIS
#include <stdexcept>
#include <string>
#endif

#include "dace/types.h"
#include "dace/xilinx/array_interface.h"
#include "dace/xilinx/reduce.h"
#include "dace/xilinx/vec.h"

namespace dace {

namespace {

template <unsigned vector_length>
struct WriteImpl {
  template <typename T>
  static void Write(T *ptr, vec<T, vector_length> const &value) {
    #pragma HLS INLINE
    value.Unpack(ptr);
  }
  template <typename T>
  static void Write(vec<T, vector_length> *ptr,
                    vec<T, vector_length> const &value) {
    #pragma HLS INLINE
    *ptr = value;
  }
  template <typename T, typename TOffset>
  static void Write(vec<T, vector_length> *ptr, TOffset offset,
                    vec<T, vector_length> const &value) {
    #pragma HLS INLINE
    ptr[offset] = value;
  }
  template <typename T, typename TOffset>
  static void Write(T *ptr, TOffset offset,
                    vec<T, vector_length> const &value) {
    #pragma HLS INLINE
    value.Unpack(&ptr[offset]);
  }
  template <typename TVec>
  static void Write(TVec &ref, TVec const &value) {
    #pragma HLS INLINE
    ref = value;
  }
};

template <>
struct WriteImpl<1> {
  template <typename T>
  static void Write(vec<T, 1> *ptr, vec<T, 1> const &value) {
    #pragma HLS INLINE
    *ptr = value;
  }
  template <typename T, typename TOffset>
  static void Write(vec<T, 1> *ptr, TOffset offset, vec<T, 1> const &value) {
    #pragma HLS INLINE
    ptr[offset] = value;
  }
  template <typename T>
  static void Write(vec<T, 1> &ref, vec<T, 1> const &value) {
    #pragma HLS INLINE
    ref = value;
  }
};

template <typename T, unsigned dims, unsigned vector_length, int num_accesses,
          typename TIndex>
class ArrayViewImpl {
 public:
  template <typename... Is>
  ArrayViewImpl(Is const &... strides)
      : strides_{static_cast<TIndex>(strides)...} {}

  template <TIndex dim, typename I, typename... Is>
  void IndexImpl(TIndex &offset, I const &index_val,
                 Is const &... index_vals) const {
    #pragma HLS INLINE
    offset += TIndex(index_val) * strides_[dim];
    IndexImpl<dim + 1, Is...>(offset, index_vals...);
  }

  template <TIndex dim, typename I>
  void IndexImpl(TIndex &offset, I const &index_val) const {
    #pragma HLS INLINE
    offset += TIndex(index_val) * strides_[dim] * vector_length;
  }

  template <typename... Is>
  TIndex Index(Is const &... indices) const {
    #pragma HLS INLINE
    TIndex offset = 0;
    IndexImpl<0>(offset, indices...);
    // We use packed types, so divide the final index expression with the vector
    // length to apply this to all dimensions
    return offset / vector_length;
  }

 private:
  TIndex strides_[dims];
};

}  // End anonymous namespace

template <typename T, unsigned dims, unsigned vector_length, int num_accesses>
class ArrayViewIn {
 private:
  using Index_t = unsigned;
  using Vec_t = vec<T, vector_length>;

 public:
  template <typename... Is>
  ArrayViewIn(ArrayInterface<Vec_t> ptr, Is const &... strides)
      : ptr_(ptr.ptr_in()), impl_{strides...} {
    static_assert(sizeof...(strides) == static_cast<int>(dims),
                  "Dimension mismatch");
  }

  template <typename... Is>
  explicit ArrayViewIn(Vec_t const *ptr, Is const &... strides)
      : ptr_(ptr), impl_{strides...} {
    static_assert(sizeof...(strides) == static_cast<int>(dims),
                  "Dimension mismatch");
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t const *ptr() const {
    #pragma HLS INLINE
    return ptr_;
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t const &ref() const {
    #pragma HLS INLINE
    return *ptr();
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t val() const {
    #pragma HLS INLINE
    return *ptr();
  }

  operator Vec_t const *() const {
    #pragma HLS INLINE
    return ptr();
  }

  template <typename... Is>
  Vec_t operator()(Is const &... indices) const {
    #pragma HLS INLINE
    static_assert(sizeof...(indices) == dims, "Dimension mismatch");
    return get(indices...);
  }

 private:
  template <typename... Is>
  Vec_t get(Is const &... indices) const {
    #pragma HLS INLINE
    const auto i = impl_.Index(indices...);
    return ptr_[i];
  }

 private:
  Vec_t const *ptr_;
  ArrayViewImpl<T, dims, vector_length, num_accesses, Index_t> impl_;
};

template <typename T, unsigned dims, unsigned vector_length, int num_accesses>
class ArrayViewOut {
 private:
  using Index_t = unsigned;
  using Vec_t = vec<T, vector_length>;

 public:
  template <typename... Is>
  ArrayViewOut(ArrayInterface<Vec_t> ptr, Is const &... strides)
      : ptr_(ptr.ptr_out()), impl_{strides...} {
    #pragma HLS INLINE
    static_assert(sizeof...(strides) == static_cast<int>(dims),
                  "Dimension mismatch");
  }

  template <typename... Is>
  explicit ArrayViewOut(Vec_t *ptr, Is const &... strides)
      : ptr_(ptr), impl_{strides...} {
    static_assert(sizeof...(strides) == static_cast<int>(dims),
                  "Dimension mismatch");
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t *ptr() {
    #pragma HLS INLINE
    return ptr_;
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t &ref() {
    #pragma HLS INLINE
    return *ptr();
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t val() {
    #pragma HLS INLINE
    return *ptr();
  }

  operator Vec_t *() {
    #pragma HLS INLINE
    return ptr();
  }

  template <typename... Is>
  void write(Vec_t const &value, Is const &... indices) {
    #pragma HLS INLINE
    static_assert(sizeof...(indices) == dims, "Dimension mismatch");
    set(value, indices...);
  }

  void operator=(Vec_t const &value) {
    #pragma HLS INLINE
    write(value);
  }

  template <ReductionType reduction_type, typename... Is>
  void write_and_resolve(vec<T, vector_length> const &value,
                         Is const &... indices) {
    #pragma HLS INLINE
    using Functor =
        typename ConvertReduction<reduction_type>::template Operator<T>;
    const auto i = impl_.Index(indices...);
    WriteImpl<vector_length>::template Write<T, decltype(i)>(
        ptr_, i,
        Reduce<T, vector_length, vector_length, Functor>(ptr_[i], value));
  }

  template <ReductionType reduction_type, typename... Is>
  void write_and_resolve_nc(vec<T, vector_length> const &value,
                            Is const &... indices) {
    #pragma HLS INLINE
    return write_and_resolve<reduction_type>(value, indices...);
  }

 private:
  template <typename... Is>
  void set(vec<T, vector_length> const &value, Is const &... indices) const {
    #pragma HLS INLINE
    const auto i = impl_.Index(indices...);
    WriteImpl<vector_length>::template Write<T, decltype(i)>(ptr_, i, value);
    // #pragma HLS DEPENDENCE variable=ptr_ false
  }

 private:
  Vec_t *ptr_;
  ArrayViewImpl<T, dims, vector_length, num_accesses, Index_t> impl_;
};

/// Scalar version special case
template <typename T, unsigned vector_length, int num_accesses>
class ArrayViewIn<T, 0, vector_length, num_accesses> {
 private:
  using Index_t = unsigned;
  using Vec_t = vec<T, vector_length>;

 public:
  ArrayViewIn(ArrayInterface<Vec_t> interface) : ptr_(interface.ptr_in()) {
    #pragma HLS INLINE
  }

  explicit ArrayViewIn(Vec_t const *ptr) : ptr_(ptr) {
    #pragma HLS INLINE
  }

  explicit ArrayViewIn(Vec_t const &ref) : ptr_(&ref) {
    #pragma HLS INLINE
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t const *ptr() const {
    #pragma HLS INLINE
    return ptr_;
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t const &ref() const {
    #pragma HLS INLINE
    return *ptr();
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t val() const {
    #pragma HLS INLINE
    return *ptr();
  }

  operator Vec_t const *() const {
    #pragma HLS INLINE
    return ptr();
  }

  operator Vec_t() const {
    #pragma HLS INLINE
    return val();
  }

  template <typename I>
  T operator()(I const &i) const {
    #pragma HLS INLINE
#ifndef DACE_SYNTHESIS
    if (i < 0 || i >= vector_length) {
      throw std::runtime_error("Vector index out of bounds: " +
                               std::to_string(i));
    }
#endif
    return val()[i];
  }

 private:
  Vec_t const *ptr_;
};

/// Scalar version special case
template <typename T, unsigned vector_length, int num_accesses>
class ArrayViewOut<T, 0, vector_length, num_accesses> {
 private:
  using Index_t = unsigned;
  using Vec_t = vec<T, vector_length>;

 public:
  ArrayViewOut(ArrayInterface<Vec_t> interface) : ptr_(interface.ptr_out()) {
    #pragma HLS INLINE
  }

  explicit ArrayViewOut(Vec_t *ptr) : ptr_(ptr) {
    #pragma HLS INLINE 
    // #pragma HLS DEPENDENCE variable=ptr_ false
  }

  ArrayViewOut(Vec_t &ref) : ptr_(&ref) {
    #pragma HLS INLINE
  }

  // Conforms to interface by allowing template argument, but will fail for
  // mismatching vector length
  template <unsigned vector_length_other = vector_length>
  Vec_t *ptr() {
    #pragma HLS INLINE
    return ptr_;
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t &ref() {
    #pragma HLS INLINE
    return *ptr();
  }

  template <unsigned vector_length_other = vector_length>
  Vec_t val() {
    #pragma HLS INLINE
    return *ptr();
  }

  operator Vec_t *() {
    #pragma HLS INLINE
    return ptr();
  }

  operator Vec_t &() {
    #pragma HLS INLINE
    return ref();
  }

  template <typename I>
  hlslib::DataPackProxy<T, vector_length> operator()(I const &i) {
    #pragma HLS INLINE
#ifndef DACE_SYNTHESIS
    if (i < 0 || i >= vector_length) {
      throw std::runtime_error("Vector index out of bounds: " +
                               std::to_string(i));
    }
#endif
    return ref()[i];
  }

  void write(Vec_t const &value) {
    #pragma HLS INLINE
    WriteImpl<vector_length>::template Write<T>(ptr_, value);
  }

  void operator=(Vec_t const &value) {
    #pragma HLS INLINE
    return write(value);
  }

  template <ReductionType reduction_type>
  void write_and_resolve(vec<T, vector_length> const &value) {
    #pragma HLS INLINE
    using Functor =
        typename ConvertReduction<reduction_type>::template Operator<T>;
    write(Reduce<T, vector_length, vector_length, Functor>(*ptr_, value));
  }

  template <ReductionType reduction_type>
  void write_and_resolve_nc(vec<T, vector_length> const &value) {
    #pragma HLS INLINE
    return write_and_resolve<reduction_type>(value);
  }

 private:
  Vec_t *ptr_;
};

}  // End namespace dace
