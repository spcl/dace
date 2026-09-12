// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_COMPLEX_H
#define __DACE_COMPLEX_H

#include <complex>
#include <type_traits>

#include "types.h"

#ifdef __CUDACC__
#include <thrust/complex.h>
#define dace_conj thrust::conj

template <typename T>
using cmplx = thrust::complex<T>;
#else
#define dace_conj std::conj

template <typename T>
using cmplx = std::complex<T>;
#endif

// Contains a complex-j class to support the native complex type in Python

namespace dace {
struct complexJ {
  int val;
  explicit DACE_HDFI complexJ(int v = 1) : val(v) {}
};

static DACE_HDFI int operator*(const complexJ& j1, const complexJ& j2) {
  return -j1.val * j2.val;
}
template <typename T>
cmplx<T> DACE_HDFI operator*(const complexJ& j, const T& other) {
  return cmplx<T>(T(0), j.val * other);
}
template <typename T>
cmplx<T> DACE_HDFI operator*(const T& other, const complexJ& j) {
  return cmplx<T>(T(0), j.val * other);
}
template <typename T>
cmplx<T> DACE_HDFI operator*(const complexJ& j, const cmplx<T>& other) {
  return cmplx<T>(T(0), j.val) * other;
}
template <typename T>
cmplx<T> DACE_HDFI operator*(const cmplx<T>& other, const complexJ& j) {
  return cmplx<T>(T(0), j.val) * other;
}
static DACE_HDFI complexJ operator*(const int& other, const complexJ& j) {
  return complexJ(j.val * other);
}
static DACE_HDFI complexJ operator*(const complexJ& j, const int& other) {
  return complexJ(j.val * other);
}
static DACE_HDFI complexJ operator-(const complexJ& j) {
  return complexJ(-j.val);
}
}  // namespace dace

#ifndef __CUDACC__
// std::complex<T> takes only a T scalar; these take any other arithmetic scalar, evaluate in the common type and
// narrow to T. thrust::complex has its own mixed-type operators.
namespace dace {
template <typename T, typename S>
using complex_with_scalar_t = std::enable_if_t<std::is_arithmetic_v<S> && !std::is_same_v<S, T>, cmplx<T>>;
}  // namespace dace

template <typename T, typename S>
DACE_HDFI dace::complex_with_scalar_t<T, S> operator*(const cmplx<T>& a, const S& b) {
  using C = std::common_type_t<T, S>;
  return cmplx<T>(cmplx<C>(a) * C(b));
}
template <typename T, typename S>
DACE_HDFI dace::complex_with_scalar_t<T, S> operator*(const S& a, const cmplx<T>& b) {
  using C = std::common_type_t<T, S>;
  return cmplx<T>(C(a) * cmplx<C>(b));
}
template <typename T, typename S>
DACE_HDFI dace::complex_with_scalar_t<T, S> operator/(const cmplx<T>& a, const S& b) {
  using C = std::common_type_t<T, S>;
  return cmplx<T>(cmplx<C>(a) / C(b));
}
template <typename T, typename S>
DACE_HDFI dace::complex_with_scalar_t<T, S> operator/(const S& a, const cmplx<T>& b) {
  using C = std::common_type_t<T, S>;
  return cmplx<T>(C(a) / cmplx<C>(b));
}
template <typename T, typename S>
DACE_HDFI dace::complex_with_scalar_t<T, S> operator+(const cmplx<T>& a, const S& b) {
  using C = std::common_type_t<T, S>;
  return cmplx<T>(cmplx<C>(a) + C(b));
}
template <typename T, typename S>
DACE_HDFI dace::complex_with_scalar_t<T, S> operator+(const S& a, const cmplx<T>& b) {
  using C = std::common_type_t<T, S>;
  return cmplx<T>(C(a) + cmplx<C>(b));
}
template <typename T, typename S>
DACE_HDFI dace::complex_with_scalar_t<T, S> operator-(const cmplx<T>& a, const S& b) {
  using C = std::common_type_t<T, S>;
  return cmplx<T>(cmplx<C>(a) - C(b));
}
template <typename T, typename S>
DACE_HDFI dace::complex_with_scalar_t<T, S> operator-(const S& a, const cmplx<T>& b) {
  using C = std::common_type_t<T, S>;
  return cmplx<T>(C(a) - cmplx<C>(b));
}
#endif

#endif  // __DACE_COMPLEX_H
