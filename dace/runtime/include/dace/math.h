// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_MATH_H
#define __DACE_MATH_H

#include <cfloat>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <type_traits>

#include "ITE.h"
#include "nan.h"
#include "pi.h"
#include "types.h"

#ifdef __CUDACC__
#include <thrust/complex.h>
#endif

// dace::math: A namespace that contains typeless math functions

// Math functions that are Python/sympy built-ins and must reside outside
// of the DaCe namespace for ease of code generation

// Math and python builtins
using std::abs;

// Ternary workarounds so that vector types work
// template <typename T>
// DACE_CONSTEXPR DACE_HDFI T min(const T& a, const T& b) {
//     return (a < b) ? a : b;
// }
// template <typename T>
// DACE_CONSTEXPR DACE_HDFI T max(const T& a, const T& b) {
//     return (a > b) ? a : b;
// }

// A later argument wins only by comparing strictly better, so a tie (or a NaN operand)
// keeps the earlier one, matching Python's and std::max/min's behavior.
template <typename T>
DACE_CONSTEXPR DACE_HDFI T min(const T& val) {
  return val;
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI typename std::common_type<T, Ts...>::type min(const T& a, const Ts&... ts) {
  return (min(ts...) < a) ? min(ts...) : a;
}

template <typename T>
DACE_CONSTEXPR DACE_HDFI T max(const T& val) {
  return val;
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI typename std::common_type<T, Ts...>::type max(const T& a, const Ts&... ts) {
  return (a < max(ts...)) ? max(ts...) : a;
}

// Implement to support a match wtih Fortran's intrinsic EXPONENT
template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI int frexp(const T& a) {
  int exponent = 0;
  std::frexp(a, &exponent);
  return exponent;
}

// Fortran SCALE(x, n) -- return x * 2^n.
template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T ldexp(const T& x, int n) {
  return std::ldexp(x, n);
}

// Fortran EXPONENT(x), named ilogb so the bridge maps _FortranAExponent* to one name.
template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI int ilogb(const T& x) {
  int e = 0;
  std::frexp(x, &e);
  return e;
}

// Implement to support Fortran's intrinsic NINT - round, but return an integer
template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI int iround(const T& a) {
  return static_cast<int>(round(a));
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T int_ceil(const T& numerator, const T2& denominator) {
  return (numerator + denominator - 1) / denominator;
}

static DACE_CONSTEXPR DACE_HDFI int ceiling(int arg) { return arg; }

static DACE_HDFI float ceiling(float /*arg*/) { return FLT_MAX; }

static DACE_HDFI double ceiling(double /*arg*/) { return DBL_MAX; }

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T int_floor(const T& numerator, const T2& denominator) {
  return numerator / denominator;
}

template <typename T>
static DACE_CONSTEXPR DACE_HDFI int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T bitwise_and(const T& left_operand, const T2& right_operand) {
  return left_operand & right_operand;
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T bitwise_or(const T& left_operand, const T2& right_operand) {
  return left_operand | right_operand;
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T bitwise_xor(const T& left_operand, const T2& right_operand) {
  return left_operand ^ right_operand;
}

// Unary: only ``T`` is deducible. A second template parameter (copied from the binary helpers
// above) made every call ``bitwise_invert(x)`` fail with "couldn't deduce template parameter 'T2'".
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T bitwise_invert(const T& value) {
  return ~value;
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T right_shift(const T& left_operand, const T2& right_operand) {
  return left_operand >> right_operand;
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T left_shift(const T& left_operand, const T2& right_operand) {
  return left_operand << right_operand;
}

// Logical (zero-fill) shifts: operate on the unsigned representation so a signed right
// shift does not sign-extend, matching Fortran ISHFT semantics.
template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T logical_left_shift(const T& left_operand, const T2& right_operand) {
  return static_cast<T>(static_cast<typename std::make_unsigned<T>::type>(left_operand) << right_operand);
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T logical_right_shift(const T& left_operand, const T2& right_operand) {
  return static_cast<T>(static_cast<typename std::make_unsigned<T>::type>(left_operand) >> right_operand);
}

#define AND(x, y) ((x) && (y))
#define OR(x, y) ((x) || (y))

template <typename T>
static DACE_CONSTEXPR DACE_HDFI T ROUND(const T& value) {
  return round(value);
}

// Workarounds for float16 in CUDA
// NOTES: * Half precision types are not trivially convertible, so other types
//          will be implicitly converted to it in min/max.
//        * half comparisons are designated "device-only", so they must call
//          device-only functions as well.
#ifdef __CUDACC__
template <typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 min(const dace::float16& a, const dace::float16& b,
                                                            const Ts&... c) {
  return (b < a) ? min(b, c...) : min(a, c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 min(const dace::float16& a, const T& b, const Ts&... c) {
  return (dace::float16(b) < a) ? min(dace::float16(b), c...) : min(a, c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 min(const T& a, const dace::float16& b, const Ts&... c) {
  return (b < dace::float16(a)) ? min(b, c...) : min(dace::float16(a), c...);
}
template <typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 max(const dace::float16& a, const dace::float16& b,
                                                            const Ts&... c) {
  return (a < b) ? max(b, c...) : max(a, c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 max(const dace::float16& a, const T& b, const Ts&... c) {
  return (a < dace::float16(b)) ? max(dace::float16(b), c...) : max(a, c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 max(const T& a, const dace::float16& b, const Ts&... c) {
  return (dace::float16(a) < b) ? max(b, c...) : max(dace::float16(a), c...);
}
#endif

#ifndef DACE_SYNTHESIS

// Computes integer floor, rounding the remainder towards negative infinity.
// https://stackoverflow.com/a/39304947
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T int_floor_ni(const T& numerator, const T& denominator) {
  // Not std::div: it is host-only, and nvcc silently drops it from device code.
  const T quotient = numerator / denominator;
  const T remainder = numerator % denominator;
  const T corr = (remainder != 0 && ((remainder < 0) != (denominator < 0)));
  return quotient - corr;
}
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T int_floor_ni(const T& numerator, const T& denominator) {
  T quotient = numerator / denominator;
  T remainder = numerator % denominator;
  T corr = (remainder != 0 && ((remainder < 0) != (denominator < 0)));
  return quotient - corr;
}

// Computes NumPy divmod: the quotient rounded toward negative infinity, and the remainder, which
// takes the divisor's sign. ``py_floor`` (``//``) and ``py_mod`` (``%``) are its two halves.
// Integers never trap: ``x // 0`` and ``x % 0`` are 0, and ``MIN // -1`` wraps to ``MIN`` with
// remainder 0, as NumPy answers.
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void py_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
  if (denominator == 0) {
    quotient = 0;
    remainder = 0;
  } else if (numerator == std::numeric_limits<T>::min() && denominator == static_cast<T>(-1)) {
    quotient = numerator;
    remainder = 0;
  } else {
    quotient = static_cast<T>(numerator / denominator);
    remainder = static_cast<T>(numerator % denominator);
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
      quotient = static_cast<T>(quotient - 1);
      remainder = static_cast<T>(remainder + denominator);
    }
  }
}
// Floating point follows NumPy's npy_divmod: the remainder is fmod's, which is exact, and the
// quotient is recovered from it and snapped to an integer. ``floor(a / b)`` is not: the division
// rounds first (``1.0 // 0.1`` would be 10, ``5 % inf`` nan, ``1e300 % 7`` off by 1e283).
template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void py_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
  // Divided before any comparison: under -fno-signed-zeros GCC substitutes +0 for a divisor known
  // to equal zero, so ``-5 // -0.`` would give -inf.
  const T ratio = numerator / denominator;
  remainder = std::fmod(numerator, denominator);
  if (denominator == 0) {
    quotient = ratio;
    return;
  }
  quotient = (numerator - remainder) / denominator;
  if (remainder == 0) {
    remainder = std::copysign(T(0), denominator);
  } else if ((denominator < 0) != (remainder < 0)) {
    remainder += denominator;
    quotient -= 1;
  }
  if (quotient == 0) {
    quotient = std::copysign(T(0), ratio);
  } else {
    const T floored = std::floor(quotient);
    quotient = (quotient - floored > T(0.5)) ? floored + 1 : floored;
  }
}

// Computes Python floor division (also NumPy floor_divide)
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T py_floor(const T& numerator, const T& denominator) {
  T quotient = 0;
  T remainder = 0;
  py_divmod(numerator, denominator, quotient, remainder);
  return quotient;
}
// Mixed operand types promote to their common type.
template <typename T1, typename T2, std::enable_if_t<!std::is_same<T1, T2>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI auto py_floor(const T1& numerator, const T2& denominator)
    -> decltype(numerator + denominator) {
  using T = decltype(numerator + denominator);
  return py_floor<T>((T)numerator, (T)denominator);
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> py_floor(const std::complex<T>& numerator,
                                                         const std::complex<T>& denominator) {
  std::complex<T> quotient = numerator / denominator;
  quotient.real(std::floor(quotient.real()));
  quotient.imag(0);
  return quotient;
}

// Computes NumPy float power
template <typename T>
static DACE_CONSTEXPR DACE_HDFI double np_float_pow(const T& base, const T& exponent) {
  return std::pow((double)base, (double)exponent);
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<double> np_float_pow(const std::complex<T>& base,
                                                                  const std::complex<T>& exponent) {
  return std::pow((std::complex<double>)base, (std::complex<double>)exponent);
}

// Computes Python modulus (also NumPy remainder): the remainder half of py_divmod
// NOTE: This is different than Python math.remainder and C remainder,
// which are equaivalent to the IEEE remainder: num - round(num / den) * den
template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T py_mod(const T& numerator, const T& denominator) {
  T quotient = 0;
  T remainder = 0;
  py_divmod(numerator, denominator, quotient, remainder);
  return remainder;
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> py_mod(const std::complex<T>& numerator,
                                                       const std::complex<T>& denominator) {
  return numerator - py_floor(numerator, denominator) * denominator;
}

template <typename T1, typename T2, std::enable_if_t<!std::is_same<T1, T2>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI auto py_mod(const T1& numerator, const T2& denominator)
    -> decltype(numerator + denominator) {
  using T = decltype(numerator + denominator);
  return py_mod<T>((T)numerator, (T)denominator);
}

// C modulus (CMod): truncating.
template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T c_mod(const T& numerator, const T& denominator) {
    return numerator % denominator;
}
template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T c_mod(const T& numerator, const T& denominator) {
    return (T)std::fmod(numerator, denominator);
}
template<typename T1, typename T2, std::enable_if_t<!std::is_same<T1, T2>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI auto c_mod(const T1& numerator, const T2& denominator) -> decltype(numerator + denominator) {
    using T = decltype(numerator + denominator);
    return c_mod<T>((T)numerator, (T)denominator);
}

// Fortran MOD (FtnMod).
template<typename T1, typename T2>
static DACE_CONSTEXPR DACE_HDFI auto ftn_mod(const T1& numerator, const T2& denominator) -> decltype(c_mod(numerator, denominator)) {
    return c_mod(numerator, denominator);
}

// Fortran MODULO (FtnModulo): floored.
template<typename T1, typename T2>
static DACE_CONSTEXPR DACE_HDFI auto ftn_modulo(const T1& numerator, const T2& denominator) -> decltype(py_mod(numerator, denominator)) {
    return py_mod(numerator, denominator);
}

// floor_mod(a, b) -- Fortran MODULO as the Fortran bridge spells it: the floored remainder, ftn_modulo.
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T floor_mod(const T& numerator, const T& denominator) {
  return ftn_modulo(numerator, denominator);
}

// Computes C/C++ divmod (std::div)
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void cpp_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
  auto divresult = std::div(numerator, denominator);
  quotient = (T)divresult.quot;
  remainder = (T)divresult.rem;
}
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void cpp_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
  quotient = numerator / denominator;
  remainder = numerator % denominator;
}
template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void cpp_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
  quotient = (T)std::floor(numerator / denominator);
  remainder = (T)std::fmod(numerator, denominator);
}

// Computes absolute value (support for unsigned integers)
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T abs(const T& a) {
  return a;
}

// Rounds to nearest integer (support for complex numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> round(const std::complex<T>& a) {
  return std::complex<T>(round(a.real()), round(a.imag()));
}

// Returns an indication of the sign of a number
// For non-complex numbers: -1 if x < 0, 0 if x == 0, 1 if x > 1
// For complex numbers: sign(x.real) + 0j if x.real !=0, else sign(x.imag) + 0j
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T sign(const T& x) {
  return T((T(0) < x) - (x < T(0)));
  // return (x < 0) ? -1 : ( (x > 0) ? 1 : 0);
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> sign(const std::complex<T>& x) {
  return (x.real() != 0) ? std::complex<T>(sign(x.real()), 0) : std::complex<T>(sign(x.imag()), 0);
}
// Numpy v2.0 or higher for complex inputs: sign(x) = x / abs(x)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T sign_numpy_2(const T& x) {
  return T((T(0) < x) - (x < T(0)));
  // return (x < 0) ? -1 : ( (x > 0) ? 1 : 0);
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> sign_numpy_2(const std::complex<T>& x) {
  return (x.real() != 0 && x.imag() != 0) ? x / std::abs(x) : std::complex<T>(0, 0);
}

// Computes the Heaviside step function
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T heaviside(const T& a, const T& b) {
  return (a < 0) ? 0 : ((a > 0) ? 1 : b);
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T heaviside(const T& a) {
  return (a > 0) ? 1 : 0;
}

// Computes the conjugate of a number (support for non-complex numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T conj(const T& a) {
  return a;
}

// Computes 2 raised to the given power n (support for complex numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> exp2(const std::complex<T>& n) {
  return std::exp(n * std::log(T(2)));
}

// Computes the base-2 logarithm of n (support for complex numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> log2(const std::complex<T>& n) {
  T radius = std::abs(n);
  T theta = std::arg(n);
  return std::complex<T>(std::log2(radius), theta / std::log(T(2)));
}

// Computes the e raised to the given power n, minus 1.0 (support for complex
// numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> expm1(const std::complex<T>& n) {
  return std::exp(n) - T(1);
}

// Computes the base-e logarithm of 1 + n (support for complex numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> log1p(const std::complex<T>& n) {
  return std::log(n + T(1));
}

// Computes the reciprocal of a number
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T reciprocal(const T& a) {
  return T(1) / a;
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> reciprocal(const std::complex<T>& a) {
  return T(1) / a;
}

#if __cplusplus < 201703L

// Compute the greates common divisor of two integers
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T gcd(T a, T b) {
  // Modern Euclidian algorithm
  // (Knuth, Art of Computer Programming - Vol. 2 Seminumerical Algorithms)
  while (b != 0) {
    auto t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute the least common multiple of two integers
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T lcm(T a, T b) {
  // lcm(a, b) = |a * b| / gcd(a, b)
  // more efficient lcm(a, b) = (|a| / gcd(a, b)) * |b|
  if (a == 0 && b == 0)  // special case
    return 0;
  return (abs(a) / gcd(a, b)) * abs(b);
}

#else

// Compute the greates common divisor of two integers
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T gcd(const T& a, const T& b) {
  return std::gcd(a, b);
}

// Compute the least common multiple of two integers
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T lcm(const T& a, const T& b) {
  return std::lcm(a, b);
}

#endif

// Converts angles from degrees to radians
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T deg2rad(const T& a) {
  return a * M_PI / T(180);
}

// Converts angles from radians to degrees
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T rad2deg(const T& a) {
  return a * T(180) / M_PI;
}

// Determines if the given (floating point) number has finite value
// (support for complex numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI bool isfinite(const std::complex<T>& a) {
  return std::isfinite(a.real()) && std::isfinite(a.imag());
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI bool isfinite(const T& a) {
  return std::isfinite(a);
}

// Determines if the given (floating point) number is a positive or negative
// infinity (support for complex numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI bool isinf(const std::complex<T>& a) {
  return std::isinf(a.real()) || std::isinf(a.imag());
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI bool isinf(const T& a) {
  return std::isinf(a);
}

// Determines if the given (floating point) number is not-a-number (NaN) value
// (support for complex numbers)
template <typename T>
static DACE_CONSTEXPR DACE_HDFI bool isnan(const std::complex<T>& a) {
  return std::isnan(a.real()) || std::isnan(a.imag());
}
template <typename T>
static DACE_CONSTEXPR DACE_HDFI bool isnan(const T& a) {
  return std::isnan(a);
}

// Determines if the given floating point number a is negative
template <typename T>
static DACE_CONSTEXPR DACE_HDFI bool signbit(const T& a) {
  return std::signbit(a);
}

// Computes modf (compatibility between Python tasklets and C++ modf)
template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void np_modf(const T& a, double& integral, double& fractional) {
  integral = double(a);
  fractional = double(0);
}
template <typename T, std::enable_if_t<!std::is_integral<T>::value && std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void np_modf(const T& a, T& integral, T& fractional) {
  fractional = std::modf(a, &integral);
}

// Computes frexp (compatibility between Python tasklets and C++ frexp)
template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void np_frexp(const T& a, T& mantissa, int& exponent) {
  mantissa = std::frexp(a, &exponent);
}

#endif

namespace dace {
namespace math {
static DACE_CONSTEXPR_HOSTDEV typeless_pi pi{};
static DACE_CONSTEXPR typeless_nan nan{};
//////////////////////////////////////////////////////

// Complex-component accessors: re(z)/im(z), which cppunparse maps tasklet-body re(_in)/im(_in)
// to. Generic over std::complex / thrust::complex via .real()/.imag().
template <typename T>
DACE_CONSTEXPR DACE_HDFI auto re(const T& z) -> decltype(z.real()) {
  return z.real();
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI auto im(const T& z) -> decltype(z.imag()) {
  return z.imag();
}
// ``np.real`` / ``np.imag`` of a REAL value: the value itself, and zero.
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
DACE_CONSTEXPR DACE_HDFI T re(const T& x) {
  return x;
}
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
DACE_CONSTEXPR DACE_HDFI T im(const T&) {
  return T(0);
}

template <typename T>
DACE_CONSTEXPR DACE_HDFI T exp(const T& a) {
  return (T)std::exp(a);
}

#ifdef __CUDACC__
template <typename T>
DACE_CONSTEXPR DACE_HDFI thrust::complex<T> pow(const thrust::complex<T>& a, const thrust::complex<T>& b) {
  return (thrust::complex<T>)thrust::pow(a, b);
}
#endif
template <typename T, typename U,
          typename std::enable_if<!(std::is_integral<T>::value && std::is_integral<U>::value)>::type* = nullptr>
DACE_CONSTEXPR DACE_HDFI auto pow(const T& a, const U& b) {
  return std::pow(a, b);
}

// An integer base raised to an integer exponent stays an integer for every integral width
// (not just int/unsigned int), since falling through to std::pow returns double where an
// OpenMP loop bound or pointer offset cannot take one. Negative exponents answer 0.
template <typename T, typename U,
          typename std::enable_if<std::is_integral<T>::value && std::is_integral<U>::value>::type* = nullptr>
DACE_CONSTEXPR DACE_HDFI T pow(const T& a, const U& b) {
  if (b < U(0)) return T(0);
  T result = T(1);
  for (U i = U(0); i < b; ++i) result *= a;
  return result;
}

// Seeds at T(1) so ipow(a, 0) == 1. Must stay DACE_CONSTEXPR: RelaxIntegerPowers lowers
// integer powers in shapes/strides to ipow, called from the codegen's constexpr helpers.
template <typename T, typename std::enable_if<std::is_constructible<T, int>::value>::type* = nullptr>
DACE_CONSTEXPR DACE_HDFI T ipow(const T a, const unsigned int b) {
  T result = T(1);
  for (unsigned int i = 0; i < b; ++i) result *= a;
  return result;
}

// Vector types have no scalar constructor, so seed at ``a``. Only ever reached with a
// compile-time exponent >= 1 (the constant-power path emits a literal 1 for exponent 0).
template <typename T, typename std::enable_if<!std::is_constructible<T, int>::value>::type* = nullptr>
DACE_CONSTEXPR DACE_HDFI T ipow(const T a, const unsigned int b) {
  T result = a;
  for (unsigned int i = 1; i < b; ++i) result *= a;
  return result;
}

template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
DACE_CONSTEXPR DACE_HDFI T ifloor(const T& a) {
  return a;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
DACE_CONSTEXPR DACE_HDFI int ifloor(const T& a) {
  return (int)std::floor(a);
}

template <typename T>
DACE_CONSTEXPR DACE_HDFI T sin(const T& a) {
  return std::sin(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T sinh(const T& a) {
  return std::sinh(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T cos(const T& a) {
  return std::cos(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T cosh(const T& a) {
  return std::cosh(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T tan(const T& a) {
  return std::tan(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T tanh(const T& a) {
  return std::tanh(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T sqrt(const T& a) {
  return std::sqrt(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T log(const T& a) {
  return std::log(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T log10(const T& a) {
  return std::log10(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T log1p(const T& a) {
  return std::log1p(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T log2(const T& a) {
  return std::log2(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T exp2(const T& a) {
  return (T)std::exp2(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T expm1(const T& a) {
  return (T)std::expm1(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T asin(const T& a) {
  return std::asin(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T asinh(const T& a) {
  return std::asinh(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T acos(const T& a) {
  return std::acos(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T acosh(const T& a) {
  return std::acosh(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T atan(const T& a) {
  return std::atan(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T atan2(const T& a, const T& b) {
  return std::atan2(a, b);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T atanh(const T& a) {
  return std::atanh(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T cbrt(const T& a) {
  return std::cbrt(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T fmod(const T& a, const T& b) {
  return std::fmod(a, b);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T lgamma(const T& a) {
  return std::lgamma(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T tgamma(const T& a) {
  return std::tgamma(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T ceil(const T& a) {
  return std::ceil(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T trunc(const T& a) {
  return std::trunc(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T erf(const T& a) {
  return std::erf(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T erfc(const T& a) {
  return std::erfc(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T nearbyint(const T& a) {
  return std::nearbyint(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T round(const T& a) {
  return std::round(a);
}
template <typename T>
DACE_CONSTEXPR DACE_HDFI T hypot(const T& a, const T& b) {
  return std::hypot(a, b);
}

// Fused multiply-add ``a*b + c``, where ``cppunparse`` sends the tasklet-body
// ``fma(a, b, c)``.  Forwards verbatim, so 32/64-bit stay bit-identical.
template <typename T, typename U, typename V>
DACE_CONSTEXPR DACE_HDFI auto fma(const T& a, const U& b, const V& c) {
  return std::fma(a, b, c);
}

// A 16-bit float makes all three std::fma overloads equally good (ambiguous), so this
// routes through float explicitly. Not DACE_CONSTEXPR: __half(float) never folds.
#define DACE_MATH_FMA_LP(TYPE)                                             \
  static DACE_HDFI TYPE fma(const TYPE& a, const TYPE& b, const TYPE& c) { \
    return TYPE(std::fma(float(a), float(b), float(c)));                   \
  }
DACE_MATH_FMA_LP(dace::float16)
DACE_MATH_FMA_LP(dace::bfloat16)
#undef DACE_MATH_FMA_LP

// 16-bit floats have no libm entry of their own: cast to fp32, call that, cast back -- the same
// route ``fma`` above needed, rather than leaning on user-defined-conversion ranking between the
// float/double/long double overloads of ``std::sqrt``/``exp``/``log``.
#define DACE_MATH_UNARY_LP(NAME, TYPE) \
  static DACE_HDFI TYPE NAME(const TYPE& a) { return TYPE(std::NAME(float(a))); }
DACE_MATH_UNARY_LP(sqrt, dace::float16)
DACE_MATH_UNARY_LP(sqrt, dace::bfloat16)
// ``dace::float16`` IS ``half`` under CUDA, where dace/cuda/halfvec.cuh already declares a native
// ``exp(half)``: a second, equally viable overload makes every fp16 ``exp`` ambiguous and nvcc
// rejects the whole translation unit. halfvec gates that group on ``!__HIPCC__``, so HIP-on-NVIDIA
// -- which defines both macros -- gets no ``exp(half)`` from either header unless excluded here.
#if !defined(__CUDACC__) || defined(__HIPCC__)
DACE_MATH_UNARY_LP(exp, dace::float16)
#endif
DACE_MATH_UNARY_LP(exp, dace::bfloat16)
DACE_MATH_UNARY_LP(log, dace::float16)
DACE_MATH_UNARY_LP(log, dace::bfloat16)
#undef DACE_MATH_UNARY_LP

#ifdef DACE_THRUST_COMPLEX
// ``std::`` has no overload for ``thrust::complex`` (the device complex128/complex64), so the
// generic forwarders above fail to compile on it. Only functions thrust implements are listed.
#define DACE_MATH_THRUST_COMPLEX(NAME)                                            \
  template <typename T>                                                           \
  DACE_CONSTEXPR DACE_HDFI thrust::complex<T> NAME(const thrust::complex<T>& a) { \
    return thrust::NAME(a);                                                       \
  }
DACE_MATH_THRUST_COMPLEX(exp)
DACE_MATH_THRUST_COMPLEX(log)
DACE_MATH_THRUST_COMPLEX(log10)
DACE_MATH_THRUST_COMPLEX(sqrt)
DACE_MATH_THRUST_COMPLEX(sin)
DACE_MATH_THRUST_COMPLEX(cos)
DACE_MATH_THRUST_COMPLEX(tan)
DACE_MATH_THRUST_COMPLEX(sinh)
DACE_MATH_THRUST_COMPLEX(cosh)
DACE_MATH_THRUST_COMPLEX(tanh)
DACE_MATH_THRUST_COMPLEX(asin)
DACE_MATH_THRUST_COMPLEX(acos)
DACE_MATH_THRUST_COMPLEX(atan)
DACE_MATH_THRUST_COMPLEX(asinh)
DACE_MATH_THRUST_COMPLEX(acosh)
DACE_MATH_THRUST_COMPLEX(atanh)
#undef DACE_MATH_THRUST_COMPLEX
#endif
}  // namespace math

namespace cmath {
template <typename T>
DACE_CONSTEXPR std::complex<T> exp(const std::complex<T>& a) {
  return std::exp(a);
}

#ifdef __CUDACC__
template <typename T>
DACE_CONSTEXPR DACE_HDFI thrust::complex<T> exp(const thrust::complex<T>& a) {
  return thrust::exp(a);
}
#endif

template <typename T>
DACE_CONSTEXPR std::complex<T> conj(const std::complex<T>& a) {
  return std::conj(a);
}

#ifdef __CUDACC__
template <typename T>
DACE_CONSTEXPR DACE_HDFI thrust::complex<T> conj(const thrust::complex<T>& a) {
  return thrust::conj(a);
}
#endif
}  // namespace cmath

}  // namespace dace

// Global-scope wrapper (like ``min`` / ``max`` / ``int_ceil``) so codegen can emit the
// bare ``ipow`` in loop bounds / interstate edges; forwards to ``dace::math::ipow``.
template <typename T, typename U>
static DACE_HDFI T ipow(const T& a, const U& b) {
  return dace::math::ipow(a, b);
}

#endif  // __DACE_MATH_H
