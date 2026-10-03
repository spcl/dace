// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_MATH_H
#define __DACE_MATH_H

#include <complex>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <type_traits>

#include "pi.h"
#include "nan.h"
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

template <typename T>
DACE_CONSTEXPR DACE_HDFI T min(const T& val)
{
    return val;
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI typename std::common_type<T, Ts...>::type min(const T& a, const Ts&... ts)
{
    return (a < min(ts...)) ? a : min(ts...);
}

template <typename T>
DACE_CONSTEXPR DACE_HDFI T max(const T& val)
{
    return val;
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI typename std::common_type<T, Ts...>::type max(const T& a, const Ts&... ts)
{
    return (a > max(ts...)) ? a : max(ts...);
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T Mod(const T& value, const T2& modulus) {
    return value % modulus;
}

// Fortran implements MOD for floating-point values as well
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T Mod_float(const T& value, const T& modulus) {
    return value - static_cast<int>(value / modulus) * modulus;
}

// Fortran implementation of MODULO
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T Modulo(const T& value, const T& modulus) {
    // Fortran implementation for integers - find R such that value = Q * modulus + R
    // However, R must be in [0, modulus)
    // To achieve that, we need to cast the division to floats.
    // Example: -17, 3 must produce 1 and not -2.
    // If we don't use cast, the floor is called on -5, producing wrong value.
    // Instead, we need to have floor(-5.6... ) to ensure it produces -6.
    // Similarly, 17, -3 must produce -1 and not 2.
    // This means that the default solution works if value and modulus have the same sign.
    return value - floor(static_cast<float>(value) / modulus) * modulus;
}

template <typename T>
static DACE_CONSTEXPR DACE_HDFI T Modulo_float(const T& value, const T& modulus) {
    return value - floor(value / modulus) * modulus;
}

// Implement to support a match with Fortran's intrinsic EXPONENT
template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI int frexp(const T& a) {
  int exponent = 0;
  std::frexp(a, &exponent);
  return exponent;
}

// Implement to support Fortran's intrinsic NINT - round, but return an integer
template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI int iround(const T& a) {
  return static_cast<int>(round(a));
}

template <typename T, typename T2>
static DACE_CONSTEXPR DACE_HDFI T int_ceil(const T& numerator, const T2& denominator) {
    return (numerator + denominator - 1) / denominator;
}

static DACE_CONSTEXPR DACE_HDFI int ceiling(int arg) {
    return arg;
}

static DACE_HDFI float ceiling(float /*arg*/) {
    return FLT_MAX;
}

static DACE_HDFI double ceiling(double /*arg*/) {
    return DBL_MAX;
}

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

template <typename T, typename T2>
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

#define AND(x, y) ((x) && (y))
#define OR(x, y) ((x) || (y))

template <typename T>
static DACE_CONSTEXPR DACE_HDFI T ROUND(const T& value) {
    return round(value);
}

// Workarounds for float16 in CUDA
// NOTES: * Half precision types are not trivially convertible, so other types
//          will be implicitly converted to it in min/max -- except a float or double, which
//          would be rounded to half before the comparison (``min(1e-20, h)`` answering 0). Those
//          compare in the wider type instead and return it.
#ifdef __CUDACC__
template <typename... Ts>
DACE_CONSTEXPR DACE_HDFI dace::float16 min(const dace::float16& a, const dace::float16& b, const Ts&... c)
{
    return (a < b) ? min(a, c...) : min(b, c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI auto min(const dace::float16& a, const T& b, const Ts&... c)
{
    if constexpr (std::is_floating_point<T>::value) {
        return min(T(a), b, c...);
    } else {
        return dace::float16((a < dace::float16(b)) ? min(a, c...) : min(dace::float16(b), c...));
    }
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI auto min(const T& a, const dace::float16& b, const Ts&... c)
{
    if constexpr (std::is_floating_point<T>::value) {
        return min(a, T(b), c...);
    } else {
        return dace::float16((dace::float16(a) < b) ? min(dace::float16(a), c...) : min(b, c...));
    }
}
template <typename... Ts>
DACE_CONSTEXPR DACE_HDFI dace::float16 max(const dace::float16& a, const dace::float16& b, const Ts&... c)
{
    return (a > b) ? max(a, c...) : max(b, c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI auto max(const dace::float16& a, const T& b, const Ts&... c)
{
    if constexpr (std::is_floating_point<T>::value) {
        return max(T(a), b, c...);
    } else {
        return dace::float16((a > dace::float16(b)) ? max(a, c...) : max(dace::float16(b), c...));
    }
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI auto max(const T& a, const dace::float16& b, const Ts&... c)
{
    if constexpr (std::is_floating_point<T>::value) {
        return max(a, T(b), c...);
    } else {
        return dace::float16((dace::float16(a) > b) ? max(dace::float16(a), c...) : max(b, c...));
    }
}

// Mixed half / built-in arithmetic operators. ``half`` converts to float and is constructible
// from every arithmetic type, so ``half < 1e-14`` or ``float / half`` matches both the built-in
// operator and ``operator<op>(__half, __half)`` and nvcc rejects it as ambiguous. These exact
// matches win: the half is promoted to float and the built-in operator runs, so the result
// follows the usual promotion (half op float -> float, half op double -> double).
#define DACE_HALF_MIXED_OP(OP)                                                     \
    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr> \
    DACE_HDFI auto operator OP(const dace::float16& a, const T& b)                 \
    {                                                                              \
        return float(a) OP b;                                                      \
    }                                                                              \
    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr> \
    DACE_HDFI auto operator OP(const T& a, const dace::float16& b)                 \
    {                                                                              \
        return a OP float(b);                                                      \
    }
DACE_HALF_MIXED_OP(+)
DACE_HALF_MIXED_OP(-)
DACE_HALF_MIXED_OP(*)
DACE_HALF_MIXED_OP(/)
DACE_HALF_MIXED_OP(<)
DACE_HALF_MIXED_OP(<=)
DACE_HALF_MIXED_OP(>)
DACE_HALF_MIXED_OP(>=)
DACE_HALF_MIXED_OP(==)
DACE_HALF_MIXED_OP(!=)
#undef DACE_HALF_MIXED_OP

// Compound assignment into a built-in (``float x; x += h``): the built-in ``+=`` takes any
// promoted arithmetic right operand, and half converts to each of them equally well.
#define DACE_HALF_MIXED_ASSIGN_OP(OP)                                              \
    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr> \
    DACE_HDFI T& operator OP##=(T& a, const dace::float16& b)                      \
    {                                                                              \
        return a = static_cast<T>(a OP float(b));                                  \
    }
DACE_HALF_MIXED_ASSIGN_OP(+)
DACE_HALF_MIXED_ASSIGN_OP(-)
DACE_HALF_MIXED_ASSIGN_OP(*)
DACE_HALF_MIXED_ASSIGN_OP(/)
#undef DACE_HALF_MIXED_ASSIGN_OP

// The half-only functions below are templates on purpose. ``half`` lives in the global namespace,
// so argument-dependent lookup finds these globals from inside ``dace::math`` too, where they would
// tie with the non-template half overloads there (``dace::math::exp(half)`` and ``tanh`` in
// dace/cuda/halfvec.cuh, the ones further down in this file). On a tie the non-template wins.
#define DACE_HALF_ONLY template <typename H, std::enable_if_t<std::is_same<H, dace::float16>::value>* = nullptr>

// ``abs(half)`` would otherwise be ambiguous between the float/double/long double std::abs.
DACE_HALF_ONLY DACE_HDFI H abs(const H& a)
{
    return __habs(a);
}

// libm functions called unqualified (``sin(h)``): each std:: overload is equally good for a half,
// so this exact match runs the float version and rounds back. A half paired with a built-in in a
// binary function follows the operators above: the half is promoted to float.
#define DACE_HALF_UNARY(NAME)                \
    DACE_HALF_ONLY DACE_HDFI H NAME(const H& a) \
    {                                        \
        return H(std::NAME(float(a)));       \
    }
#define DACE_HALF_BINARY(NAME)                                                     \
    DACE_HALF_ONLY DACE_HDFI H NAME(const H& a, const H& b)                        \
    {                                                                              \
        return H(std::NAME(float(a), float(b)));                                   \
    }                                                                              \
    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr> \
    DACE_HDFI auto NAME(const dace::float16& a, const T& b)                        \
    {                                                                              \
        return std::NAME(float(a), b);                                             \
    }                                                                              \
    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr> \
    DACE_HDFI auto NAME(const T& a, const dace::float16& b)                        \
    {                                                                              \
        return std::NAME(a, float(b));                                             \
    }
DACE_HALF_UNARY(sqrt)
DACE_HALF_UNARY(cbrt)
DACE_HALF_UNARY(exp)
DACE_HALF_UNARY(exp2)
DACE_HALF_UNARY(expm1)
DACE_HALF_UNARY(log)
DACE_HALF_UNARY(log2)
DACE_HALF_UNARY(log10)
DACE_HALF_UNARY(log1p)
DACE_HALF_UNARY(sin)
DACE_HALF_UNARY(cos)
DACE_HALF_UNARY(tan)
DACE_HALF_UNARY(asin)
DACE_HALF_UNARY(acos)
DACE_HALF_UNARY(atan)
DACE_HALF_UNARY(sinh)
DACE_HALF_UNARY(cosh)
DACE_HALF_UNARY(tanh)
DACE_HALF_UNARY(asinh)
DACE_HALF_UNARY(acosh)
DACE_HALF_UNARY(atanh)
DACE_HALF_UNARY(erf)
DACE_HALF_UNARY(erfc)
DACE_HALF_UNARY(tgamma)
DACE_HALF_UNARY(lgamma)
DACE_HALF_UNARY(floor)
DACE_HALF_UNARY(ceil)
DACE_HALF_UNARY(trunc)
DACE_HALF_UNARY(round)
DACE_HALF_UNARY(rint)
DACE_HALF_UNARY(nearbyint)
DACE_HALF_BINARY(fmod)
DACE_HALF_BINARY(remainder)
DACE_HALF_BINARY(atan2)
DACE_HALF_BINARY(hypot)
DACE_HALF_BINARY(fmin)
DACE_HALF_BINARY(fmax)
DACE_HALF_BINARY(fdim)
DACE_HALF_BINARY(copysign)
#undef DACE_HALF_UNARY
#undef DACE_HALF_BINARY
#undef DACE_HALF_ONLY

// ``std::common_type`` of a half and a built-in has no answer (``true ? h : 1.0`` is ambiguous), so
// every helper that names its result through it -- ``IfExpr``, the variadic ``min``/``max`` --
// dropped out of overload resolution. Answer it the way the operators above promote: the half
// counts as a float. Any other pairing keeps the standard rule (the type of the conditional
// expression, or no ``type`` when that is ill-formed).
template <typename A, typename B, typename = void>
struct _dace_half_ternary_type {};
template <typename A, typename B>
struct _dace_half_ternary_type<A, B, std::void_t<decltype(false ? std::declval<A>() : std::declval<B>())>>
{
    using type = std::decay_t<decltype(false ? std::declval<A>() : std::declval<B>())>;
};
namespace std
{
template <typename T>
struct common_type<dace::float16, T>
    : conditional<is_arithmetic<T>::value, common_type<float, T>, _dace_half_ternary_type<dace::float16, T>>::type
{
};
template <typename T>
struct common_type<T, dace::float16>
    : conditional<is_arithmetic<T>::value, common_type<T, float>, _dace_half_ternary_type<T, dace::float16>>::type
{
};
template <>
struct common_type<dace::float16, dace::float16>
{
    using type = dace::float16;
};
}  // namespace std
#endif


#ifndef DACE_SYNTHESIS



// Computes integer floor, rounding the remainder towards negative infinity.
// https://stackoverflow.com/a/39304947
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T int_floor_ni(const T& numerator, const T& denominator) {
    auto divresult = std::div(numerator, denominator);
    T corr = (divresult.rem != 0 && ((divresult.rem < 0) != (denominator < 0)));
    return (T)divresult.quot - corr;
}
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T int_floor_ni(const T& numerator, const T& denominator) {
    T quotient = numerator / denominator;
    T remainder = numerator % denominator;
    T corr = (remainder != 0 && ((remainder < 0) != (denominator < 0)));
    return quotient - corr;
}

// Computes Python floor division
template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T py_floor(const T& numerator, const T& denominator) {
    return int_floor_ni(numerator, denominator);
}
template<typename T, std::enable_if_t<!std::is_integral<T>::value && std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T py_floor(const T& numerator, const T& denominator) {
    return (T)std::floor(numerator / denominator);
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> py_floor(const std::complex<T>& numerator, const std::complex<T>& denominator) {
    std::complex<T> quotient = numerator / denominator;
    quotient.real(std::floor(quotient.real()));
    quotient.imag(0);
    return quotient;
}

// Computes NumPy float power
template<typename T>
static DACE_CONSTEXPR DACE_HDFI double np_float_pow(const T& base, const T& exponent) {
    return std::pow((double)base, (double)exponent);
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<double> np_float_pow(const std::complex<T>& base, const std::complex<T>& exponent) {
    return std::pow((std::complex<double>)base, (std::complex<double>)exponent);
}

// Computes Python modulus (also NumPy remainder)
// Formula: num - (num // den) * den
// NOTE: This is different than Python math.remainder and C remainder,
// which are equivalent to the IEEE remainder: num - round(num / den) * den
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T py_mod(const T& numerator, const T& denominator) {
    T quotient = py_floor(numerator, denominator);
    return (T)(numerator - quotient * denominator);
}

// Computes C/C++ modulus (operator % and fmod)
template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T cpp_mod(const T& numerator, const T& denominator) {
    return numerator % denominator;
}
template<typename T, std::enable_if_t<!std::is_integral<T>::value && std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T cpp_mod(const T& numerator, const T& denominator) {
    return (T)std::fmod(numerator, denominator);
}

// Computes C/C++ divmod (std::div)
template<typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void cpp_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
    auto divresult = std::div(numerator, denominator);
    quotient = (T)divresult.quot;
    remainder = (T)divresult.rem;
}
template<typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void cpp_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
    quotient = numerator / denominator;
    remainder = numerator % denominator;
}
template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void cpp_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
    quotient = (T)std::floor(numerator / denominator);
    remainder = (T)std::fmod(numerator, denominator);
}

// Computes Python divmod
template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void py_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
    cpp_divmod(numerator, denominator, quotient, remainder);
    T corr = (remainder != 0 && ((remainder < 0) != (denominator < 0)));
    quotient -= corr;
    remainder += corr * denominator;
}
template<typename T, std::enable_if_t<!std::is_integral<T>::value && std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void py_divmod(const T& numerator, const T& denominator, T& quotient, T& remainder) {
    quotient = (T)std::floor(numerator / denominator);
    remainder = numerator - quotient * denominator;
}

// Computes absolute value (support for unsigned integers)
template<typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI T abs(const T& a) {
    return a;
}

// Rounds to nearest integer (support for complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> round(const std::complex<T>& a) {
    return std::complex<T>(round(a.real()), round(a.imag()));
}

// Returns an indication of the sign of a number
// For non-complex numbers: -1 if x < 0, 0 if x == 0, 1 if x > 1
// For complex numbers: sign(x.real) + 0j if x.real !=0, else sign(x.imag) + 0j
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T sign(const T& x) {
    return T( (T(0) < x) - (x < T(0)) );
    // return (x < 0) ? -1 : ( (x > 0) ? 1 : 0);
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> sign(const std::complex<T>& x) {
    return (x.real() != 0) ? std::complex<T>(sign(x.real()), 0) : std::complex<T>(sign(x.imag()), 0);
}
// Numpy v2.0 or higher for complex inputs: sign(x) = x / abs(x)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T sign_numpy_2(const T& x) {
    return T( (T(0) < x) - (x < T(0)) );
    // return (x < 0) ? -1 : ( (x > 0) ? 1 : 0);
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> sign_numpy_2(const std::complex<T>& x) {
    return (x.real() != 0 && x.imag() != 0) ? x / std::abs(x) : std::complex<T>(0, 0);
}

// Computes the Heaviside step function
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T heaviside(const T& a, const T& b) {
    return (a < 0) ? 0 : ( (a > 0) ? 1 : b);
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T heaviside(const T& a) {
    return (a > 0) ? 1 : 0;
}

// Computes the conjugate of a number (support for non-complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T conj(const T& a) {
    return a;
}

// Computes 2 raised to the given power n (support for complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> exp2(const std::complex<T>& n) {
    return std::exp(n * std::log(T(2)));
}

// Computes the base-2 logarithm of n (support for complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> log2(const std::complex<T>& n) {
    T radius = std::abs(n);
    T theta = std::arg(n);
    return std::complex<T>(std::log2(radius), theta / std::log(T(2)));
}

// Computes the e raised to the given power n, minus 1.0 (support for complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> expm1(const std::complex<T>& n) {
    return std::exp(n) - T(1);
}

// Computes the base-e logarithm of 1 + n (support for complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> log1p(const std::complex<T>& n) {
    return std::log(n + T(1));
}

// Computes the reciprocal of a number
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T reciprocal(const T& a) {
    return T(1) / a;
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI std::complex<T> reciprocal(const std::complex<T>& a) {
    return T(1) / a;
}

#if __cplusplus < 201703L

// Compute the greatest common divisor of two integers
template<typename T>
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
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T lcm(T a, T b) {
    // lcm(a, b) = |a * b| / gcd(a, b)
    // more efficient lcm(a, b) = (|a| / gcd(a, b)) * |b|
    if (a == 0 && b == 0) // special case
        return 0;
    return (abs(a) / gcd(a, b)) * abs(b);
}

#else

// Compute the greatest common divisor of two integers
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T gcd(const T& a, const T& b) {
    return std::gcd(a, b);
}

// Compute the least common multiple of two integers
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T lcm(const T& a, const T& b) {
    return std::lcm(a, b);
}

#endif

// Converts angles from degrees to radians
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T deg2rad(const T& a) {
    return a * M_PI / T(180);
}

// Converts angles from radians to degrees
template<typename T>
static DACE_CONSTEXPR DACE_HDFI T rad2deg(const T& a) {
    return a * T(180) / M_PI;
}

// Determines if the given (floating point) number has finite value
// (support for complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI bool isfinite(const std::complex<T>& a) {
    return std::isfinite(a.real()) && std::isfinite(a.imag());
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI bool isfinite(const T& a) {
    return std::isfinite(a);
}

// Determines if the given (floating point) number is a positive or negative
// infinity (support for complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI bool isinf(const std::complex<T>& a) {
    return std::isinf(a.real()) || std::isinf(a.imag());
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI bool isinf(const T& a) {
    return std::isinf(a);
}

// Determines if the given (floating point) number is not-a-number (NaN) value
// (support for complex numbers)
template<typename T>
static DACE_CONSTEXPR DACE_HDFI bool isnan(const std::complex<T>& a) {
    return std::isnan(a.real()) || std::isnan(a.imag());
}
template<typename T>
static DACE_CONSTEXPR DACE_HDFI bool isnan(const T& a) {
    return std::isnan(a);
}

// Determines if the given floating point number a is negative
template<typename T>
static DACE_CONSTEXPR DACE_HDFI bool signbit(const T& a) {
    return std::signbit(a);
}

// Computes modf (compatibility between Python tasklets and C++ modf)
template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void np_modf(const T& a, double& integral, double& fractional) {
    integral = double(a);
    fractional = double(0);
}
template<typename T, std::enable_if_t<!std::is_integral<T>::value && std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void np_modf(const T& a, T& integral, T& fractional) {
    fractional = std::modf(a, &integral);
}

// Computes frexp (compatibility between Python tasklets and C++ frexp)
template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
static DACE_CONSTEXPR DACE_HDFI void np_frexp(const T& a, T& mantissa, int& exponent) {
    mantissa = std::frexp(a, &exponent);
}


#endif

namespace dace
{
    namespace math
    {
        static DACE_CONSTEXPR_HOSTDEV typeless_pi pi{};
        static DACE_CONSTEXPR typeless_nan nan{};
        //////////////////////////////////////////////////////
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T exp(const T& a)
        {
            return (T)std::exp(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T exp2(const T& a)
        {
            return (T)std::exp2(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T expm1(const T& a)
        {
            return (T)std::expm1(a);
        }

#ifdef __CUDACC__
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI thrust::complex<T> pow(const thrust::complex<T>& a, const thrust::complex<T>& b)
        {
            return (thrust::complex<T>)thrust::pow(a, b);
        }
#endif
        // ``a ** b`` by multiplication. A scalar seeds at ``T(1)``, so ``ipow(a, 0) == 1``. A vector type has no
        // scalar constructor and seeds at ``a``, so it needs ``b >= 1`` (codegen emits a literal 1 for exponent 0).
        template<typename T>
        DACE_HDFI T ipow(const T a, const unsigned int b)
        {
            if constexpr (std::is_constructible<T, int>::value)
            {
                T result = T(1);
                for (unsigned int i = 0; i < b; ++i)
                    result *= a;
                return result;
            }
            else
            {
                T result = a;
                for (unsigned int i = 1; i < b; ++i)
                    result *= a;
                return result;
            }
        }

        // ``a ** b``. An integral exponent multiplies, and takes the reciprocal for a negative ``b``; an integral
        // base with a signed exponent gives a double, as the frontend types ``int ** signed int`` (``2 ** -1`` is
        // ``0.5``). Any other exponent falls back to ``std::pow``.
        template<typename T, typename U>
        DACE_CONSTEXPR DACE_HDFI auto pow(const T& a, const U& b)
        {
            if constexpr (std::is_integral<U>::value && std::is_constructible<T, int>::value)
            {
                using R = typename std::conditional<std::is_integral<T>::value && std::is_signed<U>::value, double,
                                                    T>::type;
                if constexpr (std::is_signed<U>::value)
                {
                    if (b < 0) return R(1) / ipow(R(a), 0u - static_cast<unsigned int>(b));
                }
                return ipow(R(a), static_cast<unsigned int>(b));
            }
            else
            {
                return std::pow(a, b);
            }
        }
#ifdef __CUDACC__
        // std::pow has no half overload, so a half argument makes the float/double/long double
        // overloads equally good (ambiguous).
                template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
        DACE_HDFI auto pow(const dace::float16& a, const T& b)
        {
            return std::pow(T(a), b);
        }
        template<typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
        DACE_HDFI auto pow(const T& a, const dace::float16& b)
        {
            return std::pow(a, float(b));
        }
        static DACE_HDFI dace::float16 pow(const dace::float16& a, const dace::float16& b)
        {
            return dace::float16(std::pow(float(a), float(b)));
        }
#endif

        template<typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
        DACE_CONSTEXPR DACE_HDFI T ifloor(const T& a)
        {
            return a;
        }

        template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
        DACE_CONSTEXPR DACE_HDFI int ifloor(const T& a)
        {
            return (int)std::floor(a);
        }

        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T sin(const T& a)
        {
            return std::sin(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T asin(const T& a)
        {
            return std::asin(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T sinh(const T& a)
        {
            return std::sinh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T asinh(const T& a)
        {
            return std::asinh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T cos(const T& a)
        {
            return std::cos(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T acos(const T& a)
        {
            return std::acos(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T cosh(const T& a)
        {
            return std::cosh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T acosh(const T& a)
        {
            return std::acosh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T tan(const T& a)
        {
            return std::tan(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T atan(const T& a)
        {
            return std::atan(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T atan2(const T& a, const T& b)
        {
            return std::atan2(a, b);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T tanh(const T& a)
        {
            return std::tanh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T atanh(const T& a)
        {
            return std::atanh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T sqrt(const T& a)
        {
            return std::sqrt(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T cbrt(const T& a)
        {
            return std::cbrt(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T log(const T& a)
        {
          return std::log(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T log10(const T& a)
        {
          return std::log10(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T log1p(const T& a)
        {
            return std::log1p(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T log2(const T& a)
        {
            return std::log2(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T fmod(const T& a, const T& b)
        {
            return std::fmod(a, b);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T lgamma(const T& a)
        {
            return std::lgamma(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T tgamma(const T& a)
        {
            return std::tgamma(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T ceil(const T& a)
        {
            return std::ceil(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T trunc(const T& a)
        {
            return std::trunc(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T erf(const T& a)
        {
            return std::erf(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T erfc(const T& a)
        {
            return std::erfc(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T nearbyint(const T& a)
        {
            return std::nearbyint(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T round(const T& a)
        {
            return std::round(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T hypot(const T& a, const T& b)
        {
            return std::hypot(a, b);
        }

#ifdef __CUDACC__
        // The templates above call ``std::NAME`` directly, whose float/double/long double overloads
        // tie for a half -- and codegen spells ``x ** 0.5`` as ``dace::math::sqrt``. These exact
        // matches run the float version and round back. ``exp`` and ``tanh`` have their native half
        // overloads in dace/cuda/halfvec.cuh.
#define DACE_MATH_UNARY_HALF(NAME)                                    \
        static DACE_HDFI dace::float16 NAME(const dace::float16& a)   \
        {                                                             \
            return dace::float16(std::NAME(float(a)));                \
        }
        DACE_MATH_UNARY_HALF(sqrt)
        DACE_MATH_UNARY_HALF(cbrt)
        DACE_MATH_UNARY_HALF(exp2)
        DACE_MATH_UNARY_HALF(expm1)
        DACE_MATH_UNARY_HALF(log)
        DACE_MATH_UNARY_HALF(log10)
        DACE_MATH_UNARY_HALF(log1p)
        DACE_MATH_UNARY_HALF(log2)
        DACE_MATH_UNARY_HALF(sin)
        DACE_MATH_UNARY_HALF(sinh)
        DACE_MATH_UNARY_HALF(cos)
        DACE_MATH_UNARY_HALF(cosh)
        DACE_MATH_UNARY_HALF(tan)
        DACE_MATH_UNARY_HALF(asin)
        DACE_MATH_UNARY_HALF(asinh)
        DACE_MATH_UNARY_HALF(acos)
        DACE_MATH_UNARY_HALF(acosh)
        DACE_MATH_UNARY_HALF(atan)
        DACE_MATH_UNARY_HALF(atanh)
        DACE_MATH_UNARY_HALF(lgamma)
        DACE_MATH_UNARY_HALF(tgamma)
        DACE_MATH_UNARY_HALF(ceil)
        DACE_MATH_UNARY_HALF(trunc)
        DACE_MATH_UNARY_HALF(erf)
        DACE_MATH_UNARY_HALF(erfc)
        DACE_MATH_UNARY_HALF(nearbyint)
        DACE_MATH_UNARY_HALF(round)
#undef DACE_MATH_UNARY_HALF
        // The binary ones take ``(const T&, const T&)``, so a half beside a built-in deduces no ``T``.
        // As with the operators, the half is promoted to float; half with half stays half.
#define DACE_MATH_BINARY_HALF(NAME)                                                        \
        static DACE_HDFI dace::float16 NAME(const dace::float16& a, const dace::float16& b) \
        {                                                                                  \
            return dace::float16(std::NAME(float(a), float(b)));                           \
        }                                                                                  \
        template<typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>    \
        DACE_HDFI auto NAME(const dace::float16& a, const T& b)                            \
        {                                                                                  \
            return std::NAME(float(a), b);                                                 \
        }                                                                                  \
        template<typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>    \
        DACE_HDFI auto NAME(const T& a, const dace::float16& b)                            \
        {                                                                                  \
            return std::NAME(a, float(b));                                                 \
        }
        DACE_MATH_BINARY_HALF(atan2)
        DACE_MATH_BINARY_HALF(fmod)
        DACE_MATH_BINARY_HALF(hypot)
#undef DACE_MATH_BINARY_HALF
#endif
    }

    namespace cmath
    {
        template<typename T>
        DACE_CONSTEXPR std::complex<T> exp(const std::complex<T>& a)
        {
            return std::exp(a);
        }

        #ifdef __CUDACC__
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI thrust::complex<T> exp(const thrust::complex<T>& a)
        {
            return thrust::exp(a);
        }
        #endif

        template<typename T>
        DACE_CONSTEXPR std::complex<T> conj(const std::complex<T>& a)
        {
            return std::conj(a);
        }

        #ifdef __CUDACC__
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI thrust::complex<T> conj(const thrust::complex<T>& a)
        {
            return thrust::conj(a);
        }
        #endif
    }

}

// Codegen emits the bare ``ipow`` in loop bounds and interstate edges, only for an exponent proven non-negative
using dace::math::ipow;

#endif  // __DACE_MATH_H
