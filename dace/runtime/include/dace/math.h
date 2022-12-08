// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_MATH_H
#define __DACE_MATH_H

#include "pi.h"
#include "types.h"

#include <complex>
#include <numeric>
#include <cmath>
#include <cfloat>
#include <type_traits>

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
//          will be implicitly converted to it in min/max.
//        * half comparisons are designated "device-only", so they must call
//          device-only functions as well.
#ifdef __CUDACC__
template <typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 min(const dace::float16& a, const dace::float16& b, const Ts&... c)
{
    return (a < b) ? min(a, c...) : min(b, c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 min(const dace::float16& a, const T& b, const Ts&... c)
{
    return (a < dace::float16(b)) ? min(a, c...) : min(dace::float16(b), c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 min(const T& a, const dace::float16& b, const Ts&... c)
{
    return (dace::float16(a) < b) ? min(dace::float16(a), c...) : min(b, c...);
}
template <typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 max(const dace::float16& a, const dace::float16& b, const Ts&... c)
{
    return (a > b) ? max(a, c...) : max(b, c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 max(const dace::float16& a, const T& b, const Ts&... c)
{
    return (a > dace::float16(b)) ? max(a, c...) : max(dace::float16(b), c...);
}
template <typename T, typename... Ts>
DACE_CONSTEXPR __device__ __forceinline__ dace::float16 max(const T& a, const dace::float16& b, const Ts&... c)
{
    return (dace::float16(a) > b) ? max(dace::float16(a), c...) : max(b, c...);
}
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
// which are equaivalent to the IEEE remainder: num - round(num / den) * den
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

// Compute the greates common divisor of two integers
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

// Compute the greates common divisor of two integers
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
        static DACE_CONSTEXPR typeless_pi pi{};
        //////////////////////////////////////////////////////
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T exp(const T& a)
        {
            return (T)std::exp(a);
        }

#ifdef __CUDACC__
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI thrust::complex<T> pow(const thrust::complex<T>& a, const thrust::complex<T>& b)
        {
            return (thrust::complex<T>)thrust::pow(a, b);
        }
#endif
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T pow(const T& a, const T& b)
        {
            return (T)std::pow(a, b);
        }

#ifndef DACE_XILINX
        static DACE_CONSTEXPR DACE_HDFI int pow(const int& a, const int& b)
        {
/*#ifndef __CUDA_ARCH__
            return std::pow(a, b);
#else*/
            if (b < 0) return 0;
            int result = 1;
            for (int i = 0; i < b; ++i)
                result *= a;
            return result;
//#endif
        }
        static DACE_CONSTEXPR DACE_HDFI unsigned int pow(const unsigned int& a,
                                       const unsigned int& b)
        {
/*#ifndef __CUDA_ARCH__
            return std::pow(a, b);
#else*/
            unsigned int result = 1;
            for (unsigned int i = 0; i < b; ++i)
                result *= a;
            return result;
//#endif
        }
#endif

        template<typename T>
        DACE_HDFI T ipow(const T& a, const unsigned int& b) {
            T result = a;
            for (unsigned int i = 1; i < b; ++i)
                result *= a;
            return result;
        }

        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T pow(const T& a, const int& b)
        {
            return (T)std::pow(a, (T)b);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T pow(const T& a, const unsigned int& b)
        {
            return (T)std::pow(a, (T)b);
        }

        template<typename T>
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
        DACE_CONSTEXPR DACE_HDFI T sinh(const T& a)
        {
            return std::sinh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T cos(const T& a)
        {
            return std::cos(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T cosh(const T& a)
        {
            return std::cosh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T tan(const T& a)
        {
            return std::tan(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T tanh(const T& a)
        {
            return std::tanh(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T sqrt(const T& a)
        {
            return std::sqrt(a);
        }
        template<typename T>
        DACE_CONSTEXPR DACE_HDFI T log(const T& a)
        {
          return std::log(a);
        }
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


#endif  // __DACE_MATH_H
