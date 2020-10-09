// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_MATH_H
#define __DACE_MATH_H

#include "pi.h"
#include "types.h"

#include <complex>
#include <numeric>
#include <cfloat>

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
DACE_CONSTEXPR DACE_HDFI T min(const T& a, const T& b, const Ts&... c)
{
    return (a < b) ? min(a, c...) : min(b, c...);
}

template <typename T>
DACE_CONSTEXPR DACE_HDFI T max(const T& val)
{
    return val;
}
template <typename T, typename... Ts>
DACE_CONSTEXPR DACE_HDFI T max(const T& a, const T& b, const Ts&... c)
{
    return (a > b) ? max(a, c...) : max(b, c...);
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
// Assuming inputs are of integer type.
template <typename T>
static DACE_CONSTEXPR DACE_HDFI T int_floor_ni(const T& numerator, const T& denominator) {
    T quotient = numerator / denominator;
    T remainder = numerator % denominator;
    // This doesn't work properly if both numbers have sign 0.
    // However, in this case we crash due to division by 0.
    if (sgn(numerator) + sgn(denominator) == 0 && remainder > 0)
        return quotient - 1;
    return quotient;
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
