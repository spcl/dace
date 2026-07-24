// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_TYPES_H
#define __DACE_TYPES_H

#include <cstdint>
#include <complex>

#ifdef _MSC_VER
    //#define DACE_ALIGN(N) __declspec( align(N) )
    #define DACE_ALIGN(N)
    #undef __in
    #undef __inout
    #undef __out
    #define DACE_EXPORTED extern "C" __declspec(dllexport)
    #define DACE_PRAGMA(x) __pragma(x)
#else
    #define DACE_ALIGN(N) __attribute__((aligned(N)))
    // Explicit default visibility: the C ABI must stay exported also when the
    // program is built as a nanobind module (compiled with -fvisibility=hidden).
    #define DACE_EXPORTED extern "C" __attribute__((visibility("default")))
    #define DACE_PRAGMA(x) _Pragma(#x)
#endif

// Visual Studio (<=2017) + CUDA support
#if defined(_MSC_VER) && _MSC_VER <= 1999
#define DACE_CONSTEXPR
#else
#define DACE_CONSTEXPR constexpr
#endif

// GPU support
#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <thrust/complex.h>
    #include "cuda/multidim_gbar.cuh"

    // Workaround so that half is defined as a scalar (for reductions)
    namespace std {
        template <>
        struct is_scalar<half> : std::integral_constant<bool, true> {};
        template <>
        struct is_fundamental<half> : std::integral_constant<bool, true> {};
    }  // namespace std
#elif defined(__HIPCC__)
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
    #define DACE_HDFI __host__ __device__ __forceinline__
    #define DACE_HFI __host__ __forceinline__
    #define DACE_DFI __device__ __forceinline__
#else
    #define DACE_HDFI inline
    #define DACE_HFI inline
    #define DACE_DFI inline
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
    #define __DACE_UNROLL DACE_PRAGMA(unroll)
#else
    #define __DACE_UNROLL
#endif

// If CUDA version is 11.4 or higher, __device__ variables can be declared as constexpr
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4))
    #define DACE_CONSTEXPR_HOSTDEV constexpr __host__ __device__
#elif defined(__CUDACC__) || defined(__HIPCC__)
    #define DACE_CONSTEXPR_HOSTDEV const __host__ __device__
#else
    #define DACE_CONSTEXPR_HOSTDEV const
#endif


namespace dace
{
    typedef bool bool_;
    typedef int8_t  int8;
    typedef int16_t int16;
    typedef int32_t int32;
    typedef int64_t int64;
    typedef uint8_t  uint8;
    typedef uint16_t uint16;
    typedef uint32_t uint32;
    typedef uint64_t uint64;
    typedef unsigned int uint;
    typedef float float32;
    typedef double float64;

    #ifdef __CUDACC__
    typedef thrust::complex<float> complex64;
    typedef thrust::complex<double> complex128;
    typedef half float16;
    #elif defined(__HIPCC__)
    typedef half float16;
    #else
    typedef std::complex<float> complex64;
    typedef std::complex<double> complex128;
    struct half {
        half() {}

        // IEEE-754 binary32 -> binary16, round-to-nearest-even.
        // Based on (https://gist.github.com/rygorous/2156668)
        half(float f) {
            union { float f; uint32_t u; } in;
            in.f = f;
            uint32_t sign = in.u & 0x80000000u;
            in.u ^= sign;  // work on the magnitude

            if (in.u >= 0x47800000u) {
                // Inf or NaN (exponent overflows half range): NaN -> qNaN, else Inf.
                h = (in.u > 0x7f800000u) ? 0x7e00 : 0x7c00;
            } else if (in.u < 0x38800000u) {
                // Subnormal half or zero. Adding this magic constant and relying on
                // round-to-nearest-even FP addition aligns the mantissa correctly.
                union { uint32_t u; float f; } magic;
                magic.u = 0x3f000000u;  // 126 << 23
                in.f += magic.f;
                h = (uint16_t)(in.u - magic.u);
            } else {
                // Normal half: rebias exponent (127 -> 15) and round mantissa to even.
                uint32_t mant_odd = (in.u >> 13) & 1;
                in.u += 0xc8000000u + 0xfffu;  // ((15 - 127) << 23) + rounding bias
                in.u += mant_odd;
                h = (uint16_t)(in.u >> 13);
            }
            h |= (uint16_t)(sign >> 16);
        }

        // IEEE-754 binary16 -> binary32 (exact; every binary16 fits in binary32).
        operator float() const {
            uint32_t sign = (uint32_t)(h & 0x8000) << 16;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            union { uint32_t u; float f; } out;
            if (exp == 0) {
                if (mant == 0) {
                    out.u = sign;  // signed zero
                } else {
                    // Subnormal half: normalize into a float normal.
                    exp = 1;
                    do { exp--; mant <<= 1; } while ((mant & 0x400) == 0);
                    mant &= 0x3ff;
                    out.u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                }
            } else if (exp == 0x1f) {
                out.u = sign | 0x7f800000u | (mant << 13);  // Inf or NaN
            } else {
                out.u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);  // normal
            }
            return out.f;
        }

        uint16_t h;
    };
    typedef half float16;
    #endif

    enum NumAccesses
    {
        NA_RUNTIME = 0, // Given at runtime
    };

    template <int DIM, int... OTHER_DIMS>
    struct TotalNDSize
    {
	enum
	{
	    value = DIM * TotalNDSize<OTHER_DIMS...>::value,
	};
    };

    template <int DIM>
    struct TotalNDSize<DIM>
    {
	enum
	{
	    value = DIM,
	};
    };

    // Mirror of dace.dtypes.ReductionType
    enum class ReductionType {
        Custom = 0,
        Min = 1,
        Max = 2,
        Sum = 3,
        Product = 4,
        Logical_And = 5,
        Bitwise_And = 6,
        Logical_Or = 7,
        Bitwise_Or = 8,
        Logical_Xor = 9,
        Bitwise_Xor = 10,
        Min_Location = 11,
        Max_Location = 12,
        Exchange = 13
    };
}

#endif  // __DACE_TYPES_H
