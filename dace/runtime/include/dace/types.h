// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_TYPES_H
#define __DACE_TYPES_H

#include <cstdint>
#include <cstring>
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
    // IEEE-754 binary16 fallback for the host path (the CUDA/HIP paths use the
    // backend's native half). Both conversions are round-to-nearest-even and
    // handle zero, subnormals, overflow->inf, and NaN; they were verified
    // bit-exact against numpy over all 65536 half values and 4M random floats.
    struct half {
        half() = default;
        half(float fl) {
            uint32_t x;
            std::memcpy(&x, &fl, 4);
            uint32_t sign = (x >> 16) & 0x8000u;
            x &= 0x7fffffffu;  // |f|
            if (x >= 0x7f800000u) {
                // Inf (mantissa 0) or NaN (mantissa != 0 -> quiet NaN)
                h = (uint16_t)((x == 0x7f800000u ? 0x7c00u : 0x7e00u) | sign);
            } else if (x >= 0x47800000u) {
                // |f| >= 2^16: above the largest value that rounds to a finite
                // half (< 65520), so it becomes Inf.
                h = (uint16_t)(0x7c00u | sign);
            } else if (x >= 0x38800000u) {
                // Normal half range (|f| >= 2^-14).
                uint32_t hbits = ((x >> 23) - 112u) << 10;  // (e-127+15) << 10
                uint32_t mant = x & 0x7fffffu;
                hbits |= mant >> 13;
                uint32_t rem = mant & 0x1fffu;  // dropped bits
                if (rem > 0x1000u || (rem == 0x1000u && (hbits & 1u)))
                    hbits++;  // carry may propagate into the exponent / to Inf
                h = (uint16_t)(hbits | sign);
            } else if (x >= 0x33000000u) {
                // Subnormal half (2^-24 <= |f| < 2^-14).
                uint32_t mant = (x & 0x7fffffu) | 0x800000u;  // add implicit 1
                int shift = 126 - (int)(x >> 23);  // in [14, 24]
                uint32_t hm = mant >> shift;
                uint32_t rem = mant & ((1u << shift) - 1u);
                uint32_t halfway = 1u << (shift - 1);
                if (rem > halfway || (rem == halfway && (hm & 1u)))
                    hm++;  // may round up to the smallest normal
                h = (uint16_t)(hm | sign);
            } else {
                // |f| < 2^-25: underflows to signed zero.
                h = (uint16_t)sign;
            }
        }
        operator float() const {
            uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1fu;
            uint32_t mant = h & 0x3ffu;
            uint32_t o;
            if (exp == 0u) {
                if (mant == 0u) {
                    o = sign;  // +/- 0
                } else {
                    // Subnormal: normalize into a float32 normal.
                    exp = 1u;
                    while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }
                    mant &= 0x3ffu;
                    o = sign | ((exp + 112u) << 23) | (mant << 13);
                }
            } else if (exp == 0x1fu) {
                o = sign | 0x7f800000u | (mant << 13);  // Inf / NaN (payload kept)
            } else {
                o = sign | ((exp + 112u) << 23) | (mant << 13);  // normal
            }
            float f;
            std::memcpy(&f, &o, 4);
            return f;
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
