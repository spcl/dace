// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
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
    #define DACE_EXPORTED extern "C"
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
    #include <thrust/complex.h>
    #include "../../../external/cub/cub/grid/grid_barrier.cuh"

    // Workaround so that half is defined as a scalar (for reductions)
    namespace std {
        template <>
        struct is_scalar<half> : std::integral_constant<bool, true> {};
    }  // namespace std
#elif defined(__HIPCC__)
    #include <hip/hip_runtime.h>
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



namespace dace
{
    typedef int8_t  int8;
    typedef int16_t int16;
    typedef int32_t int32;
    typedef int64_t int64;
    typedef uint8_t  uint8;
    typedef uint16_t uint16;
    typedef uint32_t uint32;
    typedef uint64_t uint64;
    typedef float float32;
    typedef double float64;

    #ifdef __CUDACC__
    typedef thrust::complex<float> complex64;
    typedef thrust::complex<double> complex128;
    typedef half float16;
    #elif defined(__HIPCC__)
    // Not fully supported
    typedef short float16;
    #else
    typedef std::complex<float> complex64;
    typedef std::complex<double> complex128;
    typedef short half;
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
