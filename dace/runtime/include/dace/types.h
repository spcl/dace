// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_TYPES_H
#define __DACE_TYPES_H

#include <complex>
#include <cstdint>

#ifdef _MSC_VER
// #define DACE_ALIGN(N) __declspec( align(N) )
#define DACE_ALIGN(N)
#undef __in
#undef __inout
#undef __out
#define DACE_EXPORTED extern "C" __declspec(dllexport)
#define DACE_PRAGMA(x) __pragma(x)
// A symbol is not exported from a DLL unless it is explicitly dllexport'ed, so
// "hidden" is already the default and the attribute has no MSVC equivalent.
#define DACE_HIDDEN
#else
#define DACE_ALIGN(N) __attribute__((aligned(N)))
#define DACE_EXPORTED extern "C"
#define DACE_PRAGMA(x) _Pragma(#x)
// Internal linkage-visibility for a definition that is shared ACROSS generated
// translation units but must not appear in the shared library's public ABI --
// specifically a nested-SDFG function emitted to its own .cpp under
// ``compiler.cpu.codegen_params.split_nsdfg_translation_units``. The function
// keeps EXTERNAL linkage (so the static linker resolves the call from the frame
// object), while hidden visibility keeps it out of the dynamic symbol table and
// lets the linker/ThinLTO re-inline it. Applied per-declaration on purpose: a
// global -fvisibility=hidden would also hide __dace_init_* / __dace_exit_* (only
// ``extern "C"`` via DACE_EXPORTED on Linux, not dllexport-annotated) and break
// loading the program.
#define DACE_HIDDEN __attribute__((visibility("hidden")))
#endif

// Portable full-unroll hint for fixed-width (constexpr-bounded) lane loops in
// vectorized intrinsics. Clang / NVCC accept a bare ``#pragma unroll``; GCC
// needs an explicit factor (64 covers every vector width we emit and fully
// unrolls any shorter constexpr-bounded loop); MSVC has no equivalent.
#if defined(__clang__) || defined(__CUDACC__) || defined(__INTEL_LLVM_COMPILER)
#define DACE_UNROLL DACE_PRAGMA(unroll)
#elif defined(__GNUC__)
#define DACE_UNROLL DACE_PRAGMA(GCC unroll 64)
#else
#define DACE_UNROLL
#endif

// Visual Studio (<=2017) + CUDA support
#if defined(_MSC_VER) && _MSC_VER <= 1999
#define DACE_CONSTEXPR
#else
#define DACE_CONSTEXPR constexpr
#endif

// GPU support
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
#include <hip/hip_fp16.h>
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

// If CUDA version is 11.4 or higher, __device__ variables can be declared as
// constexpr
#if defined(__CUDACC__) &&        \
    (__CUDACC_VER_MAJOR__ > 11 || \
     (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4))
#define DACE_CONSTEXPR_HOSTDEV constexpr __host__ __device__
#elif defined(__CUDACC__) || defined(__HIPCC__)
#define DACE_CONSTEXPR_HOSTDEV const __host__ __device__
#else
#define DACE_CONSTEXPR_HOSTDEV const
#endif

namespace dace {
typedef bool bool_;
typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
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
// Compile-time bit reinterpretation for the ``constexpr`` half<->float conversions
// below, so ``dace::float16(<constant>)`` folds at compile time exactly like a
// ``static_cast`` (equivalent to ``dace::float64(x)`` == ``double(x)`` for the
// primitive typedefs). ``std::bit_cast`` (C++20, the default standard) is a
// constant expression; pre-C++20 falls back to a runtime ``union`` pun -- identical
// bit pattern either way (little/big-endian agnostic: it copies object bytes).
#if defined(__cpp_lib_bit_cast)
#include <bit>
#define DACE_HALF_CE constexpr
static constexpr uint32_t _dace_half_f2u(float f) { return std::bit_cast<uint32_t>(f); }
static constexpr float _dace_half_u2f(uint32_t u) { return std::bit_cast<float>(u); }
#else
#define DACE_HALF_CE inline
static inline uint32_t _dace_half_f2u(float f) { union { float f; uint32_t u; } c; c.f = f; return c.u; }
static inline float _dace_half_u2f(uint32_t u) { union { uint32_t u; float f; } c; c.u = u; return c.f; }
#endif
struct half {
  constexpr half() : h(0) {}

  // IEEE-754 binary32 -> binary16, round-to-nearest-even.
  // Based on (https://gist.github.com/rygorous/2156668)
  DACE_HALF_CE half(float f) : h(0) {
    uint32_t u = _dace_half_f2u(f);
    uint32_t sign = u & 0x80000000u;
    u ^= sign;  // work on the magnitude

    if (u >= 0x47800000u) {
      // Inf or NaN (exponent overflows half range): NaN -> qNaN, else Inf.
      h = (u > 0x7f800000u) ? 0x7e00 : 0x7c00;
    } else if (u < 0x38800000u) {
      // Subnormal half or zero. Adding this magic constant and relying on
      // round-to-nearest-even FP addition aligns the mantissa correctly.
      float mf = _dace_half_u2f(u) + _dace_half_u2f(0x3f000000u);  // 126 << 23
      h = (uint16_t)(_dace_half_f2u(mf) - 0x3f000000u);
    } else {
      // Normal half: rebias exponent (127 -> 15) and round mantissa to even.
      uint32_t mant_odd = (u >> 13) & 1;
      u += 0xc8000000u + 0xfffu;  // ((15 - 127) << 23) + rounding bias
      u += mant_odd;
      h = (uint16_t)(u >> 13);
    }
    h |= (uint16_t)(sign >> 16);
  }

  // IEEE-754 binary16 -> binary32 (exact; every binary16 fits in binary32).
  DACE_HALF_CE operator float() const {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t out;
    if (exp == 0) {
      if (mant == 0) {
        out = sign;  // signed zero
      } else {
        // Subnormal half: normalize into a float normal.
        exp = 1;
        do {
          exp--;
          mant <<= 1;
        } while ((mant & 0x400) == 0);
        mant &= 0x3ff;
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
      }
    } else if (exp == 0x1f) {
      out = sign | 0x7f800000u | (mant << 13);  // Inf or NaN
    } else {
      out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);  // normal
    }
    return _dace_half_u2f(out);
  }

  uint16_t h;
};
typedef half float16;
#endif

enum NumAccesses {
  NA_RUNTIME = 0,  // Given at runtime
};

template <int DIM, int... OTHER_DIMS>
struct TotalNDSize {
  enum {
    value = DIM * TotalNDSize<OTHER_DIMS...>::value,
  };
};

template <int DIM>
struct TotalNDSize<DIM> {
  enum {
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
}  // namespace dace

#endif  // __DACE_TYPES_H
