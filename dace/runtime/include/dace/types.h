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

// ---------------------------------------------------------------------------
// Native binary16 backing type for the float <-> half CONVERSIONS.
//
// ``dace::half`` stays a class whose arithmetic is evaluated in ``float`` (see
// the operator section below): that is what the emulation has always done, so
// results are unchanged. The only thing the native type replaces is the two
// conversion routines, turning ~15 integer ALU ops into a single hardware
// instruction. Verified bit-identical to the emulation for every one of the
// 2^32 float bit patterns and all 2^16 half bit patterns, NaN payloads excepted
// (the hardware quiets NaNs per IEEE-754; the emulation did not) --
// see tests/cpp/half_ops_test.cpp.
//
// We only switch where the conversion is a real instruction. ``_Float16`` is
// *available* far more widely than it is *fast*: on a plain x86-64 target
// without F16C, GCC/Clang lower every conversion to a libgcc/compiler-rt call
// (``__extendhfsf2`` / ``__truncsfhf2``), which is slower than the inline
// emulation. So gate on the ISA feature macro, never on mere type availability.
//
//   x86:     __F16C__ (VCVTPH2PS/VCVTPS2PH, Ivy Bridge+) or __AVX512FP16__.
//   AArch64: FCVT between half and single is base ARMv8-A, so IEEE-format
//            _Float16 conversions are always single instructions.
//   ARM32:   stays on the emulation. The conversion needs VFPv3-fp16/VFPv4 and
//            there is no way to test it here; the emulation is always correct.
//
// Two hazards that normally make ``_Float16`` risky do NOT apply here, because
// it is confined to these two functions and never appears in a type, a struct
// member or a function signature:
//
//  * Excess precision. GCC and Clang both evaluate ``_Float16`` *arithmetic* in
//    float and only round at explicit casts and assignments unless you pass
//    -fexcess-precision=16 / -ffloat16-excess-precision=none, so ``a*b+c`` in
//    _Float16 rounds once, not three times. We never do arithmetic in the type
//    -- each function is a single explicit conversion -- so the rounding is
//    unambiguous and no extra build flag is needed.
//  * ABI. ``_Float16`` is passed in an SSE/FP register, is classified as an HFA
//    inside aggregates on AArch64, and mangles as ``DF16_``. ``dace::half``
//    remains ``struct half { uint16_t h; }``, so calling convention, aggregate
//    classification and C++ mangling of every generated symbol are untouched.
//
// Gating macros. NOT ``__FLT16_MAX__``: GCC defines the ``__FLT16_*`` macros
// even on targets where ``_Float16`` is storage-only and every use is a hard
// error (verified: ``g++ -m32 -mno-sse2`` defines 16 of them), which is why GCC
// 14's release notes tell you to test ``__SSE2__`` instead. x86 ``_Float16``
// arrived in GCC 12 and Clang 15, hence the version guards.
// ``__ARM_FP16_FORMAT_IEEE`` is the ACLE macro saying binary16 rather than the
// Arm "alternative" format, which has no infinities and would not be
// bit-compatible with the emulation; AArch64 only ever uses the IEEE format.
// AArch64 needs no arithmetic-feature check: FCVT between half and single is
// base ARMv8-A, so the conversion is one instruction even without FEAT_FP16
// (which is OPTIONAL from ARMv8.2 and is not mandatory in any ARMv8/v9 level --
// ``__ARM_FEATURE_FP16_SCALAR_ARITHMETIC`` is the only sound way to detect it,
// and it would only matter if we did arithmetic in the type, which we do not).
// 32-bit ARM deliberately stays on the emulation: it is untested here.
#if !defined(DACE_HALF_NO_NATIVE)
#if (defined(__x86_64__) || defined(__i386__)) && defined(__SSE2__) &&           \
    (defined(__F16C__) || defined(__AVX512FP16__)) &&                           \
    ((defined(__clang__) && __clang_major__ >= 15) ||                           \
     (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 12))
#define DACE_HALF_NATIVE_T _Float16
#elif defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE) &&                 \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 7))
#define DACE_HALF_NATIVE_T _Float16
#endif
#endif

#if defined(DACE_HALF_NATIVE_T)
// A 2-byte native type bit-cast to/from uint16_t: no reinterpretation of the
// value, so the stored representation is the same IEEE binary16 either way.
DACE_HALF_CE uint16_t dace_half_from_float(float f) {
#if defined(__cpp_lib_bit_cast)
  return std::bit_cast<uint16_t>((DACE_HALF_NATIVE_T)f);
#else
  union { DACE_HALF_NATIVE_T n; uint16_t u; } c;
  c.n = (DACE_HALF_NATIVE_T)f;
  return c.u;
#endif
}
DACE_HALF_CE float dace_half_to_float(uint16_t h) {
#if defined(__cpp_lib_bit_cast)
  return (float)std::bit_cast<DACE_HALF_NATIVE_T>(h);
#else
  union { uint16_t u; DACE_HALF_NATIVE_T n; } c;
  c.u = h;
  return (float)c.n;
#endif
}
#else
// IEEE-754 binary32 -> binary16, round-to-nearest-even.
// Based on (https://gist.github.com/rygorous/2156668)
DACE_HALF_CE uint16_t dace_half_from_float(float f) {
  uint32_t u = _dace_half_f2u(f);
  uint32_t sign = u & 0x80000000u;
  u ^= sign;  // work on the magnitude
  uint16_t h;

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
  return (uint16_t)(h | (uint16_t)(sign >> 16));
}

// IEEE-754 binary16 -> binary32 (exact; every binary16 fits in binary32).
DACE_HALF_CE float dace_half_to_float(uint16_t h) {
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
#endif

struct half {
  constexpr half() : h(0) {}
  DACE_HALF_CE half(float f) : h(dace_half_from_float(f)) {}
  DACE_HALF_CE operator float() const { return dace_half_to_float(h); }

  // Compound assignment. Binary arithmetic, comparisons and unary minus already
  // work through the implicit ``operator float()`` and are deliberately NOT
  // overloaded here: an ``operator+(half, half)`` would tie with the built-in
  // ``operator+(float, float)`` for a mixed expression such as ``h + 1.0f``
  // (one user-defined conversion either way) and make it ambiguous. Compound
  // assignment is the one part of the surface that conversion cannot supply,
  // because the built-in ``operator+=`` needs an lvalue of arithmetic type and
  // ``operator float()`` yields a prvalue.
  //
  // A single ``float`` parameter covers half/float/double/integer right-hand
  // sides via one implicit conversion each, so there is exactly one candidate
  // and no ambiguity. The value semantics are ``h = half(float(h) OP x)`` --
  // the same convert-out / compute-in-float / round-back the emulation has
  // always given for ``h = h OP x``.
#define DACE_HALF_COMPOUND(OP)                                     \
  DACE_HALF_CE half &operator OP##=(float f) {                     \
    *this = half((float)*this OP f);                               \
    return *this;                                                  \
  }
  DACE_HALF_COMPOUND(+)
  DACE_HALF_COMPOUND(-)
  DACE_HALF_COMPOUND(*)
  DACE_HALF_COMPOUND(/)
#undef DACE_HALF_COMPOUND

  DACE_HALF_CE half &operator++() { return *this += 1.0f; }
  DACE_HALF_CE half &operator--() { return *this -= 1.0f; }
  DACE_HALF_CE half operator++(int) { half t = *this; *this += 1.0f; return t; }
  DACE_HALF_CE half operator--(int) { half t = *this; *this -= 1.0f; return t; }

  uint16_t h;
};
typedef half float16;

#ifdef _OPENMP
// OpenMP has no built-in reduction over a class type: ``reduction(+: x)`` on a
// ``dace::half`` accumulator is rejected with "user defined reduction not
// found" even once ``operator+=`` exists, because the clause needs an identity
// and a combiner it cannot synthesize for a non-arithmetic type. Declaring them
// here -- rather than in the CPU codegen -- means the declaration follows
// whichever of the two ``half`` implementations above is in effect, and every
// generated translation unit that includes <dace/types.h> gets it.
//
// The identities mirror OpenMP's built-in floating-point ones (0, 1, +inf,
// -inf), so a thread whose chunk is empty contributes nothing. ``omp_out`` is
// combined with ``+=`` / ``*=`` (defined above) and min/max through the
// implicit float comparison.
//
// ``-`` is declared because ExpandReduceOpenMP emits ``reduction(-: ...)`` for
// a Sub node and OpenMP accepts it for the primitive float types. Its combiner
// is ``+=``, NOT ``-=`` -- do not "fix" this. The clause negates within each
// private copy and the copies are then summed, which is what OpenMP defines the
// ``-`` reduction to mean and what GCC does for ``float``: the answer is
// ``initial - sum(x)``, and a ``-=`` combiner would flip the sign of every
// thread's contribution but the first.
//
// The set is exactly the one OpenMP accepts for ``float`` -- no more, no less.
// ``&``, ``|`` and ``^`` are deliberately absent: OpenMP rejects them for the
// primitive floating-point types too ("user defined reduction not found for
// 'float'"), and C++ has no bitwise operator on a floating-point operand.
// Declaring them for ``half`` would make float16 accept a program that fails to
// compile for float32, and would silently give it bit-pattern semantics that no
// other float type has. ExpandReduceOpenMP refuses those combinations up front
// instead.
#pragma omp declare reduction(+ : dace::half : omp_out += omp_in) initializer(omp_priv = dace::half(0.0f))
#pragma omp declare reduction(- : dace::half : omp_out += omp_in) initializer(omp_priv = dace::half(0.0f))
#pragma omp declare reduction(* : dace::half : omp_out *= omp_in) initializer(omp_priv = dace::half(1.0f))
#pragma omp declare reduction(min : dace::half : omp_out = (float)omp_in < (float)omp_out ? omp_in : omp_out) \
    initializer(omp_priv = dace::half(__builtin_huge_valf()))
#pragma omp declare reduction(max : dace::half : omp_out = (float)omp_in > (float)omp_out ? omp_in : omp_out) \
    initializer(omp_priv = dace::half(-__builtin_huge_valf()))
// Logical reductions operate on truthiness and normalize the result to exactly
// 1.0 or 0.0, because ``a && b`` yields ``bool`` and the assignment converts it
// back -- the same normalization ``float`` gets, where a truthy 5.0f also
// reduces to 1.0f. Identities are the neutral truth values: true for &&,
// false for ||.
#pragma omp declare reduction(&& : dace::half : omp_out = dace::half((float)((float)omp_out && (float)omp_in))) \
    initializer(omp_priv = dace::half(1.0f))
#pragma omp declare reduction(|| : dace::half : omp_out = dace::half((float)((float)omp_out || (float)omp_in))) \
    initializer(omp_priv = dace::half(0.0f))
#endif
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
