// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_TYPES_H
#define __DACE_TYPES_H

#include <complex>
#include <cstdint>
#include <type_traits>

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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "cuda/multidim_gbar.cuh"

// Workaround so that half and __nv_bfloat16 are defined as scalars (for reductions)
namespace std {
template <>
struct is_scalar<half> : std::integral_constant<bool, true> {};
template <>
struct is_fundamental<half> : std::integral_constant<bool, true> {};
template <>
struct is_scalar<__nv_bfloat16> : std::integral_constant<bool, true> {};
template <>
struct is_fundamental<__nv_bfloat16> : std::integral_constant<bool, true> {};
}  // namespace std
#elif defined(__HIPCC__)
// <hip/hip_bf16.h> defines __hip_bfloat16, the CURRENT ROCm bfloat16 -- deliberately
// NOT the legacy <hip/hip_bfloat16.h> / hip_bfloat16 (no leading underscores). The two
// are distinct types with no implicit conversion between them, different HIP APIs
// expect different ones, and mixing them in one translation unit is a known build
// breakage (ROCm/ROCm#2534). __hip_bfloat16 mirrors CUDA's __nv_bfloat16 almost
// exactly -- same header role, same __float2bfloat16 / __bfloat162float spellings --
// which is why a single ``dace::bfloat16`` typedef covers both vendors.
//
// Probed with __has_include because the header only appeared in ROCm 5.7; on an
// older ROCm the build keeps working and simply has no dace::bfloat16, which
// DACE_HAS_BFLOAT16 reports.
#if __has_include(<hip/hip_bf16.h>)
#include <hip/hip_bf16.h>
#define DACE_HIP_HAS_BF16 1
#endif
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


// ---------------------------------------------------------------------------
// Layout contract for the 16-bit low-precision types.
//
// The host side of a GPU program does data COPY, not arithmetic: dace::float16 /
// dace::bfloat16 in a .cpp translation unit and the vendor types they become in a
// .cu / .hip one must therefore be byte-for-byte interchangeable, or every
// host<->device transfer silently corrupts. Asserting it here means any future edit
// that adds a member, a vtable or an alignment attribute fails the BUILD instead.
#define DACE_ASSERT_LOWP_LAYOUT(T)                                                    \
  static_assert(sizeof(T) == 2, #T " must be exactly 2 bytes");                       \
  static_assert(alignof(T) == 2, #T " must have 2-byte alignment");                   \
  static_assert(std::is_standard_layout<T>::value, #T " must be standard layout");    \
  static_assert(std::is_trivially_copyable<T>::value, #T " must be trivially copyable")

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
// Native bfloat16. ``__nv_bfloat16`` has hardware arithmetic from sm_80 (Ampere);
// below that, cuda_bf16.h still supplies the full operator surface by evaluating
// each operation in float.
//
// That per-operation float evaluation agrees with the host emulation for a SINGLE
// operation, and only for one: a fused expression does NOT agree. The host
// computes ``a * b + c`` entirely in float and rounds once, whereas per-operation
// device arithmetic rounds the product to bf16 before the add -- two roundings,
// and the results genuinely differ. This is the ordinary consequence of doing
// arithmetic in a narrow type and is not specific to bf16; it is called out here
// only so the equivalence is not over-read. The layout, asserted below, is what
// host<->device copies actually depend on, and that IS identical.
typedef __nv_bfloat16 bfloat16;
#define DACE_HAS_BFLOAT16 1
DACE_ASSERT_LOWP_LAYOUT(float16);
DACE_ASSERT_LOWP_LAYOUT(bfloat16);
#elif defined(__HIPCC__)
typedef half float16;
// AMD's native bfloat16, the counterpart of __nv_bfloat16 above: 2 bytes of
// unsigned short with the standard bf16 layout, a float constructor and an
// operator float(), so it satisfies the same contract as the CUDA and host types.
// UNTESTED AT RUNTIME -- there is no ROCm toolchain on the machine this was
// developed on, so this branch is written from the ROCm documentation and has not
// been compiled or executed.
#ifdef DACE_HIP_HAS_BF16
typedef __hip_bfloat16 bfloat16;
#define DACE_HAS_BFLOAT16 1
DACE_ASSERT_LOWP_LAYOUT(bfloat16);
#endif
DACE_ASSERT_LOWP_LAYOUT(float16);
#else
typedef std::complex<float> complex64;
typedef std::complex<double> complex128;
// Compile-time bit reinterpretation for the ``constexpr`` conversions between
// ``float`` and the 16-bit low-precision types below, so ``dace::float16(<constant>)``
// and ``dace::bfloat16(<constant>)`` fold at compile time exactly like a
// ``static_cast`` (equivalent to ``dace::float64(x)`` == ``double(x)`` for the
// primitive typedefs). ``std::bit_cast`` (C++20, the default standard) is a
// constant expression; pre-C++20 falls back to a runtime ``union`` pun -- identical
// bit pattern either way (little/big-endian agnostic: it copies object bytes).
#if defined(__cpp_lib_bit_cast)
#include <bit>
#define DACE_LOWP_CE constexpr
static constexpr uint32_t dace_bits_f2u(float f) { return std::bit_cast<uint32_t>(f); }
static constexpr float dace_bits_u2f(uint32_t u) { return std::bit_cast<float>(u); }
#else
#define DACE_LOWP_CE inline
static inline uint32_t dace_bits_f2u(float f) { union { float f; uint32_t u; } c; c.f = f; return c.u; }
static inline float dace_bits_u2f(uint32_t u) { union { uint32_t u; float f; } c; c.u = u; return c.f; }
#endif

// ---------------------------------------------------------------------------
// Is a native 16-bit floating-point type available? One gate, one answer, used by
// both the half and the bfloat16 sections below.
//
// A compiler VERSION FLOOR first, the compiler's own feature test second. Neither
// alone is sound:
//
//  * Version floor alone is not enough: x86 GCC 12 has no ``__bf16`` at all
//    ("unknown type name"), so a build could pass the floor and still lack the type.
//  * Feature test alone is not enough, and this is the subtle one: on AArch64 GCC
//    defines ``__bf16`` -- and therefore ``__BFLT16_MAX__`` -- unconditionally, but
//    the type was STORAGE-ONLY there until GCC 13. A feature test alone would report
//    "native bf16 available" on a compiler that rejects every use of it. The same
//    class of trap applies to ``__FLT16_MAX__``, which GCC defines even on targets
//    where ``_Float16`` is storage-only (``-m32 -mno-sse2``); the per-type ISA checks
//    further down are what cover that residue.
//
// Floors are GCC 15 and Clang 20 -- comfortably past GCC 13 (where AArch64 ``__bf16``
// became a real arithmetic type) and Clang 17 (where ``__bf16`` stopped being
// storage-only on every target). Deliberately NOT a probe of ``__ARM_FEATURE_BF16``:
// it was not predefined before GCC 15 but the fix was backported to 12.5, 13.4 and
// 14.2, so a version test on it is wrong in both directions. Not
// ``__ARM_BF16_FORMAT_ALTERNATIVE`` either -- that is the ARM32 port's macro, and
// GCC on AArch64 provides ``__bf16`` without ever defining it.
//
// Clang defines ``__GNUC__`` too, so its branch MUST come first. Clang also defines
// no bf16 macro of any kind, which is why the bf16 probe there is ``__is_identifier``
// rather than a ``__BFLT16_*`` test that would silently fail closed.
//
// NOTE the native type is used for STORAGE AND CONVERSION ONLY, never for
// arithmetic: every operator on dace::half / dace::bfloat16 evaluates in ``float``.
// That is what makes the storage-only-vs-arithmetic distinction above merely a
// build-safety question rather than a semantic one, and it matches the hardware --
// there is no scalar bf16 ALU operation on any shipping x86 or AArch64 CPU.
#if defined(__clang__)
#if __clang_major__ >= 20
#if defined(__FLT16_MAX__)
#define DACE_NATIVE_FP16 1
#endif
#if defined(__is_identifier) && !__is_identifier(__bf16)
#define DACE_NATIVE_BF16 1
#endif
#endif
#elif defined(__GNUC__)
#if __GNUC__ >= 15
#if defined(__FLT16_MAX__)
#define DACE_NATIVE_FP16 1
#endif
#if defined(__BFLT16_MAX__)
#define DACE_NATIVE_BF16 1
#endif
#endif
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
// ``__ARM_FP16_FORMAT_IEEE`` is the ACLE macro saying binary16 rather than the
// Arm "alternative" format, which has no infinities and would not be
// bit-compatible with the emulation; AArch64 only ever uses the IEEE format.
// AArch64 needs no arithmetic-feature check: FCVT between half and single is
// base ARMv8-A, so the conversion is one instruction even without FEAT_FP16
// (which is OPTIONAL from ARMv8.2 and is not mandatory in any ARMv8/v9 level --
// ``__ARM_FEATURE_FP16_SCALAR_ARITHMETIC`` is the only sound way to detect it,
// and it would only matter if we did arithmetic in the type, which we do not).
// 32-bit ARM deliberately stays on the emulation: it is untested here.
#if defined(DACE_NATIVE_FP16) && !defined(DACE_HALF_NO_NATIVE)
#if (defined(__x86_64__) || defined(__i386__)) && defined(__SSE2__) &&           \
    (defined(__F16C__) || defined(__AVX512FP16__))
#define DACE_HALF_NATIVE_T _Float16
#elif defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE)
#define DACE_HALF_NATIVE_T _Float16
#endif
#endif

#if defined(DACE_HALF_NATIVE_T)
// A 2-byte native type bit-cast to/from uint16_t: no reinterpretation of the
// value, so the stored representation is the same IEEE binary16 either way.
DACE_LOWP_CE uint16_t dace_half_from_float(float f) {
#if defined(__cpp_lib_bit_cast)
  return std::bit_cast<uint16_t>((DACE_HALF_NATIVE_T)f);
#else
  union { DACE_HALF_NATIVE_T n; uint16_t u; } c;
  c.n = (DACE_HALF_NATIVE_T)f;
  return c.u;
#endif
}
DACE_LOWP_CE float dace_half_to_float(uint16_t h) {
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
DACE_LOWP_CE uint16_t dace_half_from_float(float f) {
  uint32_t u = dace_bits_f2u(f);
  uint32_t sign = u & 0x80000000u;
  u ^= sign;  // work on the magnitude
  uint16_t h;

  if (u >= 0x47800000u) {
    // Inf or NaN (exponent overflows half range): NaN -> qNaN, else Inf.
    h = (u > 0x7f800000u) ? 0x7e00 : 0x7c00;
  } else if (u < 0x38800000u) {
    // Subnormal half or zero. Adding this magic constant and relying on
    // round-to-nearest-even FP addition aligns the mantissa correctly.
    float mf = dace_bits_u2f(u) + dace_bits_u2f(0x3f000000u);  // 126 << 23
    h = (uint16_t)(dace_bits_f2u(mf) - 0x3f000000u);
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
DACE_LOWP_CE float dace_half_to_float(uint16_t h) {
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
  return dace_bits_u2f(out);
}
#endif

struct alignas(2) half {
  constexpr half() : h(0) {}
  DACE_LOWP_CE half(float f) : h(dace_half_from_float(f)) {}
  DACE_LOWP_CE operator float() const { return dace_half_to_float(h); }

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
  DACE_LOWP_CE half &operator OP##=(float f) {                     \
    *this = half((float)*this OP f);                               \
    return *this;                                                  \
  }
  DACE_HALF_COMPOUND(+)
  DACE_HALF_COMPOUND(-)
  DACE_HALF_COMPOUND(*)
  DACE_HALF_COMPOUND(/)
#undef DACE_HALF_COMPOUND

  DACE_LOWP_CE half &operator++() { return *this += 1.0f; }
  DACE_LOWP_CE half &operator--() { return *this -= 1.0f; }
  DACE_LOWP_CE half operator++(int) { half t = *this; *this += 1.0f; return t; }
  DACE_LOWP_CE half operator--(int) { half t = *this; *this -= 1.0f; return t; }

  uint16_t h;
};
typedef half float16;
DACE_ASSERT_LOWP_LAYOUT(float16);

// ---------------------------------------------------------------------------
// bfloat16: the leading 16 bits of an IEEE-754 binary32. Same 8-bit exponent as
// float, mantissa truncated from 23 explicit bits to 7.
//
// The conversions are emulated BY DEFAULT even where a native ``__bf16`` exists,
// which is the opposite of the choice made for ``half`` above. That is a measured
// decision, not an oversight:
//
//  * bf16 -> float is a 16-bit left shift. Exact for every one of the 2^16 bit
//    patterns, NaN payloads and subnormals included, with no branch at all.
//  * float -> bf16 is one add (the round-to-nearest-even bias) and one shift,
//    plus a NaN test. There is no exponent rebias and no subnormal path, because
//    the exponent ranges are identical -- exactly what makes binary16's
//    conversion expensive and bf16's nearly free.
//
// There is no scalar bf16 arithmetic or conversion instruction to switch to on
// any shipping CPU. AVX512-BF16 is three instructions (VCVTNE2PS2BF16,
// VCVTNEPS2BF16, VDPBF16PS) -- packed conversion and a dot product; AMX-BF16 is a
// tile dot product; AVX10.2 adds packed-only bf16 arithmetic. On AArch64,
// FEAT_BF16 is BFCVT plus widening dot/matrix ops, and Arm's ACLE says so
// outright: "It is expected that arithmetic using standard C operators be used
// using a single-precision floating point format and the value be converted to
// __bf16 when required" (arm-software.github.io/acle).
//
// So ``__bf16`` is a compiler abstraction over the same convert-compute-in-float
// dance, and under GCC it is a disastrously slow one -- GCC never inlines
// __truncsfbf2 / __extendbfsf2, so each element costs a PLT call and the loop
// cannot vectorize. Measured here (2^24 conversions, -O2 -march=native, AVX512-BF16
// host): emulation 4.0 ms, native __bf16 129 ms -- 32x slower. Enabling it by
// default would be a large regression on the primary compiler.
//
// Two further semantic notes, both found by differential testing:
//  * Narrowing through ``__bf16`` is bit-identical to the emulation for all 2^32
//    float inputs, NaN payloads included -- so the opt-in below is safe.
//  * Widening is NOT identical: the native path quiets signalling NaNs (126 of the
//    2^16 patterns change), while the shift reproduces the bits exactly. Widening
//    therefore always uses the shift, opt-in or not.
//
// Native ``__bf16`` is therefore detected (DACE_NATIVE_BF16, above) but NOT used
// unless DACE_BFLOAT16_USE_NATIVE is defined explicitly. That is the one place this
// header does not simply follow the native gate, and it is not a preference -- both
// compilers were measured and both are worse:
//
//  * GCC 15, -O2 -march=native, 2^24 conversions: emulation 4.0 ms, native 129 ms.
//    GCC never inlines __truncsfbf2, so each element is a PLT call and the loop
//    cannot vectorize. A 32x regression.
//  * Clang 21, -march=native: native narrowing FLUSHES SUBNORMALS TO ZERO. The
//    AVX512-BF16 VCVTNEPS2BF16 instruction always flushes denormal output and treats
//    denormal input as zero, and does not consult MXCSR (Intel SDM). Verified here:
//    the float 0x00008001 narrows to 0x0000 where the emulation gives 0x0001, so
//    every bf16 subnormal silently becomes zero. That is a numerical change, not a
//    speed trade.
//
// So the emulation is the default on both, and the opt-in exists for a target where
// neither applies. Narrowing through ``__bf16`` was verified bit-identical to the
// emulation for all 2^32 float inputs under GCC (NaN payloads included), which is
// what makes the opt-in safe there.
#if defined(DACE_BFLOAT16_USE_NATIVE) && defined(DACE_NATIVE_BF16)
#define DACE_BFLOAT16_NATIVE_T __bf16
#endif

DACE_LOWP_CE uint16_t dace_bfloat16_from_float(float f) {
#if defined(DACE_BFLOAT16_NATIVE_T)
#if defined(__cpp_lib_bit_cast)
  return std::bit_cast<uint16_t>((DACE_BFLOAT16_NATIVE_T)f);
#else
  union { DACE_BFLOAT16_NATIVE_T n; uint16_t u; } c;
  c.n = (DACE_BFLOAT16_NATIVE_T)f;
  return c.u;
#endif
#else
  uint32_t u = dace_bits_f2u(f);
  // NaN must be handled before rounding: the bias below can carry into the
  // exponent and turn a NaN whose payload lives in the discarded low bits into
  // an infinity. Keep sign and high payload bits, and force the mantissa MSB so
  // the result stays a (quiet) NaN.
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return (uint16_t)((u >> 16) | 0x0040u);
  }
  // Round-to-nearest-even on the 16 discarded bits: add half an ulp, plus one
  // more when the retained bit is odd so exact ties go to the even neighbour.
  // No overflow check is needed -- a finite float that rounds past the largest
  // finite bf16 becomes infinity, which is the correct IEEE result, and the
  // largest non-NaN input (0xff800000) cannot wrap the 32-bit add.
  uint32_t bias = 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)((u + bias) >> 16);
#endif
}

// Exact for every bit pattern: bf16 IS the top half of a float. Always the shift,
// never the native widening, which quiets signalling NaNs (see above).
DACE_LOWP_CE float dace_bfloat16_to_float(uint16_t b) { return dace_bits_u2f((uint32_t)b << 16); }

// alignas(2), NOT __attribute__((packed)). A single uint16_t member cannot have
// padding, so "packed" buys nothing -- and bare packed would drop alignof to 1,
// which would BREAK the layout identity with __nv_bfloat16 / __hip_bfloat16
// (both size 2, align 2) that makes a host<->device copy bit-exact. The
// static_asserts below are what actually pin the layout, and they fail the build
// rather than silently corrupting a copy.
struct alignas(2) bfloat16 {
  constexpr bfloat16() : h(0) {}
  DACE_LOWP_CE bfloat16(float f) : h(dace_bfloat16_from_float(f)) {}
  DACE_LOWP_CE operator float() const { return dace_bfloat16_to_float(h); }

  // Operator surface identical to ``half`` above, for the same reasons: binary
  // arithmetic, comparisons and unary minus come from the implicit
  // ``operator float()``, and overloading them here would make a mixed
  // expression such as ``b + 1.0f`` ambiguous. Only compound assignment needs
  // an explicit definition, because the built-in one requires an lvalue of
  // arithmetic type. Semantics are ``b = bfloat16(float(b) OP x)`` -- compute in
  // float, round back once.
#define DACE_BFLOAT16_COMPOUND(OP)                                 \
  DACE_LOWP_CE bfloat16 &operator OP##=(float f) {                 \
    *this = bfloat16((float)*this OP f);                           \
    return *this;                                                  \
  }
  DACE_BFLOAT16_COMPOUND(+)
  DACE_BFLOAT16_COMPOUND(-)
  DACE_BFLOAT16_COMPOUND(*)
  DACE_BFLOAT16_COMPOUND(/)
#undef DACE_BFLOAT16_COMPOUND

  DACE_LOWP_CE bfloat16 &operator++() { return *this += 1.0f; }
  DACE_LOWP_CE bfloat16 &operator--() { return *this -= 1.0f; }
  DACE_LOWP_CE bfloat16 operator++(int) { bfloat16 t = *this; *this += 1.0f; return t; }
  DACE_LOWP_CE bfloat16 operator--(int) { bfloat16 t = *this; *this -= 1.0f; return t; }

  uint16_t h;
};
#define DACE_HAS_BFLOAT16 1
DACE_ASSERT_LOWP_LAYOUT(bfloat16);

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

// The identical set for ``dace::bfloat16``, for the identical reasons -- see the
// commentary above, all of which applies verbatim. In particular ``-`` combines
// with ``+=`` (OpenMP sums the per-thread negated copies), and ``&``, ``|``,
// ``^`` are absent because OpenMP rejects them for ``float`` as well: bfloat16
// must accept exactly what float32 accepts, no more.
#pragma omp declare reduction(+ : dace::bfloat16 : omp_out += omp_in) initializer(omp_priv = dace::bfloat16(0.0f))
#pragma omp declare reduction(- : dace::bfloat16 : omp_out += omp_in) initializer(omp_priv = dace::bfloat16(0.0f))
#pragma omp declare reduction(* : dace::bfloat16 : omp_out *= omp_in) initializer(omp_priv = dace::bfloat16(1.0f))
#pragma omp declare reduction(min : dace::bfloat16 : omp_out = (float)omp_in < (float)omp_out ? omp_in : omp_out) \
    initializer(omp_priv = dace::bfloat16(__builtin_huge_valf()))
#pragma omp declare reduction(max : dace::bfloat16 : omp_out = (float)omp_in > (float)omp_out ? omp_in : omp_out) \
    initializer(omp_priv = dace::bfloat16(-__builtin_huge_valf()))
#pragma omp declare reduction(&& : dace::bfloat16 :                                                   \
    omp_out = dace::bfloat16((float)((float)omp_out && (float)omp_in)))                               \
    initializer(omp_priv = dace::bfloat16(1.0f))
#pragma omp declare reduction(|| : dace::bfloat16 :                                                   \
    omp_out = dace::bfloat16((float)((float)omp_out || (float)omp_in)))                               \
    initializer(omp_priv = dace::bfloat16(0.0f))
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
