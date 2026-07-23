// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_TYPES_H
#define __DACE_TYPES_H

#include <cstdint>
#include <complex>
#include <type_traits>
#include <bit>  // std::bit_cast (C++20)

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
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #include <cuda_fp8.h>
    #include <thrust/complex.h>
    #include "cuda/multidim_gbar.cuh"

    // Workaround so that the native low-precision types are scalars (for reductions)
    namespace std {
        template <> struct is_scalar<half> : std::integral_constant<bool, true> {};
        template <> struct is_fundamental<half> : std::integral_constant<bool, true> {};
        template <> struct is_scalar<__nv_bfloat16> : std::integral_constant<bool, true> {};
        template <> struct is_fundamental<__nv_bfloat16> : std::integral_constant<bool, true> {};
        template <> struct is_scalar<__nv_fp8_e4m3> : std::integral_constant<bool, true> {};
        template <> struct is_fundamental<__nv_fp8_e4m3> : std::integral_constant<bool, true> {};
        template <> struct is_scalar<__nv_fp8_e5m2> : std::integral_constant<bool, true> {};
        template <> struct is_fundamental<__nv_fp8_e5m2> : std::integral_constant<bool, true> {};
    }  // namespace std
#elif defined(__HIPCC__)
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
    #include <hip/hip_bf16.h>
    #include <hip/hip_fp8.h>

    namespace std {
        template <> struct is_scalar<half> : std::integral_constant<bool, true> {};
        template <> struct is_fundamental<half> : std::integral_constant<bool, true> {};
        template <> struct is_scalar<__hip_bfloat16> : std::integral_constant<bool, true> {};
        template <> struct is_fundamental<__hip_bfloat16> : std::integral_constant<bool, true> {};
        template <> struct is_scalar<__hip_fp8_e4m3> : std::integral_constant<bool, true> {};
        template <> struct is_fundamental<__hip_fp8_e4m3> : std::integral_constant<bool, true> {};
        template <> struct is_scalar<__hip_fp8_e5m2> : std::integral_constant<bool, true> {};
        template <> struct is_fundamental<__hip_fp8_e5m2> : std::integral_constant<bool, true> {};
    }  // namespace std
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

    #if defined(__CUDACC__) || defined(__HIPCC__)
    #ifdef __CUDACC__
    typedef thrust::complex<float> complex64;
    typedef thrust::complex<double> complex128;
    #endif
    // GPU native low-precision types. Bit-identical to the CPU structs below (checked next).
    // e4m3fn == the OCP finite E4M3 (max +-448): __nv_fp8_e4m3 / __hip_fp8_e4m3, NOT the fnuz form.
    typedef half float16;
    #ifdef __CUDACC__
    typedef __nv_bfloat16 bfloat16;
    typedef __nv_fp8_e4m3 float8_e4m3fn;
    typedef __nv_fp8_e5m2 float8_e5m2;
    #else  // __HIPCC__
    typedef __hip_bfloat16 bfloat16;
    typedef __hip_fp8_e4m3 float8_e4m3fn;
    typedef __hip_fp8_e5m2 float8_e5m2;
    #endif
    static_assert(sizeof(float16) == 2 && sizeof(bfloat16) == 2, "16-bit low-precision must be 2 bytes");
    static_assert(sizeof(float8_e4m3fn) == 1 && sizeof(float8_e5m2) == 1, "fp8 must be 1 byte");
    #else
    typedef std::complex<float> complex64;
    typedef std::complex<double> complex128;

    static constexpr uint32_t dace_f2u(float f) { return std::bit_cast<uint32_t>(f); }
    static constexpr float dace_u2f(uint32_t u) { return std::bit_cast<float>(u); }

    // Native _Float16 for the float<->half CONVERSIONS only -- confined to the two routines below,
    // never a member/signature/operand, so ABI and mangling are untouched; arithmetic still runs in
    // float. Gate on the ISA that has a hardware convert (any compiler advertising it also provides
    // _Float16), so NVHPC / Intel LLVM / GCC / Clang are all covered; everything else stays on the
    // (correct) emulation. x86: AVX512-FP16 (not plain F16C, whose _Float16 lowers to a slow libcall).
    // AArch64: FCVT is base ARMv8-A. Override with -DDACE_HALF_FORCE_NATIVE / -DDACE_HALF_NO_NATIVE.
    #if !defined(DACE_HALF_NO_NATIVE) &&                                            \
        (defined(DACE_HALF_FORCE_NATIVE) ||                                         \
         ((defined(__x86_64__) || defined(__i386__)) && defined(__AVX512FP16__)) || \
         (defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE)))
    #define DACE_HALF_NATIVE_T _Float16
    #endif

    #if defined(DACE_HALF_NATIVE_T)
    constexpr uint16_t dace_half_from_float(float f) { return std::bit_cast<uint16_t>((DACE_HALF_NATIVE_T)f); }
    constexpr float dace_half_to_float(uint16_t h) { return (float)std::bit_cast<DACE_HALF_NATIVE_T>(h); }
    #else
    // Emulated IEEE binary32 -> binary16, RNE. Based on https://gist.github.com/rygorous/2156668
    constexpr uint16_t dace_half_from_float(float f) {
        uint32_t u = dace_f2u(f), sign = u & 0x80000000u; u ^= sign; uint16_t h;
        if (u >= 0x47800000u) h = (u > 0x7f800000u) ? 0x7e00 : 0x7c00;      // NaN / Inf
        else if (u < 0x38800000u) { float m = dace_u2f(u) + dace_u2f(0x3f000000u); h = (uint16_t)(dace_f2u(m) - 0x3f000000u); }  // subnormal
        else { uint32_t mo = (u >> 13) & 1; u += 0xc8000000u + 0xfffu; u += mo; h = (uint16_t)(u >> 13); }  // normal
        return (uint16_t)(h | (uint16_t)(sign >> 16));
    }
    constexpr float dace_half_to_float(uint16_t h) {
        uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, out;
        if (exp == 0) {
            if (mant == 0) out = sign;
            else { exp = 1; do { exp--; mant <<= 1; } while ((mant & 0x400) == 0); mant &= 0x3ff; out = sign | ((exp + 112) << 23) | (mant << 13); }
        } else if (exp == 0x1f) out = sign | 0x7f800000u | (mant << 13);
        else out = sign | ((exp + 112) << 23) | (mant << 13);
        return dace_u2f(out);
    }
    #endif

    // bfloat16 <-> float, RNE. bf16 is the high 16 bits of binary32.
    constexpr uint16_t dace_bf16_from_float(float f) {
        uint32_t u = dace_f2u(f);
        if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);  // NaN
        return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    constexpr float dace_bf16_to_float(uint16_t h) { return dace_u2f((uint32_t)h << 16); }

    // float -> (1 sign, TE exp, TM mant) fp8, RNE. has_inf: e5m2 (exp all-ones is inf/nan,
    // overflow saturates to inf) vs e4m3fn (only all-ones magnitude is NaN, overflow -> NaN).
    template <int TE, int TM, bool has_inf>
    constexpr uint8_t dace_fp8_from_float(float f) {
        const int TB = (1 << (TE - 1)) - 1;
        const uint32_t ALLONES = (1u << (TE + TM)) - 1, INF = ((uint32_t)((1 << TE) - 1)) << TM;
        const uint32_t NANC = has_inf ? (INF | (1u << (TM - 1))) : ALLONES;
        const uint32_t MAXF = has_inf ? (INF - 1u) : (ALLONES - 1u), OVF = has_inf ? INF : NANC;
        uint32_t u = dace_f2u(f); uint8_t s = (uint8_t)((u >> 31) << (TE + TM)); u &= 0x7fffffffu;
        if (u > 0x7f800000u) return (uint8_t)(s | NANC);
        if (u == 0x7f800000u) return (uint8_t)(s | OVF);
        int e = (int)(u >> 23) - 127, te = e + TB; uint32_t m = u & 0x7fffffu, mag;
        if (te >= (1 << TE)) return (uint8_t)(s | OVF);
        else if (te <= 0) {
            uint32_t full = (u >> 23) ? (m | 0x800000u) : m; int r = (23 - TM) + (1 - te);
            if (r > 31) return (uint8_t)s;
            uint32_t rb = (full >> (r - 1)) & 1u, st = (full & (((uint32_t)1 << (r - 1)) - 1)) != 0;
            mag = full >> r; if (rb && (st || (mag & 1u))) mag++;
        } else {
            int r = 23 - TM; uint32_t rb = (m >> (r - 1)) & 1u, st = (m & (((uint32_t)1 << (r - 1)) - 1)) != 0;
            uint32_t tm = m >> r, tu = (uint32_t)te;
            if (rb && (st || (tm & 1u))) { tm++; if (tm >> TM) { tm = 0; tu++; } }
            mag = (tu << TM) | tm;
        }
        return (uint8_t)(s | (mag > MAXF ? OVF : mag));
    }
    template <int TE, int TM, bool has_inf>
    constexpr float dace_fp8_to_float(uint8_t v) {
        const int TB = (1 << (TE - 1)) - 1;
        uint32_t sign = (uint32_t)(v >> (TE + TM)) << 31, exp = (v >> TM) & ((1u << TE) - 1), mant = v & ((1u << TM) - 1), out;
        if (exp == (1u << TE) - 1 && (has_inf || mant == ((1u << TM) - 1)))
            out = (has_inf && mant == 0) ? (sign | 0x7f800000u) : (sign | 0x7fc00000u);  // inf / nan
        else if (exp == 0) {
            if (mant == 0) out = sign;
            else { int e = 1; uint32_t mm = mant; while (!(mm & (1u << TM))) { e--; mm <<= 1; } mm &= (1u << TM) - 1; out = sign | ((uint32_t)(e + (127 - TB)) << 23) | (mm << (23 - TM)); }
        } else out = sign | ((exp + (127 - TB)) << 23) | (mant << (23 - TM));
        return dace_u2f(out);
    }

    // Storage + conversion + full float operator surface. Binary arithmetic, comparisons and unary
    // minus come from ``operator float()``; declaring member binary overloads would tie with the
    // built-ins on ``x OP 1.0f`` and be ambiguous. Only compound assignment and inc/dec are added.
    #define DACE_LP_STRUCT(TYPE, STORE, FROMF, TOF)                                                     \
    struct alignas(sizeof(STORE)) TYPE {                                                                \
        STORE h;                                                                                        \
        constexpr TYPE() : h(0) {}                                                                      \
        constexpr TYPE(float f) : h(FROMF(f)) {}                                                        \
        constexpr operator float() const { return TOF(h); }                                            \
        constexpr TYPE &operator+=(float f) { *this = TYPE((float)*this + f); return *this; }           \
        constexpr TYPE &operator-=(float f) { *this = TYPE((float)*this - f); return *this; }           \
        constexpr TYPE &operator*=(float f) { *this = TYPE((float)*this * f); return *this; }           \
        constexpr TYPE &operator/=(float f) { *this = TYPE((float)*this / f); return *this; }           \
        constexpr TYPE &operator++() { return *this += 1.0f; }                                          \
        constexpr TYPE &operator--() { return *this -= 1.0f; }                                          \
        constexpr TYPE operator++(int) { TYPE t = *this; *this += 1.0f; return t; }                     \
        constexpr TYPE operator--(int) { TYPE t = *this; *this -= 1.0f; return t; }                     \
    }
    DACE_LP_STRUCT(half, uint16_t, dace_half_from_float, dace_half_to_float);
    DACE_LP_STRUCT(bfloat16, uint16_t, dace_bf16_from_float, dace_bf16_to_float);
    DACE_LP_STRUCT(float8_e5m2, uint8_t, (dace_fp8_from_float<5, 2, true>), (dace_fp8_to_float<5, 2, true>));
    DACE_LP_STRUCT(float8_e4m3fn, uint8_t, (dace_fp8_from_float<4, 3, false>), (dace_fp8_to_float<4, 3, false>));
    #undef DACE_LP_STRUCT
    typedef half float16;

    // Representation guardrails: prove the CPU structs are copy-compatible with the GPU natives.
    #define DACE_LP_CHECK(TYPE, N)                                                                      \
        static_assert(sizeof(TYPE) == (N) && alignof(TYPE) == (N), #TYPE " must be " #N " byte(s)");    \
        static_assert(std::is_trivially_copyable<TYPE>::value, #TYPE " must be trivially copyable");    \
        static_assert(std::is_standard_layout<TYPE>::value, #TYPE " must be standard-layout")
    DACE_LP_CHECK(half, 2); DACE_LP_CHECK(bfloat16, 2);
    DACE_LP_CHECK(float8_e4m3fn, 1); DACE_LP_CHECK(float8_e5m2, 1);
    #undef DACE_LP_CHECK

    // Known-value bit patterns pin every path to IEEE binary16 / bfloat16 / e4m3 / e5m2.
    static_assert(std::bit_cast<uint16_t>(half(1.0f)) == 0x3C00 && std::bit_cast<uint16_t>(half(-2.0f)) == 0xC000, "half repr");
    static_assert(std::bit_cast<uint16_t>(bfloat16(1.0f)) == 0x3F80 && std::bit_cast<uint16_t>(bfloat16(-2.0f)) == 0xC000, "bf16 repr");
    static_assert(std::bit_cast<uint8_t>(float8_e5m2(1.0f)) == 0x3C && std::bit_cast<uint8_t>(float8_e4m3fn(1.0f)) == 0x38, "fp8 repr");

    // OpenMP has no built-in reduction over a class type; declare the float set (no bitwise ops)
    // for each. '-' combines with '+=' (initial - sum), matching OpenMP's '-' reduction for float.
    #ifdef _OPENMP
    #define DACE_LP_OMP_RED(T)                                                                                              \
        DACE_PRAGMA(omp declare reduction(+ : T : omp_out += omp_in) initializer(omp_priv = T(0.0f)))                        \
        DACE_PRAGMA(omp declare reduction(- : T : omp_out += omp_in) initializer(omp_priv = T(0.0f)))                        \
        DACE_PRAGMA(omp declare reduction(* : T : omp_out *= omp_in) initializer(omp_priv = T(1.0f)))                        \
        DACE_PRAGMA(omp declare reduction(min : T : omp_out = (float)omp_in < (float)omp_out ? omp_in : omp_out) initializer(omp_priv = T(__builtin_huge_valf())))  \
        DACE_PRAGMA(omp declare reduction(max : T : omp_out = (float)omp_in > (float)omp_out ? omp_in : omp_out) initializer(omp_priv = T(-__builtin_huge_valf()))) \
        DACE_PRAGMA(omp declare reduction(&& : T : omp_out = T((float)((float)omp_out && (float)omp_in))) initializer(omp_priv = T(1.0f)))                           \
        DACE_PRAGMA(omp declare reduction(|| : T : omp_out = T((float)((float)omp_out || (float)omp_in))) initializer(omp_priv = T(0.0f)))
    DACE_LP_OMP_RED(half)
    DACE_LP_OMP_RED(bfloat16)
    DACE_LP_OMP_RED(float8_e4m3fn)
    DACE_LP_OMP_RED(float8_e5m2)
    #undef DACE_LP_OMP_RED
    #endif
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
