// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

// Select exactly one ISA backend for the vectorizable-math primitives, keyed on
// the DaCe intrinsics config (``__DACE_USE_INTRINSICS`` / ``__DACE_USE_AVX512`` /
// ``__DACE_USE_SVE``) and the compiler's ISA feature macros. AVX-512 -> SVE ->
// NEON -> portable scalar fallback. Each backend header is ``#pragma once``
// guarded, so this resolves to a single definition set per translation unit.
#if defined(__DACE_USE_INTRINSICS) && (__DACE_USE_INTRINSICS == 1)
#if defined(__AVX512F__) && defined(__DACE_USE_AVX512) && (__DACE_USE_AVX512 == 1)
#include "dace/cpu_vectorizable_math_avx512.h"
#else
#if defined(__DACE_USE_SVE) && (__DACE_USE_SVE == 1)
#if defined(__ARM_FEATURE_SVE)
#include "dace/cpu_vectorizable_math_arm_sve.h"
#else
#include "dace/cpu_vectorizable_math_scalar.h"
#endif
#else
#if defined(__ARM_NEON)
#include "dace/cpu_vectorizable_math_arm_neon.h"
#else
#include "dace/cpu_vectorizable_math_scalar.h"
#endif
#endif
#endif
#else
#include "dace/cpu_vectorizable_math_scalar.h"
#endif

// ============================================================================
// Constant-stride overloads (ISA-independent).
//
// When the stride is known at code-generation time the generator emits
// ``strided_load<T, vector_width, stride>(A, B)`` so that *every* compile-time
// constant -- both the lane count and the stride -- is a template (non-type)
// argument rather than a runtime function parameter. These thin overloads
// forward to the runtime-stride form selected above for the active ISA;
// because that form is ``static inline`` and width-templated, the constexpr
// ``stride`` is fully constant-folded -- the generated code is identical to a
// hand-written constant-stride loop. The runtime-stride form is retained for
// genuinely symbolic strides (e.g. a multi-dim ``N`` that is an SDFG symbol),
// which the generator emits as ``strided_load<T, vector_width>(A, B, stride)``.
//
// Defined here, after the single ISA header include, so the runtime-stride
// declarations they forward to are already in scope for every configuration.
// ============================================================================
template <typename T, int vector_width, int64_t stride>
static inline void strided_load(const T* __restrict__ A, T* __restrict__ B) {
  strided_load<T, vector_width>(A, B, stride);
}

template <typename T, int vector_width, int64_t stride>
static inline void strided_store(const T* __restrict__ A, T* __restrict__ B) {
  strided_store<T, vector_width>(A, B, stride);
}

template <typename T, int vector_width, int64_t stride>
static inline void strided_load_masked(const T* __restrict__ A, T* __restrict__ B, const bool* __restrict__ mask) {
  strided_load_masked<T, vector_width>(A, B, stride, mask);
}

template <typename T, int vector_width, int64_t stride>
static inline void strided_store_masked(const T* __restrict__ A, T* __restrict__ B, const bool* __restrict__ mask) {
  strided_store_masked<T, vector_width>(A, B, stride, mask);
}
