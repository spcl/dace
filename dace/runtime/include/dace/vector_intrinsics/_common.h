// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Common preamble for per-op vector intrinsic headers under
// dace/vector_intrinsics/. Owns the _dace_vectorize macro and the
// architecture-detection includes so each vector_<op>.h file can be a thin
// shell of one (T, W) template per direction (unmasked + masked) with an
// internal #if cascade.
//
// Mask semantics for the *_masked variants: ``mask`` is ``bool[W]``; lanes
// where ``mask[i] == false`` leave ``out[i]`` unchanged (read-modify-write
// per lane). This matches _iter_mask and _cond_mask (M1) verbatim and lets
// the emitter combine masks with a plain ``&`` without further branching.

#pragma once

#include <cmath>
#include <algorithm>
#include <cstdint>

#define DACE_INTRINSIC_STRINGIZE(x) DACE_INTRINSIC_STRINGIZE_IMPL(x)
#define DACE_INTRINSIC_STRINGIZE_IMPL(x) #x

// Compiler-vectorize hint for scalar-loop fallback paths. Picks `clang loop
// vectorize(enable)` on clang, `omp simd` elsewhere (gcc/msvc).
#if defined(__clang__)
  #define _dace_vectorize(width) _Pragma(DACE_INTRINSIC_STRINGIZE(clang loop vectorize(enable)))
#else
  #define _dace_vectorize(width) _Pragma(DACE_INTRINSIC_STRINGIZE(omp simd))
#endif

// Per-arch intrinsic headers — guarded so each vector_<op>.h file can use
// e.g. ``__m512d`` / ``svfloat64_t`` / ``float64x2_t`` without each file
// repeating the same conditional include block.
#if defined(__AVX512F__) || defined(__AVX2__)
  #include <immintrin.h>
#endif
#if defined(__ARM_FEATURE_SVE)
  #include <arm_sve.h>
#endif
#if defined(__ARM_NEON)
  #include <arm_neon.h>
#endif
