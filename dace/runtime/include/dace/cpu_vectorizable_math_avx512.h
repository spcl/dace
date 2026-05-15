// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <algorithm>
#include <cmath>

#define STRINGIZE(x) STRINGIZE_IMPL(x)
#define STRINGIZE_IMPL(x) #x

#if defined(__clang__)
#define _dace_vectorize(width) _Pragma(STRINGIZE(clang loop vectorize(enable)))
#else
#define _dace_vectorize(width) _Pragma(STRINGIZE(omp simd))
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
// Option F overlay: per-op escape-hatch siblings (vector_<op>_pscalar /
// vector_<op>_av). Arch-independent; one source of truth, included by
// every arch file so the binary always has them regardless of which
// backend the dispatcher selected.
#include "dace/cpu_vectorizable_math_common.h"
#endif

#if !defined(__AVX512F__)
#error Included the AVX512 header without support AVX512
#endif

// --------------------------- vector_mult ---------------------------

template <typename T, int vector_width>
inline void vector_mult(T* __restrict__ c, const T* __restrict__ a,
                        const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __m512 vc = _mm512_mul_ps(va, vb);
      _mm512_storeu_ps(c + i, vc);
    }
    for (; i < vector_width; ++i) {
      c[i] = a[i] * b[i];
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __m512d vc = _mm512_mul_pd(va, vb);
      _mm512_storeu_pd(c + i, vc);
    }
    for (; i < vector_width; ++i) {
      c[i] = a[i] * b[i];
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    c[i] = a[i] * b[i];
  }
#endif
}

// --------------------------- vector_mult_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_mult_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                 const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vc = _mm512_mul_ps(va, vconst);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = a[i] * constant;
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vc = _mm512_mul_pd(va, vconst);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = a[i] * constant;
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = a[i] * constant;
  }
#endif
}

// --------------------------- vector_add ---------------------------

template <typename T, int vector_width>
inline void vector_add(T* __restrict__ c, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __m512 vc = _mm512_add_ps(va, vb);
      _mm512_storeu_ps(c + i, vc);
    }
    for (; i < vector_width; ++i) {
      c[i] = a[i] + b[i];
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __m512d vc = _mm512_add_pd(va, vb);
      _mm512_storeu_pd(c + i, vc);
    }
    for (; i < vector_width; ++i) {
      c[i] = a[i] + b[i];
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    c[i] = a[i] + b[i];
  }
#endif
}

// --------------------------- vector_add_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_add_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vc = _mm512_add_ps(va, vconst);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = a[i] + constant;
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vc = _mm512_add_pd(va, vconst);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = a[i] + constant;
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = a[i] + constant;
  }
#endif
}

// --------------------------- vector_add_masked ---------------------------
// Masked add: lanes where mask[i] == false leave out[i] unchanged in memory
// (RMW). Built on _mm512_mask_storeu_{ps,pd} which writes only active lanes.

template <typename T, int vector_width>
inline void vector_add_masked(T* __restrict__ out, const T* __restrict__ a,
                              const T* __restrict__ b,
                              const bool* __restrict__ mask) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __mmask16 k = 0;
      for (int j = 0; j < W; ++j)
        if (mask[i + j]) k |= __mmask16(1) << j;
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      _mm512_mask_storeu_ps(out + i, k, _mm512_add_ps(va, vb));
    }
    for (; i < vector_width; ++i) {
      if (mask[i]) out[i] = a[i] + b[i];
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __mmask8 k = 0;
      for (int j = 0; j < W; ++j)
        if (mask[i + j]) k |= __mmask8(1) << j;
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      _mm512_mask_storeu_pd(out + i, k, _mm512_add_pd(va, vb));
    }
    for (; i < vector_width; ++i) {
      if (mask[i]) out[i] = a[i] + b[i];
    }
    return;
  }
  // Non-float/double under AVX-512: fall through to scalar fallback below.
#endif
  for (int i = 0; i < vector_width; i++) {
    if (mask[i]) out[i] = a[i] + b[i];
  }
}

// --------------------------- vector_add_w_scalar_masked
// ---------------------------

template <typename T, int vector_width>
inline void vector_add_w_scalar_masked(T* __restrict__ out,
                                       const T* __restrict__ a,
                                       const T constant,
                                       const bool* __restrict__ mask) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __mmask16 k = 0;
      for (int j = 0; j < W; ++j)
        if (mask[i + j]) k |= __mmask16(1) << j;
      __m512 va = _mm512_loadu_ps(a + i);
      _mm512_mask_storeu_ps(out + i, k, _mm512_add_ps(va, vconst));
    }
    for (; i < vector_width; ++i) {
      if (mask[i]) out[i] = a[i] + constant;
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __mmask8 k = 0;
      for (int j = 0; j < W; ++j)
        if (mask[i + j]) k |= __mmask8(1) << j;
      __m512d va = _mm512_loadu_pd(a + i);
      _mm512_mask_storeu_pd(out + i, k, _mm512_add_pd(va, vconst));
    }
    for (; i < vector_width; ++i) {
      if (mask[i]) out[i] = a[i] + constant;
    }
    return;
  }
  // Non-float/double under AVX-512: fall through to scalar fallback below.
#endif
  for (int i = 0; i < vector_width; i++) {
    if (mask[i]) out[i] = a[i] + constant;
  }
}

// --------------------------- vector_sub ---------------------------

template <typename T, int vector_width>
inline void vector_sub(T* __restrict__ c, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __m512 vc = _mm512_sub_ps(va, vb);
      _mm512_storeu_ps(c + i, vc);
    }
    for (; i < vector_width; ++i) {
      c[i] = a[i] - b[i];
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __m512d vc = _mm512_sub_pd(va, vb);
      _mm512_storeu_pd(c + i, vc);
    }
    for (; i < vector_width; ++i) {
      c[i] = a[i] - b[i];
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    c[i] = a[i] - b[i];
  }
#endif
}

// --------------------------- vector_sub_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_sub_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vc = _mm512_sub_ps(va, vconst);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = a[i] - constant;
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vc = _mm512_sub_pd(va, vconst);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = a[i] - constant;
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = a[i] - constant;
  }
#endif
}

// --------------------------- vector_sub_w_scalar_c ---------------------------

template <typename T, int vector_width>
inline void vector_sub_w_scalar_c(T* __restrict__ out, const T constant,
                                  const T* __restrict__ a) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vc = _mm512_sub_ps(vconst, va);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = constant - a[i];
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vc = _mm512_sub_pd(vconst, va);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = constant - a[i];
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = constant - a[i];
  }
#endif
}

// --------------------------- vector_div ---------------------------

template <typename T, int vector_width>
inline void vector_div(T* __restrict__ c, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __m512 vc = _mm512_div_ps(va, vb);
      _mm512_storeu_ps(c + i, vc);
    }
    for (; i < vector_width; ++i) {
      c[i] = a[i] / b[i];
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __m512d vc = _mm512_div_pd(va, vb);
      _mm512_storeu_pd(c + i, vc);
    }
    for (; i < vector_width; ++i) {
      c[i] = a[i] / b[i];
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    c[i] = a[i] / b[i];
  }
#endif
}

// --------------------------- vector_div_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_div_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vc = _mm512_div_ps(va, vconst);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = a[i] / constant;
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vc = _mm512_div_pd(va, vconst);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = a[i] / constant;
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = a[i] / constant;
  }
#endif
}

// --------------------------- vector_div_w_scalar_c ---------------------------

template <typename T, int vector_width>
inline void vector_div_w_scalar_c(T* __restrict__ out, const T constant,
                                  const T* __restrict__ a) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vc = _mm512_div_ps(vconst, va);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = constant / a[i];
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vc = _mm512_div_pd(vconst, va);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = constant / a[i];
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = constant / a[i];
  }
#endif
}

// --------------------------- vector_copy ---------------------------

template <typename T, int vector_width>
inline void vector_copy(T* __restrict__ dst, const T* __restrict__ src) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 v = _mm512_loadu_ps(src + i);
      _mm512_storeu_ps(dst + i, v);
    }
    for (; i < vector_width; ++i) {
      dst[i] = src[i];
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d v = _mm512_loadu_pd(src + i);
      _mm512_storeu_pd(dst + i, v);
    }
    for (; i < vector_width; ++i) {
      dst[i] = src[i];
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    dst[i] = src[i];
  }
#endif
}

// --------------------------- vector_copy_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_copy_w_scalar(T* __restrict__ dst, const T a) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 va = _mm512_set1_ps(a);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      _mm512_storeu_ps(dst + i, va);
    }
    for (; i < vector_width; ++i) {
      dst[i] = a;
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d va = _mm512_set1_pd(a);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      _mm512_storeu_pd(dst + i, va);
    }
    for (; i < vector_width; ++i) {
      dst[i] = a;
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    dst[i] = a;
  }
#endif
}

// --------------------------- vector_min ---------------------------

template <typename T, int vector_width>
inline void vector_min(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __m512 vc = _mm512_min_ps(va, vb);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = std::min(a[i], b[i]);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __m512d vc = _mm512_min_pd(va, vb);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = std::min(a[i], b[i]);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = std::min(a[i], b[i]);
  }
#endif
}

// --------------------------- vector_min_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_min_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vc = _mm512_min_ps(va, vconst);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = std::min(a[i], constant);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vc = _mm512_min_pd(va, vconst);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = std::min(a[i], constant);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = std::min(a[i], constant);
  }
#endif
}

// --------------------------- vector_max ---------------------------

template <typename T, int vector_width>
inline void vector_max(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __m512 vc = _mm512_max_ps(va, vb);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = std::max(a[i], b[i]);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __m512d vc = _mm512_max_pd(va, vb);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = std::max(a[i], b[i]);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = std::max(a[i], b[i]);
  }
#endif
}

// --------------------------- vector_max_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_max_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vc = _mm512_max_ps(va, vconst);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = std::max(a[i], constant);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vc = _mm512_max_pd(va, vconst);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = std::max(a[i], constant);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = std::max(a[i], constant);
  }
#endif
}

// --------------------------- vector_gt ---------------------------

template <typename T, int vector_width>
inline void vector_gt(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vb, _CMP_GT_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] > b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vb, _CMP_GT_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] > b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] > b[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_gt_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_gt_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vconst, _CMP_GT_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] > constant) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vconst, _CMP_GT_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] > constant) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] > constant) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_gt_w_scalar_c ---------------------------

template <typename T, int vector_width>
inline void vector_gt_w_scalar_c(T* __restrict__ out, const T constant,
                                 const T* __restrict__ a) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(vconst, va, _CMP_GT_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (constant > a[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(vconst, va, _CMP_GT_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (constant > a[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (constant > a[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_lt ---------------------------

template <typename T, int vector_width>
inline void vector_lt(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vb, _CMP_LT_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] < b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vb, _CMP_LT_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] < b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] < b[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_lt_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_lt_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vconst, _CMP_LT_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] < constant) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vconst, _CMP_LT_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] < constant) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] < constant) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_lt_w_scalar_c ---------------------------

template <typename T, int vector_width>
inline void vector_lt_w_scalar_c(T* __restrict__ out, const T constant,
                                 const T* __restrict__ a) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(vconst, va, _CMP_LT_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (constant < a[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(vconst, va, _CMP_LT_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (constant < a[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (constant < a[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_ge ---------------------------

template <typename T, int vector_width>
inline void vector_ge(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vb, _CMP_GE_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] >= b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vb, _CMP_GE_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] >= b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] >= b[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_ge_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_ge_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vconst, _CMP_GE_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] >= constant) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vconst, _CMP_GE_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] >= constant) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] >= constant) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_ge_w_scalar_c ---------------------------

template <typename T, int vector_width>
inline void vector_ge_w_scalar_c(T* __restrict__ out, const T constant,
                                 const T* __restrict__ a) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(vconst, va, _CMP_GE_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (constant >= a[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(vconst, va, _CMP_GE_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (constant >= a[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (constant >= a[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_le ---------------------------

template <typename T, int vector_width>
inline void vector_le(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vb, _CMP_LE_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] <= b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vb, _CMP_LE_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] <= b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] <= b[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_le_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_le_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vconst, _CMP_LE_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] <= constant) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vconst, _CMP_LE_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] <= constant) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] <= constant) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_le_w_scalar_c ---------------------------

template <typename T, int vector_width>
inline void vector_le_w_scalar_c(T* __restrict__ out, const T constant,
                                 const T* __restrict__ a) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(vconst, va, _CMP_LE_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (constant <= a[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(vconst, va, _CMP_LE_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (constant <= a[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (constant <= a[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_eq ---------------------------

template <typename T, int vector_width>
inline void vector_eq(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vb, _CMP_EQ_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] == b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vb, _CMP_EQ_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] == b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] == b[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_eq_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_eq_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vconst, _CMP_EQ_OQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] == constant) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vconst, _CMP_EQ_OQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] == constant) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] == constant) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_ne ---------------------------

template <typename T, int vector_width>
inline void vector_ne(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __m512 vb = _mm512_loadu_ps(b + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vb, _CMP_NEQ_UQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] != b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __m512d vb = _mm512_loadu_pd(b + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vb, _CMP_NEQ_UQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] != b[i]) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] != b[i]) ? T(1.0) : T(0.0);
  }
#endif
}

// --------------------------- vector_ne_w_scalar ---------------------------

template <typename T, int vector_width>
inline void vector_ne_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 16;
    __m512 vconst = _mm512_set1_ps(constant);
    __m512 vzero = _mm512_set1_ps(0.0f);
    __m512 vone = _mm512_set1_ps(1.0f);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512 va = _mm512_loadu_ps(a + i);
      __mmask16 m = _mm512_cmp_ps_mask(va, vconst, _CMP_NEQ_UQ);
      __m512 vc = _mm512_mask_mov_ps(vzero, m, vone);
      _mm512_storeu_ps(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] != constant) ? T(1.0) : T(0.0);
    }
    return;
  }
  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 8;
    __m512d vconst = _mm512_set1_pd(constant);
    __m512d vzero = _mm512_set1_pd(0.0);
    __m512d vone = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      __m512d va = _mm512_loadu_pd(a + i);
      __mmask8 m = _mm512_cmp_pd_mask(va, vconst, _CMP_NEQ_UQ);
      __m512d vc = _mm512_mask_mov_pd(vzero, m, vone);
      _mm512_storeu_pd(out + i, vc);
    }
    for (; i < vector_width; ++i) {
      out[i] = (a[i] != constant) ? T(1.0) : T(0.0);
    }
    return;
  }
#else
  for (int i = 0; i < vector_width; i++) {
    out[i] = (a[i] != constant) ? T(1.0) : T(0.0);
  }
#endif
}

template <typename T, int vector_width>
inline void vector_exp(T* __restrict__ out, const T* __restrict__ a) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) {
    out[i] = std::exp(a[i]);
  }
}

template <typename T, int vector_width>
inline void vector_log(T* __restrict__ out, const T* __restrict__ a) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) {
    out[i] = std::log(a[i]);
  }
}

template <typename T, int vector_width, typename CondT = bool>
inline void vector_select(T* __restrict__ out, const CondT* __restrict__ cond,
                          const T* __restrict__ t, const T* __restrict__ e) {
  _dace_vectorize(vector_width) for (int i = 0; i < vector_width; i++) {
    out[i] = cond[i] ? t[i] : e[i];
  }
}

// ============================================================================
// Runtime-length scatter / gather / strided load+store (moved from
// vector_intrinsics/{gather,scatter,strided_load,strided_store}.h).
// AVX-512 specializations using native gather/scatter intrinsics. Scalar
// tail handling for length not divisible by 8.
// ============================================================================

template <typename T>
inline void gather(const T* __restrict__ A, const int64_t* __restrict__ idx,
                   T* __restrict__ B, const int64_t length) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, double>::value) {
    if (length >= 8) {
      int64_t i = 0;
      for (; i + 8 <= length; i += 8) {
        __m512i vindex = _mm512_loadu_si512((const __m512i*)&idx[i]);
        __m512d vdata = _mm512_i64gather_pd(vindex, A, 8);
        _mm512_storeu_pd(&B[i], vdata);
      }
      for (; i < length; ++i) B[i] = A[idx[i]];
    } else {
      for (int64_t i = 0; i < length; ++i) B[i] = A[idx[i]];
    }
    return;
  }
#endif
  // Scalar fallback for non-double T or non-AVX-512 builds (also covers float).
  for (int64_t i = 0; i < length; ++i) B[i] = A[idx[i]];
}

template <typename T>
inline void scatter(const T* __restrict__ A, const int64_t* __restrict__ idx,
                    T* __restrict__ B, const int64_t length) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, double>::value) {
    if (length >= 8) {
      int64_t i = 0;
      for (; i + 8 <= length; i += 8) {
        __m512i vindex = _mm512_loadu_si512((const __m512i*)&idx[i]);
        __m512d vdata = _mm512_loadu_pd(&A[i]);
        _mm512_i64scatter_pd(B, vindex, vdata, 8);
      }
      for (; i < length; ++i) B[idx[i]] = A[i];
    } else {
      for (int64_t i = 0; i < length; ++i) B[idx[i]] = A[i];
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[idx[i]] = A[i];
}

template <typename T>
inline void strided_load(const T* __restrict__ A, T* __restrict__ B,
                         const int64_t length, const int64_t stride) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, double>::value) {
    if (length >= 8) {
      for (int64_t i = 0; i < length; i += 8) {
        int64_t idx_buf[8];
        int lane_count = 0;
        for (int lane = 0; lane < 8 && (i + lane) < length; ++lane) {
          idx_buf[lane] = (i + lane) * stride;
          ++lane_count;
        }
        __m512i vindex = _mm512_loadu_si512((const __m512i*)idx_buf);
        __m512d vdata = _mm512_i64gather_pd(vindex, A, 8);
        double tmp[8];
        _mm512_storeu_pd(tmp, vdata);
        for (int lane = 0; lane < lane_count; ++lane) B[i + lane] = tmp[lane];
      }
    } else {
      for (int64_t i = 0; i < length; ++i) B[i] = A[i * stride];
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[i] = A[i * stride];
}

template <typename T>
inline void strided_store(const T* __restrict__ A, T* __restrict__ B,
                          const int64_t length, const int64_t stride) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, double>::value) {
    if (length >= 8) {
      for (int64_t i = 0; i < length; i += 8) {
        int64_t idx_buf[8];
        double val_buf[8];
        int lane_count = 0;
        for (int lane = 0; lane < 8 && (i + lane) < length; ++lane) {
          idx_buf[lane] = (i + lane) * stride;
          val_buf[lane] = A[i + lane];
          ++lane_count;
        }
        __m512i vindex = _mm512_loadu_si512((const __m512i*)idx_buf);
        __m512d vdata = _mm512_loadu_pd(val_buf);
        _mm512_i64scatter_pd(B, vindex, vdata, 8);
      }
    } else {
      for (int64_t i = 0; i < length; ++i) B[i * stride] = A[i];
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[i * stride] = A[i];
}

// --------------------------- masked variants (RMW) ---------------------------
// Inactive lanes leave the destination memory unchanged. Built on
// _mm512_mask_i64{gather,scatter}_pd and _mm512_mask_storeu_pd which honour
// the lane mask. Matches the vector_<op>_masked convention.

template <typename T>
inline void gather_masked(const T* __restrict__ A,
                          const int64_t* __restrict__ idx, T* __restrict__ B,
                          const int64_t length, const bool* __restrict__ mask) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, double>::value) {
    if (length >= 8) {
      int64_t i = 0;
      for (; i + 8 <= length; i += 8) {
        __mmask8 k = 0;
        for (int lane = 0; lane < 8; ++lane)
          if (mask[i + lane]) k |= __mmask8(1) << lane;
        __m512i vindex = _mm512_loadu_si512((const __m512i*)&idx[i]);
        __m512d vdata =
            _mm512_mask_i64gather_pd(_mm512_setzero_pd(), k, vindex, A, 8);
        _mm512_mask_storeu_pd(&B[i], k, vdata);
      }
      for (; i < length; ++i)
        if (mask[i]) B[i] = A[idx[i]];
    } else {
      for (int64_t i = 0; i < length; ++i)
        if (mask[i]) B[i] = A[idx[i]];
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[i] = A[idx[i]];
}

template <typename T>
inline void scatter_masked(const T* __restrict__ A,
                           const int64_t* __restrict__ idx, T* __restrict__ B,
                           const int64_t length,
                           const bool* __restrict__ mask) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, double>::value) {
    if (length >= 8) {
      int64_t i = 0;
      for (; i + 8 <= length; i += 8) {
        __mmask8 k = 0;
        for (int lane = 0; lane < 8; ++lane)
          if (mask[i + lane]) k |= __mmask8(1) << lane;
        __m512i vindex = _mm512_loadu_si512((const __m512i*)&idx[i]);
        __m512d vdata = _mm512_loadu_pd(&A[i]);
        _mm512_mask_i64scatter_pd(B, k, vindex, vdata, 8);
      }
      for (; i < length; ++i)
        if (mask[i]) B[idx[i]] = A[i];
    } else {
      for (int64_t i = 0; i < length; ++i)
        if (mask[i]) B[idx[i]] = A[i];
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[idx[i]] = A[i];
}

template <typename T>
inline void strided_load_masked(const T* __restrict__ A, T* __restrict__ B,
                                const int64_t length, const int64_t stride,
                                const bool* __restrict__ mask) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, double>::value) {
    if (length >= 8) {
      for (int64_t i = 0; i < length; i += 8) {
        int64_t idx_buf[8];
        __mmask8 k = 0;
        for (int lane = 0; lane < 8 && (i + lane) < length; ++lane) {
          idx_buf[lane] = (i + lane) * stride;
          if (mask[i + lane]) k |= __mmask8(1) << lane;
        }
        __m512i vindex = _mm512_loadu_si512((const __m512i*)idx_buf);
        __m512d vdata =
            _mm512_mask_i64gather_pd(_mm512_setzero_pd(), k, vindex, A, 8);
        _mm512_mask_storeu_pd(&B[i], k, vdata);
      }
    } else {
      for (int64_t i = 0; i < length; ++i)
        if (mask[i]) B[i] = A[i * stride];
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[i] = A[i * stride];
}

template <typename T>
inline void strided_store_masked(const T* __restrict__ A, T* __restrict__ B,
                                 const int64_t length, const int64_t stride,
                                 const bool* __restrict__ mask) {
#if defined(__AVX512F__)
  if constexpr (std::is_same<T, double>::value) {
    if (length >= 8) {
      for (int64_t i = 0; i < length; i += 8) {
        int64_t idx_buf[8];
        double val_buf[8];
        __mmask8 k = 0;
        for (int lane = 0; lane < 8 && (i + lane) < length; ++lane) {
          idx_buf[lane] = (i + lane) * stride;
          val_buf[lane] = A[i + lane];
          if (mask[i + lane]) k |= __mmask8(1) << lane;
        }
        __m512i vindex = _mm512_loadu_si512((const __m512i*)idx_buf);
        __m512d vdata = _mm512_loadu_pd(val_buf);
        _mm512_mask_i64scatter_pd(B, k, vindex, vdata, 8);
      }
    } else {
      for (int64_t i = 0; i < length; ++i)
        if (mask[i]) B[i * stride] = A[i];
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[i * stride] = A[i];
}
