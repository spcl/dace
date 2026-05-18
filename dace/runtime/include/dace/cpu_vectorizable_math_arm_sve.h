// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#define STRINGIZE(x) STRINGIZE_IMPL(x)
#define STRINGIZE_IMPL(x) #x

#if defined(__clang__)
#define _dace_vectorize(width) _Pragma(STRINGIZE(clang loop vectorize(enable)))
#else
#define _dace_vectorize(width) _Pragma(STRINGIZE(omp simd))
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
// Option F overlay: per-op escape-hatch siblings (vector_<op>_pscalar /
// vector_<op>_av). Arch-independent; one source of truth, included by
// every arch file so the binary always has them regardless of which
// backend the dispatcher selected.
#include "dace/cpu_vectorizable_math_common.h"
#endif

#if !defined(__ARM_FEATURE_SVE)
#error Included the SVE header without support SVE
#endif

// ============================================================================
// Arithmetic
// ============================================================================

// vector_mult
template <typename T, int vector_width>
inline void vector_mult(T* __restrict__ out, const T* __restrict__ a,
                        const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svfloat32_t vc = svmul_f32_m(pg, va, vb);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svfloat64_t vc = svmul_f64_m(pg, va, vb);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else  // __ARM_FEATURE_SVE

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] * b[i];

#endif
}

// vector_mult_w_scalar
template <typename T, int vector_width>
inline void vector_mult_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                 const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vc = svmul_f32_m(pg, va, vconst);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vc = svmul_f64_m(pg, va, vconst);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] * constant;

#endif
}

// vector_add
template <typename T, int vector_width>
inline void vector_add(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svfloat32_t vc = svadd_f32_m(pg, va, vb);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svfloat64_t vc = svadd_f64_m(pg, va, vb);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else  // __ARM_FEATURE_SVE

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] + b[i];

#endif
}

// vector_add_w_scalar
template <typename T, int vector_width>
inline void vector_add_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vc = svadd_f32_m(pg, va, vconst);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vc = svadd_f64_m(pg, va, vconst);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] + constant;

#endif
}

// vector_add_masked: store ONLY lanes where mask[i] == true; inactive
// lanes are left untouched.
//
// The bool[W] mask is loaded byte-wise under the VLS iteration predicate
// ``pg`` and compared against zero to derive the active-lane predicate
// ``m`` (m is a subset of pg, so its active lanes are always in-bounds).
// The operand loads and the store are then predicated by ``m``: per the
// Arm C Language Extensions, a predicated ``svld1`` does not access
// memory for inactive lanes (no fault) and a predicated ``svst1`` does
// not write them. The previous form computed under ``pg`` and
// ``svsel``-blended against ``svld1(pg, out)``, then stored under ``pg``
// — that store wrote all ``pg`` lanes, so at a masked remainder where
// ``out = arr + tile_i`` the trailing inactive lanes index past the
// array end -> OOB heap write (the TSVC s2710 masked-merge-65 segfault).
// Predicating the store by ``m`` removes both the OOB store and the
// (also-OOB) ``vold`` load.
template <typename T, int vector_width>
inline void vector_add_masked(T* __restrict__ out, const T* __restrict__ a,
                              const T* __restrict__ b,
                              const bool* __restrict__ mask) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svuint32_t mv = svld1ub_u32(pg, (const uint8_t*)(mask + i));
      svbool_t m = svcmpne_n_u32(pg, mv, 0);
      svfloat32_t va = svld1_f32(m, a + i);
      svfloat32_t vb = svld1_f32(m, b + i);
      svfloat32_t sum = svadd_f32_x(m, va, vb);
      svst1_f32(m, out + i, sum);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svuint64_t mv = svld1ub_u64(pg, (const uint8_t*)(mask + i));
      svbool_t m = svcmpne_n_u64(pg, mv, 0);
      svfloat64_t va = svld1_f64(m, a + i);
      svfloat64_t vb = svld1_f64(m, b + i);
      svfloat64_t sum = svadd_f64_x(m, va, vb);
      svst1_f64(m, out + i, sum);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    if (mask[i]) out[i] = a[i] + b[i];

#endif
}

// vector_add_w_scalar_masked: same mask-predicated load/store as
// vector_add_masked above (drops the OOB pg-wide store + vold load).
template <typename T, int vector_width>
inline void vector_add_w_scalar_masked(T* __restrict__ out,
                                       const T* __restrict__ a,
                                       const T constant,
                                       const bool* __restrict__ mask) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svuint32_t mv = svld1ub_u32(pg, (const uint8_t*)(mask + i));
      svbool_t m = svcmpne_n_u32(pg, mv, 0);
      svfloat32_t va = svld1_f32(m, a + i);
      svfloat32_t sum = svadd_f32_x(m, va, vconst);
      svst1_f32(m, out + i, sum);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svuint64_t mv = svld1ub_u64(pg, (const uint8_t*)(mask + i));
      svbool_t m = svcmpne_n_u64(pg, mv, 0);
      svfloat64_t va = svld1_f64(m, a + i);
      svfloat64_t sum = svadd_f64_x(m, va, vconst);
      svst1_f64(m, out + i, sum);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    if (mask[i]) out[i] = a[i] + constant;

#endif
}

// vector_sub
template <typename T, int vector_width>
inline void vector_sub(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svfloat32_t vc = svsub_f32_m(pg, va, vb);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svfloat64_t vc = svsub_f64_m(pg, va, vb);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] - b[i];

#endif
}

// vector_sub_w_scalar (a[i] - constant)
template <typename T, int vector_width>
inline void vector_sub_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vc = svsub_f32_m(pg, va, vconst);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vc = svsub_f64_m(pg, va, vconst);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] - constant;

#endif
}

// vector_sub_w_scalar_c (constant - a[i])
template <typename T, int vector_width>
inline void vector_sub_w_scalar_c(T* __restrict__ out, const T constant,
                                  const T* __restrict__ a) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vc = svsub_f32_m(pg, vconst, va);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vc = svsub_f64_m(pg, vconst, va);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = constant - a[i];

#endif
}

// vector_div
template <typename T, int vector_width>
inline void vector_div(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svfloat32_t vc = svdiv_f32_m(pg, va, vb);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svfloat64_t vc = svdiv_f64_m(pg, va, vb);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] / b[i];

#endif
}

// vector_div_w_scalar (a[i] / constant)
template <typename T, int vector_width>
inline void vector_div_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vc = svdiv_f32_m(pg, va, vconst);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vc = svdiv_f64_m(pg, va, vconst);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] / constant;

#endif
}

// vector_div_w_scalar_c (constant / a[i])
template <typename T, int vector_width>
inline void vector_div_w_scalar_c(T* __restrict__ out, const T constant,
                                  const T* __restrict__ a) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vc = svdiv_f32_m(pg, vconst, va);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vc = svdiv_f64_m(pg, vconst, va);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = constant / a[i];

#endif
}

// vector_copy
template <typename T, int vector_width>
inline void vector_copy(T* __restrict__ dst, const T* __restrict__ src) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t v = svld1_f32(pg, src + i);
      svst1_f32(pg, dst + i, v);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t v = svld1_f64(pg, src + i);
      svst1_f64(pg, dst + i, v);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) dst[i] = src[i];

#endif
}

// vector_copy_w_scalar
template <typename T, int vector_width>
inline void vector_copy_w_scalar(T* __restrict__ dst, const T a) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t va = svdup_f32(a);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svst1_f32(pg, dst + i, va);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t va = svdup_f64(a);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svst1_f64(pg, dst + i, va);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) dst[i] = a;

#endif
}

// ============================================================================
// Elementwise non-linear (scalar only as per your requirement)
// ============================================================================

template <typename T, int vector_width>
inline void vector_exp(T* __restrict__ out, const T* __restrict__ a) {
  for (int i = 0; i < vector_width; ++i) out[i] = std::exp(a[i]);
}

template <typename T, int vector_width>
inline void vector_log(T* __restrict__ out, const T* __restrict__ a) {
  for (int i = 0; i < vector_width; ++i) out[i] = std::log(a[i]);
}

// ============================================================================
// Min / Max
// ============================================================================

// vector_min
template <typename T, int vector_width>
inline void vector_min(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svfloat32_t vc = svmin_f32_m(pg, va, vb);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svfloat64_t vc = svmin_f64_m(pg, va, vb);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = std::min(a[i], b[i]);

#endif
}

// vector_min_w_scalar
template <typename T, int vector_width>
inline void vector_min_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vc = svmin_f32_m(pg, va, vconst);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vc = svmin_f64_m(pg, va, vconst);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = std::min(a[i], constant);

#endif
}

// vector_max
template <typename T, int vector_width>
inline void vector_max(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svfloat32_t vc = svmax_f32_m(pg, va, vb);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svfloat64_t vc = svmax_f64_m(pg, va, vb);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = std::max(a[i], b[i]);

#endif
}

// vector_max_w_scalar
template <typename T, int vector_width>
inline void vector_max_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vc = svmax_f32_m(pg, va, vconst);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vc = svmax_f64_m(pg, va, vconst);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = std::max(a[i], constant);

#endif
}

// ============================================================================
// Comparisons (result in 0.0 / 1.0)
// ============================================================================

// vector_gt
template <typename T, int vector_width>
inline void vector_gt(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svbool_t m = svcmpgt_f32(pg, va, vb);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svbool_t m = svcmpgt_f64(pg, va, vb);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] > b[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_gt_w_scalar (a[i] > constant)
template <typename T, int vector_width>
inline void vector_gt_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmpgt_f32(pg, va, vconst);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmpgt_f64(pg, va, vconst);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] > constant) ? T(1.0) : T(0.0);

#endif
}

// vector_gt_w_scalar_c (constant > a[i])
template <typename T, int vector_width>
inline void vector_gt_w_scalar_c(T* __restrict__ out, const T constant,
                                 const T* __restrict__ a) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmpgt_f32(pg, vconst, va);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmpgt_f64(pg, vconst, va);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (constant > a[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_lt
template <typename T, int vector_width>
inline void vector_lt(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svbool_t m = svcmplt_f32(pg, va, vb);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svbool_t m = svcmplt_f64(pg, va, vb);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] < b[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_lt_w_scalar (a[i] < constant)
template <typename T, int vector_width>
inline void vector_lt_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmplt_f32(pg, va, vconst);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmplt_f64(pg, va, vconst);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] < constant) ? T(1.0) : T(0.0);

#endif
}

// vector_lt_w_scalar_c (constant < a[i])
template <typename T, int vector_width>
inline void vector_lt_w_scalar_c(T* __restrict__ out, const T constant,
                                 const T* __restrict__ a) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmplt_f32(pg, vconst, va);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmplt_f64(pg, vconst, va);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (constant < a[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_ge
template <typename T, int vector_width>
inline void vector_ge(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svbool_t m = svcmpge_f32(pg, va, vb);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svbool_t m = svcmpge_f64(pg, va, vb);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] >= b[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_ge_w_scalar (a[i] >= constant)
template <typename T, int vector_width>
inline void vector_ge_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmpge_f32(pg, va, vconst);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmpge_f64(pg, va, vconst);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] >= constant) ? T(1.0) : T(0.0);

#endif
}

// vector_ge_w_scalar_c (constant >= a[i])
template <typename T, int vector_width>
inline void vector_ge_w_scalar_c(T* __restrict__ out, const T constant,
                                 const T* __restrict__ a) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmpge_f32(pg, vconst, va);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmpge_f64(pg, vconst, va);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (constant >= a[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_le
template <typename T, int vector_width>
inline void vector_le(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svbool_t m = svcmple_f32(pg, va, vb);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svbool_t m = svcmple_f64(pg, va, vb);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] <= b[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_le_w_scalar (a[i] <= constant)
template <typename T, int vector_width>
inline void vector_le_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmple_f32(pg, va, vconst);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmple_f64(pg, va, vconst);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] <= constant) ? T(1.0) : T(0.0);

#endif
}

// vector_le_w_scalar_c (constant <= a[i])
template <typename T, int vector_width>
inline void vector_le_w_scalar_c(T* __restrict__ out, const T constant,
                                 const T* __restrict__ a) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmple_f32(pg, vconst, va);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmple_f64(pg, vconst, va);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (constant <= a[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_eq
template <typename T, int vector_width>
inline void vector_eq(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svbool_t m = svcmpeq_f32(pg, va, vb);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svbool_t m = svcmpeq_f64(pg, va, vb);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] == b[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_eq_w_scalar
template <typename T, int vector_width>
inline void vector_eq_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmpeq_f32(pg, va, vconst);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmpeq_f64(pg, va, vconst);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] == constant) ? T(1.0) : T(0.0);

#endif
}

// vector_ne
template <typename T, int vector_width>
inline void vector_ne(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svfloat32_t vb = svld1_f32(pg, b + i);
      svbool_t m = svcmpne_f32(pg, va, vb);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svfloat64_t vb = svld1_f64(pg, b + i);
      svbool_t m = svcmpne_f64(pg, va, vb);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] != b[i]) ? T(1.0) : T(0.0);

#endif
}

// vector_ne_w_scalar
template <typename T, int vector_width>
inline void vector_ne_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                               const T constant) {
#if defined(__ARM_FEATURE_SVE)

  if constexpr (std::is_same<T, float>::value) {
    svfloat32_t vconst = svdup_f32(constant);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t zero = svdup_f32(0.0f);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b32(i, vector_width);
      svfloat32_t va = svld1_f32(pg, a + i);
      svbool_t m = svcmpne_f32(pg, va, vconst);
      svfloat32_t vc = svsel_f32(m, one, zero);
      svst1_f32(pg, out + i, vc);
      i += svcntw();
    }
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    svfloat64_t vconst = svdup_f64(constant);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t zero = svdup_f64(0.0);
    int i = 0;
    while (i < vector_width) {
      svbool_t pg = svwhilelt_b64(i, vector_width);
      svfloat64_t va = svld1_f64(pg, a + i);
      svbool_t m = svcmpne_f64(pg, va, vconst);
      svfloat64_t vc = svsel_f64(m, one, zero);
      svst1_f64(pg, out + i, vc);
      i += svcntd();
    }
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] != constant) ? T(1.0) : T(0.0);

#endif
}

template <typename T, int vector_width, typename CondT = bool>
inline void vector_select(T* __restrict__ out, const CondT* __restrict__ cond,
                          const T* __restrict__ t, const T* __restrict__ e) {
  for (int i = 0; i < vector_width; ++i) out[i] = cond[i] ? t[i] : e[i];
}

// ============================================================================
// Runtime-length scatter / gather / strided load+store (moved from
// vector_intrinsics/{gather,scatter,strided_load,strided_store}.h).
// SVE uses native gather/scatter intrinsics with svwhilelt-driven predicates;
// the whole loop runs without remainder.
// ============================================================================

#include <stdint.h>

template <typename T>
inline void gather(const T* __restrict__ A, const int64_t* __restrict__ idx,
                   T* __restrict__ B, const int64_t length) {
#if defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    while (i < length) {
      svbool_t pg = svwhilelt_b64(i, length);
      svint64_t vindex = svld1_s64(pg, &idx[i]);
      svfloat64_t vdata =
          svld1_gather_s64index_f64(pg, (const double*)A, vindex);
      svst1_f64(pg, (double*)&B[i], vdata);
      i += svcntd();
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[i] = A[idx[i]];
}

template <typename T>
inline void scatter(const T* __restrict__ A, const int64_t* __restrict__ idx,
                    T* __restrict__ B, const int64_t length) {
#if defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    while (i < length) {
      svbool_t pg = svwhilelt_b64(i, length);
      svfloat64_t vdata = svld1_f64(pg, (const double*)&A[i]);
      svint64_t vindex = svld1_s64(pg, &idx[i]);
      svst1_scatter_s64index_f64(pg, (double*)B, vindex, vdata);
      i += svcntd();
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[idx[i]] = A[i];
}

template <typename T>
inline void strided_load(const T* __restrict__ A, T* __restrict__ B,
                         const int64_t length, const int64_t stride) {
#if defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    while (i < length) {
      svbool_t pg = svwhilelt_b64(i, length);
      svint64_t vi = svindex_s64(i, 1);
      svint64_t vindex = svmul_n_s64_x(pg, vi, stride);
      svfloat64_t vdata =
          svld1_gather_s64index_f64(pg, (const double*)A, vindex);
      svst1_f64(pg, (double*)&B[i], vdata);
      i += svcntd();
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[i] = A[i * stride];
}

template <typename T>
inline void strided_store(const T* __restrict__ A, T* __restrict__ B,
                          const int64_t length, const int64_t stride) {
#if defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    while (i < length) {
      svbool_t pg = svwhilelt_b64(i, length);
      svfloat64_t vdata = svld1_f64(pg, (const double*)&A[i]);
      svint64_t vi = svindex_s64(i, 1);
      svint64_t vindex = svmul_n_s64_x(pg, vi, stride);
      svst1_scatter_s64index_f64(pg, (double*)B, vindex, vdata);
      i += svcntd();
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[i * stride] = A[i];
}

// --------------------------- masked variants (RMW) ---------------------------
// SVE has native gather/scatter intrinsics with an svbool_t predicate.
// Build the predicate by AND-ing the whilelt iteration predicate with the
// user mask (loaded as u64-per-lane then compared against zero). Inactive
// lanes leave destination memory unchanged.

#if defined(__ARM_FEATURE_SVE)
static inline svbool_t _dace_load_bool_mask_b64(const bool* __restrict__ mask,
                                                int64_t i, int64_t length) {
  // Materialize one mask lane per active u64 element via a per-lane fill
  // to a uint64 temp buffer. Keeps the contract simple at the cost of a
  // small stack buffer; SVE's vector-length is implementation-defined so
  // we don't assume svcntd <= 16. The caller has already bounded the
  // iteration via svwhilelt_b64 to the lane count, so the fill is bounded.
  uint64_t buf[svcntd()];
  for (int64_t lane = 0; lane < (int64_t)svcntd(); ++lane) {
    const int64_t pos = i + lane;
    buf[lane] = (pos < length && mask[pos]) ? UINT64_C(~0) : UINT64_C(0);
  }
  svbool_t pg_iter = svwhilelt_b64(i, length);
  svuint64_t user = svld1_u64(pg_iter, buf);
  return svcmpne_n_u64(pg_iter, user, 0);
}
#endif

template <typename T>
inline void gather_masked(const T* __restrict__ A,
                          const int64_t* __restrict__ idx, T* __restrict__ B,
                          const int64_t length, const bool* __restrict__ mask) {
#if defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    while (i < length) {
      svbool_t pg = _dace_load_bool_mask_b64(mask, i, length);
      svint64_t vindex = svld1_s64(pg, &idx[i]);
      svfloat64_t vdata =
          svld1_gather_s64index_f64(pg, (const double*)A, vindex);
      svst1_f64(pg, (double*)&B[i], vdata);
      i += svcntd();
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
#if defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    while (i < length) {
      svbool_t pg = _dace_load_bool_mask_b64(mask, i, length);
      svfloat64_t vdata = svld1_f64(pg, (const double*)&A[i]);
      svint64_t vindex = svld1_s64(pg, &idx[i]);
      svst1_scatter_s64index_f64(pg, (double*)B, vindex, vdata);
      i += svcntd();
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
#if defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    while (i < length) {
      svbool_t pg = _dace_load_bool_mask_b64(mask, i, length);
      svint64_t vi = svindex_s64(i, 1);
      svint64_t vindex = svmul_n_s64_x(pg, vi, stride);
      svfloat64_t vdata =
          svld1_gather_s64index_f64(pg, (const double*)A, vindex);
      svst1_f64(pg, (double*)&B[i], vdata);
      i += svcntd();
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
#if defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    while (i < length) {
      svbool_t pg = _dace_load_bool_mask_b64(mask, i, length);
      svfloat64_t vdata = svld1_f64(pg, (const double*)&A[i]);
      svint64_t vi = svindex_s64(i, 1);
      svint64_t vindex = svmul_n_s64_x(pg, vi, stride);
      svst1_scatter_s64index_f64(pg, (double*)B, vindex, vdata);
      i += svcntd();
    }
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[i * stride] = A[i];
}

// ---------------------- horizontal reductions ----------------------
// SVE one-shot reduce for sum / max / min over the floating-point
// accumulator types (svaddv / svmaxv / svminv), VL-agnostic: a
// predicated chunk loop accumulates lane-wise (svwhilelt tail
// predicate) then a single across-vector reduce. Product, bitwise and
// every non-floating type delegate to the portable log-depth tree in
// the common header (SVE has no svmulv; bitwise reductions are
// integer-only and never emitted on an fp accumulator). SFINAE keeps
// the tree as the correctness safety net. (Numeric behaviour validated
// on SVE hardware; types/intrinsic names validated via aarch64+sve
// ``-fsyntax-only``.)
#if defined(__ARM_FEATURE_SVE)
template <typename T, int vector_width>
inline typename std::enable_if<std::is_same<T, double>::value, T>::type
horizontal_reduce_add(const T* __restrict__ a) {
  svfloat64_t acc = svdup_n_f64(0.0);
  for (int64_t i = 0; i < vector_width; i += (int64_t)svcntd()) {
    svbool_t pg = svwhilelt_b64(i, (int64_t)vector_width);
    acc = svadd_f64_m(pg, acc, svld1_f64(pg, a + i));
  }
  return svaddv_f64(svptrue_b64(), acc);
}
template <typename T, int vector_width>
inline typename std::enable_if<std::is_same<T, float>::value, T>::type
horizontal_reduce_add(const T* __restrict__ a) {
  svfloat32_t acc = svdup_n_f32(0.0f);
  for (int64_t i = 0; i < vector_width; i += (int64_t)svcntw()) {
    svbool_t pg = svwhilelt_b32(i, (int64_t)vector_width);
    acc = svadd_f32_m(pg, acc, svld1_f32(pg, a + i));
  }
  return svaddv_f32(svptrue_b32(), acc);
}
template <typename T, int vector_width>
inline typename std::enable_if<std::is_same<T, double>::value, T>::type
horizontal_reduce_max(const T* __restrict__ a) {
  svfloat64_t acc = svdup_n_f64(-INFINITY);
  for (int64_t i = 0; i < vector_width; i += (int64_t)svcntd()) {
    svbool_t pg = svwhilelt_b64(i, (int64_t)vector_width);
    acc = svmax_f64_m(pg, acc, svld1_f64(pg, a + i));
  }
  return svmaxv_f64(svptrue_b64(), acc);
}
template <typename T, int vector_width>
inline typename std::enable_if<std::is_same<T, float>::value, T>::type
horizontal_reduce_max(const T* __restrict__ a) {
  svfloat32_t acc = svdup_n_f32(-INFINITY);
  for (int64_t i = 0; i < vector_width; i += (int64_t)svcntw()) {
    svbool_t pg = svwhilelt_b32(i, (int64_t)vector_width);
    acc = svmax_f32_m(pg, acc, svld1_f32(pg, a + i));
  }
  return svmaxv_f32(svptrue_b32(), acc);
}
template <typename T, int vector_width>
inline typename std::enable_if<std::is_same<T, double>::value, T>::type
horizontal_reduce_min(const T* __restrict__ a) {
  svfloat64_t acc = svdup_n_f64(INFINITY);
  for (int64_t i = 0; i < vector_width; i += (int64_t)svcntd()) {
    svbool_t pg = svwhilelt_b64(i, (int64_t)vector_width);
    acc = svmin_f64_m(pg, acc, svld1_f64(pg, a + i));
  }
  return svminv_f64(svptrue_b64(), acc);
}
template <typename T, int vector_width>
inline typename std::enable_if<std::is_same<T, float>::value, T>::type
horizontal_reduce_min(const T* __restrict__ a) {
  svfloat32_t acc = svdup_n_f32(INFINITY);
  for (int64_t i = 0; i < vector_width; i += (int64_t)svcntw()) {
    svbool_t pg = svwhilelt_b32(i, (int64_t)vector_width);
    acc = svmin_f32_m(pg, acc, svld1_f32(pg, a + i));
  }
  return svminv_f32(svptrue_b32(), acc);
}
template <typename T, int vector_width>
inline typename std::enable_if<!std::is_same<T, double>::value &&
                                   !std::is_same<T, float>::value,
                               T>::type
horizontal_reduce_add(const T* __restrict__ a) {
  return _dace_horizontal_tree_add<T, vector_width>(a);
}
template <typename T, int vector_width>
inline typename std::enable_if<!std::is_same<T, double>::value &&
                                   !std::is_same<T, float>::value,
                               T>::type
horizontal_reduce_max(const T* __restrict__ a) {
  return _dace_horizontal_tree_max<T, vector_width>(a);
}
template <typename T, int vector_width>
inline typename std::enable_if<!std::is_same<T, double>::value &&
                                   !std::is_same<T, float>::value,
                               T>::type
horizontal_reduce_min(const T* __restrict__ a) {
  return _dace_horizontal_tree_min<T, vector_width>(a);
}
template <typename T, int vector_width>
inline T horizontal_reduce_mul(const T* __restrict__ a) {
  return _dace_horizontal_tree_mul<T, vector_width>(a);
}
template <typename T, int vector_width>
inline T horizontal_reduce_band(const T* __restrict__ a) {
  return _dace_horizontal_tree_band<T, vector_width>(a);
}
template <typename T, int vector_width>
inline T horizontal_reduce_bor(const T* __restrict__ a) {
  return _dace_horizontal_tree_bor<T, vector_width>(a);
}
template <typename T, int vector_width>
inline T horizontal_reduce_bxor(const T* __restrict__ a) {
  return _dace_horizontal_tree_bxor<T, vector_width>(a);
}
#endif
