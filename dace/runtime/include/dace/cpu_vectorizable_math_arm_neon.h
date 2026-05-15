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

#if defined(__ARM_NEON)
#include <arm_neon.h>
// Option F overlay: per-op escape-hatch siblings (vector_<op>_pscalar /
// vector_<op>_av). Arch-independent; one source of truth, included by
// every arch file so the binary always has them regardless of which
// backend the dispatcher selected.
#include "dace/cpu_vectorizable_math_common.h"
#endif

#if !defined(__ARM_NEON)
#error Included the Neon header without support for Neon
#endif

// vector_mult
template <typename T, int vector_width>
inline void vector_mult(T* __restrict__ out, const T* __restrict__ a,
                        const T* __restrict__ b) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;  // float32x4_t
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      float32x4_t vc = vmulq_f32(va, vb);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] * b[i];
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;  // float64x2_t
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      float64x2_t vc = vmulq_f64(va, vb);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] * b[i];
    return;
  }

#else  // __ARM_NEON

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] * b[i];

#endif
}

// vector_mult_w_scalar
template <typename T, int vector_width>
inline void vector_mult_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                 const T constant) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vc = vmulq_f32(va, vconst);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] * constant;
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vc = vmulq_f64(va, vconst);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] * constant;
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;  // NEON float32x4_t
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      float32x4_t vc = vaddq_f32(va, vb);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] + b[i];
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;  // NEON float64x2_t
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      float64x2_t vc = vaddq_f64(va, vb);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] + b[i];
    return;
  }

#else  // __ARM_NEON

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] + b[i];

#endif
}

// vector_add_w_scalar
template <typename T, int vector_width>
inline void vector_add_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vc = vaddq_f32(va, vconst);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] + constant;
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vc = vaddq_f64(va, vconst);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] + constant;
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] + constant;

#endif
}

// vector_add_masked: store ONLY lanes where mask[i] == true; inactive
// lanes are left untouched.
//
// ARMv8-A AdvSIMD (NEON) has no predicated / per-lane masked store —
// every ``vst1q`` writes the full 128-bit register. The previous
// load-blend-store (``vld1q(out); vbslq; vst1q``) wrote the old value
// back on inactive lanes, but that store still touches all W addresses:
// at a masked remainder the writeback target is ``arr + tile_i`` and the
// trailing inactive lanes index past the array end -> OOB heap write
// (the TSVC s2710 masked-merge-65 segfault). NEON cannot express a
// masked store, so the only OOB-safe form is a scalar gated store; the
// compiler still autovectorises the active in-bounds lanes.
template <typename T, int vector_width>
inline void vector_add_masked(T* __restrict__ out, const T* __restrict__ a,
                              const T* __restrict__ b,
                              const bool* __restrict__ mask) {
  for (int i = 0; i < vector_width; ++i)
    if (mask[i]) out[i] = a[i] + b[i];
}

// vector_add_w_scalar_masked: scalar gated store for the same reason as
// vector_add_masked above (NEON has no masked store; a full-width vst1q
// at a masked remainder writes past the array end).
template <typename T, int vector_width>
inline void vector_add_w_scalar_masked(T* __restrict__ out,
                                       const T* __restrict__ a,
                                       const T constant,
                                       const bool* __restrict__ mask) {
  for (int i = 0; i < vector_width; ++i)
    if (mask[i]) out[i] = a[i] + constant;
}

// vector_sub
template <typename T, int vector_width>
inline void vector_sub(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      float32x4_t vc = vsubq_f32(va, vb);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] - b[i];
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      float64x2_t vc = vsubq_f64(va, vb);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] - b[i];
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = a[i] - b[i];

#endif
}

// vector_sub_w_scalar
template <typename T, int vector_width>
inline void vector_sub_w_scalar(T* __restrict__ out, const T* __restrict__ a,
                                const T constant) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vc = vsubq_f32(va, vconst);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] - constant;
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vc = vsubq_f64(va, vconst);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] - constant;
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vc = vsubq_f32(vconst, va);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = constant - a[i];
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vc = vsubq_f64(vconst, va);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = constant - a[i];
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      float32x4_t vc = vdivq_f32(va, vb);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] / b[i];
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      float64x2_t vc = vdivq_f64(va, vb);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] / b[i];
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vc = vdivq_f32(va, vconst);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] / constant;
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vc = vdivq_f64(va, vconst);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = a[i] / constant;
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vc = vdivq_f32(vconst, va);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = constant / a[i];
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vc = vdivq_f64(vconst, va);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = constant / a[i];
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = constant / a[i];

#endif
}

// vector_copy
template <typename T, int vector_width>
inline void vector_copy(T* __restrict__ dst, const T* __restrict__ src) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t v = vld1q_f32(src + i);
      vst1q_f32(dst + i, v);
    }
    for (; i < vector_width; ++i) dst[i] = src[i];
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t v = vld1q_f64(src + i);
      vst1q_f64(dst + i, v);
    }
    for (; i < vector_width; ++i) dst[i] = src[i];
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) dst[i] = src[i];

#endif
}

// vector_copy_w_scalar
template <typename T, int vector_width>
inline void vector_copy_w_scalar(T* __restrict__ dst, const T a) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + W <= vector_width; i += W) vst1q_f32(dst + i, va);
    for (; i < vector_width; ++i) dst[i] = a;
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t va = vdupq_n_f64(a);
    int i = 0;
    for (; i + W <= vector_width; i += W) vst1q_f64(dst + i, va);
    for (; i < vector_width; ++i) dst[i] = a;
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) dst[i] = a;

#endif
}

// ============================================================================
// Min / Max
// ============================================================================

// vector_min
template <typename T, int vector_width>
inline void vector_min(T* __restrict__ out, const T* __restrict__ a,
                       const T* __restrict__ b) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      float32x4_t vc = vminq_f32(va, vb);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = std::min(a[i], b[i]);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      float64x2_t vc = vminq_f64(va, vb);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = std::min(a[i], b[i]);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vc = vminq_f32(va, vconst);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = std::min(a[i], constant);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vc = vminq_f64(va, vconst);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = std::min(a[i], constant);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      float32x4_t vc = vmaxq_f32(va, vb);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = std::max(a[i], b[i]);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      float64x2_t vc = vmaxq_f64(va, vb);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = std::max(a[i], b[i]);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vc = vmaxq_f32(va, vconst);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = std::max(a[i], constant);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vc = vmaxq_f64(va, vconst);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = std::max(a[i], constant);
    return;
  }

#else

  for (int i = 0; i < vector_width; ++i) out[i] = std::max(a[i], constant);

#endif
}

// ============================================================================
// Comparisons (result in 0.0 / 1.0)
// ============================================================================

// Helper: build 0/1 vectors for float32x4 / float64x2 masks
#if defined(__ARM_NEON)
inline float32x4_t neon_select_0_1_f32(uint32x4_t mask) {
  float32x4_t vzero = vdupq_n_f32(0.0f);
  float32x4_t vone = vdupq_n_f32(1.0f);
  return vbslq_f32(mask, vone, vzero);
}
inline float64x2_t neon_select_0_1_f64(uint64x2_t mask) {
  float64x2_t vzero = vdupq_n_f64(0.0);
  float64x2_t vone = vdupq_n_f64(1.0);
  return vbslq_f64(mask, vone, vzero);
}
#endif

// vector_gt
template <typename T, int vector_width>
inline void vector_gt(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      uint32x4_t m = vcgtq_f32(va, vb);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] > b[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      uint64x2_t m = vcgtq_f64(va, vb);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] > b[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vcgtq_f32(va, vconst);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] > constant) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vcgtq_f64(va, vconst);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] > constant) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vcgtq_f32(vconst, va);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (constant > a[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vcgtq_f64(vconst, va);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (constant > a[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      uint32x4_t m = vcltq_f32(va, vb);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] < b[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      uint64x2_t m = vcltq_f64(va, vb);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] < b[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vcltq_f32(va, vconst);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] < constant) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vcltq_f64(va, vconst);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] < constant) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vcltq_f32(vconst, va);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (constant < a[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vcltq_f64(vconst, va);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (constant < a[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      uint32x4_t m = vcgeq_f32(va, vb);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] >= b[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      uint64x2_t m = vcgeq_f64(va, vb);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] >= b[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vcgeq_f32(va, vconst);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] >= constant) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vcgeq_f64(va, vconst);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] >= constant) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vcgeq_f32(vconst, va);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (constant >= a[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vcgeq_f64(vconst, va);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (constant >= a[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      uint32x4_t m = vcleq_f32(va, vb);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] <= b[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      uint64x2_t m = vcleq_f64(va, vb);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] <= b[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vcleq_f32(va, vconst);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] <= constant) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vcleq_f64(va, vconst);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] <= constant) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vcleq_f32(vconst, va);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (constant <= a[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vcleq_f64(vconst, va);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (constant <= a[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      uint32x4_t m = vceqq_f32(va, vb);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] == b[i]) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      uint64x2_t m = vceqq_f64(va, vb);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] == b[i]) ? T(1.0) : T(0.0);
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
#if defined(__ARM_NEON)

  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    float32x4_t vconst = vdupq_n_f32(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      uint32x4_t m = vceqq_f32(va, vconst);
      float32x4_t vc = neon_select_0_1_f32(m);
      vst1q_f32(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] == constant) ? T(1.0) : T(0.0);
    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    float64x2_t vconst = vdupq_n_f64(constant);
    int i = 0;
    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      uint64x2_t m = vceqq_f64(va, vconst);
      float64x2_t vc = neon_select_0_1_f64(m);
      vst1q_f64(out + i, vc);
    }
    for (; i < vector_width; ++i) out[i] = (a[i] == constant) ? T(1.0) : T(0.0);
    return;
  }

#else
  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] == constant) ? T(1.0) : T(0.0);

#endif
}

#include <arm_neon.h>

template <typename T, int vector_width>
inline void vector_ne(T* __restrict__ out, const T* __restrict__ a,
                      const T* __restrict__ b) {
#if defined(__ARM_NEON)

  // ---------------------------------------------------------
  // FLOAT32 version (W = 4)
  // ---------------------------------------------------------
  if constexpr (std::is_same<T, float>::value) {
    constexpr int W = 4;
    int i = 0;

    for (; i + W <= vector_width; i += W) {
      float32x4_t va = vld1q_f32(a + i);
      float32x4_t vb = vld1q_f32(b + i);
      uint32x4_t meq = vceqq_f32(va, vb);
      uint32x4_t mne = vmvnq_u32(meq);
      uint32x4_t onezero = vshrq_n_u32(mne, 31);
      float32x4_t vc = vcvtq_f32_u32(onezero);
      vst1q_f32(out + i, vc);
    }

    for (; i < vector_width; ++i) out[i] = (a[i] != b[i]) ? 1.0f : 0.0f;

    return;
  }

  if constexpr (std::is_same<T, double>::value) {
    constexpr int W = 2;
    int i = 0;

    for (; i + W <= vector_width; i += W) {
      float64x2_t va = vld1q_f64(a + i);
      float64x2_t vb = vld1q_f64(b + i);
      uint64x2_t meq = vceqq_f64(va, vb);
      const uint64x2_t all1 = vdupq_n_u64(~0ULL);
      uint64x2_t mne = veorq_u64(meq, all1);
      uint64x2_t onezero = vshrq_n_u64(mne, 63);
      float64x2_t vc = vcvtq_f64_u64(onezero);
      vst1q_f64(out + i, vc);
    }

    for (; i < vector_width; ++i) out[i] = (a[i] != b[i]) ? 1.0 : 0.0;

    return;
  }
#endif
  for (int i = 0; i < vector_width; ++i)
    out[i] = (a[i] != b[i]) ? T(1.0) : T(0.0);
}

// ============================================================================
// Elementwise non-linear (always scalar)
// ============================================================================

template <typename T, int vector_width>
inline void vector_exp(T* __restrict__ out, const T* __restrict__ a) {
  for (int i = 0; i < vector_width; ++i) out[i] = std::exp(a[i]);
}

template <typename T, int vector_width>
inline void vector_log(T* __restrict__ out, const T* __restrict__ a) {
  for (int i = 0; i < vector_width; ++i) out[i] = std::log(a[i]);
}

template <typename T, int vector_width, typename CondT = bool>
inline void vector_select(T* __restrict__ out, const CondT* __restrict__ cond,
                          const T* __restrict__ t, const T* __restrict__ e) {
  for (int i = 0; i < vector_width; ++i) out[i] = cond[i] ? t[i] : e[i];
}

// ============================================================================
// Runtime-length scatter / gather / strided load+store (moved from
// vector_intrinsics/{gather,scatter,strided_load,strided_store}.h).
// NEON has no native gather/scatter; we pack 2-lane chunks where possible
// and fall back to scalar.
// ============================================================================

#include <stdint.h>

template <typename T>
inline void gather(const T* __restrict__ A, const int64_t* __restrict__ idx,
                   T* __restrict__ B, const int64_t length) {
#if defined(__ARM_NEON)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    for (; i + 1 < length; i += 2) {
      float64x2_t vdata = {A[idx[i]], A[idx[i + 1]]};
      vst1q_f64((double*)&B[i], vdata);
    }
    for (; i < length; ++i) B[i] = A[idx[i]];
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[i] = A[idx[i]];
}

template <typename T>
inline void scatter(const T* __restrict__ A, const int64_t* __restrict__ idx,
                    T* __restrict__ B, const int64_t length) {
  for (int64_t i = 0; i < length; ++i) B[idx[i]] = A[i];
}

template <typename T>
inline void strided_load(const T* __restrict__ A, T* __restrict__ B,
                         const int64_t length, const int64_t stride) {
#if defined(__ARM_NEON)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    for (; i + 1 < length; i += 2) {
      float64x2_t vdata = {A[(i + 0) * stride], A[(i + 1) * stride]};
      vst1q_f64((double*)&B[i], vdata);
    }
    for (; i < length; ++i) B[i] = A[i * stride];
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[i] = A[i * stride];
}

template <typename T>
inline void strided_store(const T* __restrict__ A, T* __restrict__ B,
                          const int64_t length, const int64_t stride) {
#if defined(__ARM_NEON)
  if constexpr (std::is_same<T, double>::value) {
    int64_t i = 0;
    for (; i + 1 < length; i += 2) {
      float64x2_t vdata = vld1q_f64((const double*)&A[i]);
      double tmp[2];
      vst1q_f64(tmp, vdata);
      B[(i + 0) * stride] = tmp[0];
      B[(i + 1) * stride] = tmp[1];
    }
    for (; i < length; ++i) B[i * stride] = A[i];
    return;
  }
#endif
  for (int64_t i = 0; i < length; ++i) B[i * stride] = A[i];
}

// --------------------------- masked variants (RMW) ---------------------------
// NEON has no native masked gather/scatter; scalar fallback gates each lane
// via mask[i]. Inactive lanes leave destination memory unchanged.

template <typename T>
inline void gather_masked(const T* __restrict__ A,
                          const int64_t* __restrict__ idx, T* __restrict__ B,
                          const int64_t length, const bool* __restrict__ mask) {
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[i] = A[idx[i]];
}

template <typename T>
inline void scatter_masked(const T* __restrict__ A,
                           const int64_t* __restrict__ idx, T* __restrict__ B,
                           const int64_t length,
                           const bool* __restrict__ mask) {
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[idx[i]] = A[i];
}

template <typename T>
inline void strided_load_masked(const T* __restrict__ A, T* __restrict__ B,
                                const int64_t length, const int64_t stride,
                                const bool* __restrict__ mask) {
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[i] = A[i * stride];
}

template <typename T>
inline void strided_store_masked(const T* __restrict__ A, T* __restrict__ B,
                                 const int64_t length, const int64_t stride,
                                 const bool* __restrict__ mask) {
  for (int64_t i = 0; i < length; ++i)
    if (mask[i]) B[i * stride] = A[i];
}
