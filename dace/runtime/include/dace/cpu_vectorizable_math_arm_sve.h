// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cmath>
#include <algorithm>

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
# error Included the SVE header without support SVE
#endif

// ============================================================================
// Arithmetic
// ============================================================================

// vector_mult
template<typename T, int vector_width>
inline void vector_mult(T* __restrict__ out,
                        const T* __restrict__ a,
                        const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = a[i] * b[i];

#endif
}

// vector_mult_w_scalar
template<typename T, int vector_width>
inline void vector_mult_w_scalar(T* __restrict__ out,
                                 const T* __restrict__ a,
                                 const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = a[i] * constant;

#endif
}

// vector_add
template<typename T, int vector_width>
inline void vector_add(T* __restrict__ out,
                       const T* __restrict__ a,
                       const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = a[i] + b[i];

#endif
}

// vector_add_w_scalar
template<typename T, int vector_width>
inline void vector_add_w_scalar(T* __restrict__ out,
                                const T* __restrict__ a,
                                const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = a[i] + constant;

#endif
}

// vector_add_masked: lanes where mask[i] == false leave out[i] unchanged
// (RMW). The bool[W] mask is loaded byte-wise into a uint64/uint32 lane
// vector, compared against zero to derive a svbool_t predicate. The sum
// is computed unmasked under svwhilelt_b{64,32} (the VLS tile predicate)
// and svsel blends it against the prior out where mask is false.
template<typename T, int vector_width>
inline void vector_add_masked(T* __restrict__ out,
                              const T* __restrict__ a,
                              const T* __restrict__ b,
                              const bool* __restrict__ mask)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        int i = 0;
        while (i < vector_width) {
            svbool_t pg = svwhilelt_b32(i, vector_width);
            svuint32_t mv = svld1ub_u32(pg, (const uint8_t*)(mask + i));
            svbool_t m = svcmpne_n_u32(pg, mv, 0);
            svfloat32_t va = svld1_f32(pg, a + i);
            svfloat32_t vb = svld1_f32(pg, b + i);
            svfloat32_t vold = svld1_f32(pg, out + i);
            svfloat32_t sum = svadd_f32_x(pg, va, vb);
            svst1_f32(pg, out + i, svsel_f32(m, sum, vold));
            i += svcntw();
        }
        return;
    }

    if constexpr (std::is_same<T,double>::value) {
        int i = 0;
        while (i < vector_width) {
            svbool_t pg = svwhilelt_b64(i, vector_width);
            svuint64_t mv = svld1ub_u64(pg, (const uint8_t*)(mask + i));
            svbool_t m = svcmpne_n_u64(pg, mv, 0);
            svfloat64_t va = svld1_f64(pg, a + i);
            svfloat64_t vb = svld1_f64(pg, b + i);
            svfloat64_t vold = svld1_f64(pg, out + i);
            svfloat64_t sum = svadd_f64_x(pg, va, vb);
            svst1_f64(pg, out + i, svsel_f64(m, sum, vold));
            i += svcntd();
        }
        return;
    }

#else

    for (int i = 0; i < vector_width; ++i)
        if (mask[i]) out[i] = a[i] + b[i];

#endif
}

// vector_add_w_scalar_masked
template<typename T, int vector_width>
inline void vector_add_w_scalar_masked(T* __restrict__ out,
                                       const T* __restrict__ a,
                                       const T constant,
                                       const bool* __restrict__ mask)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        int i = 0;
        while (i < vector_width) {
            svbool_t pg = svwhilelt_b32(i, vector_width);
            svuint32_t mv = svld1ub_u32(pg, (const uint8_t*)(mask + i));
            svbool_t m = svcmpne_n_u32(pg, mv, 0);
            svfloat32_t va = svld1_f32(pg, a + i);
            svfloat32_t vold = svld1_f32(pg, out + i);
            svfloat32_t sum = svadd_f32_x(pg, va, vconst);
            svst1_f32(pg, out + i, svsel_f32(m, sum, vold));
            i += svcntw();
        }
        return;
    }

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        int i = 0;
        while (i < vector_width) {
            svbool_t pg = svwhilelt_b64(i, vector_width);
            svuint64_t mv = svld1ub_u64(pg, (const uint8_t*)(mask + i));
            svbool_t m = svcmpne_n_u64(pg, mv, 0);
            svfloat64_t va = svld1_f64(pg, a + i);
            svfloat64_t vold = svld1_f64(pg, out + i);
            svfloat64_t sum = svadd_f64_x(pg, va, vconst);
            svst1_f64(pg, out + i, svsel_f64(m, sum, vold));
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
template<typename T, int vector_width>
inline void vector_sub(T* __restrict__ out,
                       const T* __restrict__ a,
                       const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = a[i] - b[i];

#endif
}

// vector_sub_w_scalar (a[i] - constant)
template<typename T, int vector_width>
inline void vector_sub_w_scalar(T* __restrict__ out,
                                const T* __restrict__ a,
                                const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = a[i] - constant;

#endif
}

// vector_sub_w_scalar_c (constant - a[i])
template<typename T, int vector_width>
inline void vector_sub_w_scalar_c(T* __restrict__ out,
                                  const T constant,
                                  const T* __restrict__ a)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = constant - a[i];

#endif
}

// vector_div
template<typename T, int vector_width>
inline void vector_div(T* __restrict__ out,
                       const T* __restrict__ a,
                       const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = a[i] / b[i];

#endif
}

// vector_div_w_scalar (a[i] / constant)
template<typename T, int vector_width>
inline void vector_div_w_scalar(T* __restrict__ out,
                                const T* __restrict__ a,
                                const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = a[i] / constant;

#endif
}

// vector_div_w_scalar_c (constant / a[i])
template<typename T, int vector_width>
inline void vector_div_w_scalar_c(T* __restrict__ out,
                                  const T constant,
                                  const T* __restrict__ a)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = constant / a[i];

#endif
}

// vector_copy
template<typename T, int vector_width>
inline void vector_copy(T* __restrict__ dst,
                        const T* __restrict__ src)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        int i = 0;
        while (i < vector_width) {
            svbool_t pg = svwhilelt_b32(i, vector_width);
            svfloat32_t v = svld1_f32(pg, src + i);
            svst1_f32(pg, dst + i, v);
            i += svcntw();
        }
        return;
    }

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        dst[i] = src[i];

#endif
}

// vector_copy_w_scalar
template<typename T, int vector_width>
inline void vector_copy_w_scalar(T* __restrict__ dst,
                                 const T a)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t va = svdup_f32(a);
        int i = 0;
        while (i < vector_width) {
            svbool_t pg = svwhilelt_b32(i, vector_width);
            svst1_f32(pg, dst + i, va);
            i += svcntw();
        }
        return;
    }

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        dst[i] = a;

#endif
}

// ============================================================================
// Elementwise non-linear (scalar only as per your requirement)
// ============================================================================

template<typename T, int vector_width>
inline void vector_exp(T* __restrict__ out,
                       const T* __restrict__ a)
{
    for (int i = 0; i < vector_width; ++i)
        out[i] = std::exp(a[i]);
}

template<typename T, int vector_width>
inline void vector_log(T* __restrict__ out,
                       const T* __restrict__ a)
{
    for (int i = 0; i < vector_width; ++i)
        out[i] = std::log(a[i]);
}

// ============================================================================
// Min / Max
// ============================================================================

// vector_min
template<typename T, int vector_width>
inline void vector_min(T* __restrict__ out,
                       const T* __restrict__ a,
                       const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = std::min(a[i], b[i]);

#endif
}

// vector_min_w_scalar
template<typename T, int vector_width>
inline void vector_min_w_scalar(T* __restrict__ out,
                                const T* __restrict__ a,
                                const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = std::min(a[i], constant);

#endif
}

// vector_max
template<typename T, int vector_width>
inline void vector_max(T* __restrict__ out,
                       const T* __restrict__ a,
                       const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = std::max(a[i], b[i]);

#endif
}

// vector_max_w_scalar
template<typename T, int vector_width>
inline void vector_max_w_scalar(T* __restrict__ out,
                                const T* __restrict__ a,
                                const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
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

    if constexpr (std::is_same<T,double>::value) {
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

    for (int i = 0; i < vector_width; ++i)
        out[i] = std::max(a[i], constant);

#endif
}

// ============================================================================
// Comparisons (result in 0.0 / 1.0)
// ============================================================================

// vector_gt
template<typename T, int vector_width>
inline void vector_gt(T* __restrict__ out,
                      const T* __restrict__ a,
                      const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_gt_w_scalar(T* __restrict__ out,
                               const T* __restrict__ a,
                               const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_gt_w_scalar_c(T* __restrict__ out,
                                 const T constant,
                                 const T* __restrict__ a)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_lt(T* __restrict__ out,
                      const T* __restrict__ a,
                      const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_lt_w_scalar(T* __restrict__ out,
                               const T* __restrict__ a,
                               const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_lt_w_scalar_c(T* __restrict__ out,
                                 const T constant,
                                 const T* __restrict__ a)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_ge(T* __restrict__ out,
                      const T* __restrict__ a,
                      const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_ge_w_scalar(T* __restrict__ out,
                               const T* __restrict__ a,
                               const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_ge_w_scalar_c(T* __restrict__ out,
                                 const T constant,
                                 const T* __restrict__ a)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_le(T* __restrict__ out,
                      const T* __restrict__ a,
                      const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_le_w_scalar(T* __restrict__ out,
                               const T* __restrict__ a,
                               const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_le_w_scalar_c(T* __restrict__ out,
                                 const T constant,
                                 const T* __restrict__ a)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_eq(T* __restrict__ out,
                      const T* __restrict__ a,
                      const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_eq_w_scalar(T* __restrict__ out,
                               const T* __restrict__ a,
                               const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_ne(T* __restrict__ out,
                      const T* __restrict__ a,
                      const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t one  = svdup_f64(1.0);
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
template<typename T, int vector_width>
inline void vector_ne_w_scalar(T* __restrict__ out,
                               const T* __restrict__ a,
                               const T constant)
{
#if defined(__ARM_FEATURE_SVE)

    if constexpr (std::is_same<T,float>::value) {
        svfloat32_t vconst = svdup_f32(constant);
        svfloat32_t one  = svdup_f32(1.0f);
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

    if constexpr (std::is_same<T,double>::value) {
        svfloat64_t vconst = svdup_f64(constant);
        svfloat64_t one  = svdup_f64(1.0);
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

template<typename T, int vector_width, typename CondT = bool>
inline void vector_select(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                          const T* __restrict__ e)
{
    for (int i = 0; i < vector_width; ++i)
        out[i] = cond[i] ? t[i] : e[i];
}

// ============================================================================
// Runtime-length scatter / gather / strided load+store (moved from
// vector_intrinsics/{gather,scatter,strided_load,strided_store}.h).
// SVE uses native gather/scatter intrinsics with svwhilelt-driven predicates;
// the whole loop runs without remainder.
// ============================================================================

#include <stdint.h>

inline void gather_double(const double* __restrict__ A,
                          const int64_t* __restrict__ idx,
                          double* __restrict__ B,
                          const int64_t length)
{
    int64_t i = 0;
    while (i < length) {
        svbool_t pg = svwhilelt_b64(i, length);
        svint64_t vindex = svld1_s64(pg, &idx[i]);
        svfloat64_t vdata = svld1_gather_s64index_f64(pg, A, vindex);
        svst1_f64(pg, &B[i], vdata);
        i += svcntd();
    }
}

inline void scatter_double(const double* __restrict__ A,
                           const int64_t* __restrict__ idx,
                           double* __restrict__ B,
                           const int64_t length)
{
    int64_t i = 0;
    while (i < length) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vdata = svld1_f64(pg, &A[i]);
        svint64_t vindex = svld1_s64(pg, &idx[i]);
        svst1_scatter_s64index_f64(pg, B, vindex, vdata);
        i += svcntd();
    }
}

inline void strided_load_double(const double* __restrict__ A,
                                double* __restrict__ B,
                                const int64_t length,
                                const int64_t stride)
{
    int64_t i = 0;
    while (i < length) {
        svbool_t pg = svwhilelt_b64(i, length);
        svint64_t vi = svindex_s64(i, 1);
        svint64_t vindex = svmul_n_s64_x(pg, vi, stride);
        svfloat64_t vdata = svld1_gather_s64index_f64(pg, A, vindex);
        svst1_f64(pg, &B[i], vdata);
        i += svcntd();
    }
}

inline void strided_store_double(const double* __restrict__ A,
                                 double* __restrict__ B,
                                 const int64_t length,
                                 const int64_t stride)
{
    int64_t i = 0;
    while (i < length) {
        svbool_t pg = svwhilelt_b64(i, length);
        svfloat64_t vdata = svld1_f64(pg, &A[i]);
        svint64_t vi = svindex_s64(i, 1);
        svint64_t vindex = svmul_n_s64_x(pg, vi, stride);
        svst1_scatter_s64index_f64(pg, B, vindex, vdata);
        i += svcntd();
    }
}
