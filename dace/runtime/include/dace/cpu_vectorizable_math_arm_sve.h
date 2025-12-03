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

#if defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
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
#if defined(__ARM_FEATURE_SVE2)

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

#else  // __ARM_FEATURE_SVE2

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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

#else  // __ARM_FEATURE_SVE2

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
#if defined(__ARM_FEATURE_SVE2)

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

// vector_sub
template<typename T, int vector_width>
inline void vector_sub(T* __restrict__ out,
                       const T* __restrict__ a,
                       const T* __restrict__ b)
{
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
#if defined(__ARM_FEATURE_SVE2)

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
