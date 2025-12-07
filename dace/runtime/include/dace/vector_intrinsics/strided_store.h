#pragma once

#include <stdint.h>

#if defined(__AVX512F__) || defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__ARM_FEATURE_SVE)
  #include <arm_sve.h>
#elif defined(__ARM_NEON)
  #include <arm_neon.h>
#endif


// ------------------------------------------------------------
// Strided store: B[i * stride] = A[i]
// ------------------------------------------------------------
void strided_store_double(const double* __restrict__ A,
                          double* __restrict__ B,
                          const int64_t length,
                          const int64_t stride) {
#if defined(__AVX512F__)
    for (int64_t i = 0; i < length; i += 8) {
        int64_t idx_buf[8];
        double  val_buf[8];
        int lane_count = 0;
        for (int lane = 0; lane < 8 && (i + lane) < length; ++lane) {
            idx_buf[lane] = (i + lane) * stride;
            val_buf[lane] = A[i + lane];
            ++lane_count;
        }

        __m512i vindex = _mm512_loadu_si512((const __m512i*)idx_buf);
        __m512d vdata  = _mm512_loadu_pd(val_buf);
        _mm512_i64scatter_pd(B, vindex, vdata, 8);
    }

#elif defined(__AVX2__)
    // No scatter; scalar stores
    for (int64_t i = 0; i < length; i += 4) {
        for (int lane = 0; lane < 4 && (i + lane) < length; ++lane) {
            B[(i + lane) * stride] = A[i + lane];
        }
    }

#elif defined(__ARM_FEATURE_SVE)
    int64_t i = 0;
    while (i < length) {
        svbool_t pg = svwhilelt_b64(i, length);

        svfloat64_t vdata = svld1_f64(pg, &A[i]);

        svint64_t vi     = svindex_s64(i, 1);
        svint64_t vindex = svmul_n_s64_x(pg, vi, stride);

        svst1_scatter_s64index_f64(pg, B, vindex, vdata);

        i += svcntd();
    }

#elif defined(__ARM_NEON)
    // Scalar strided stores
    int64_t i = 0;
    for (; i + 1 < length; i += 2) {
        float64x2_t vdata = vld1q_f64(&A[i]);
        double tmp[2];
        vst1q_f64(tmp, vdata);
        B[(i + 0) * stride] = tmp[0];
        B[(i + 1) * stride] = tmp[1];
    }
    for (; i < length; ++i) {
        B[i * stride] = A[i];
    }

#else
    // Scalar fallback
    for (int64_t i = 0; i < length; ++i) {
        B[i * stride] = A[i];
    }
#endif
}
