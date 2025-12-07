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
// Strided load: B[i] = A[i * stride]
// ------------------------------------------------------------
void strided_load_double(const double* __restrict__ A,
                         double* __restrict__ B,
                         const int64_t length,
                         const int64_t stride) {
#if defined(__AVX512F__)
    for (int64_t i = 0; i < length; i += 8) {
        int64_t idx_buf[8];
        int lane_count = 0;
        for (int lane = 0; lane < 8 && (i + lane) < length; ++lane) {
            idx_buf[lane] = (i + lane) * stride;
            ++lane_count;
        }

        __m512i vindex = _mm512_loadu_si512((const __m512i*)idx_buf);
        __m512d vdata  = _mm512_i64gather_pd(vindex, A, 8);

        double tmp[8];
        _mm512_storeu_pd(tmp, vdata);
        for (int lane = 0; lane < lane_count; ++lane) {
            B[i + lane] = tmp[lane];
        }
    }

#elif defined(__AVX2__)
    for (int64_t i = 0; i < length; i += 4) {
        int64_t idx_buf[4];
        int lane_count = 0;
        for (int lane = 0; lane < 4 && (i + lane) < length; ++lane) {
            idx_buf[lane] = (i + lane) * stride;
            ++lane_count;
        }

        __m256d vdata;
        if (lane_count == 4) {
            __m256i vindex = _mm256_loadu_si256((const __m256i*)idx_buf);
            vdata = _mm256_i64gather_pd(A, vindex, 8);
        } else {
            // Tail: just scalar load into a temp buffer
            double tmp[4];
            for (int lane = 0; lane < lane_count; ++lane) {
                tmp[lane] = A[idx_buf[lane]];
            }
            vdata = _mm256_loadu_pd(tmp);
        }

        double tmp[4];
        _mm256_storeu_pd(tmp, vdata);
        for (int lane = 0; lane < lane_count; ++lane) {
            B[i + lane] = tmp[lane];
        }
    }

#elif defined(__ARM_FEATURE_SVE)
    int64_t i = 0;
    while (i < length) {
        svbool_t pg = svwhilelt_b64(i, length);

        // vi = [i, i+1, ..., i+vl-1]
        svint64_t vi = svindex_s64(i, 1);
        // vindex = vi * stride
        svint64_t vindex = svmul_n_s64_x(pg, vi, stride);

        svfloat64_t vdata = svld1_gather_s64index_f64(pg, A, vindex);
        svst1_f64(pg, &B[i], vdata);

        i += svcntd();
    }

#elif defined(__ARM_NEON)
    // NEON: scalar strided load, optionally pack into vectors
    int64_t i = 0;
    for (; i + 1 < length; i += 2) {
        float64x2_t vdata = {
            A[(i + 0) * stride],
            A[(i + 1) * stride]
        };
        vst1q_f64(&B[i], vdata);
    }
    for (; i < length; ++i) {
        B[i] = A[i * stride];
    }

#else
    // Scalar fallback
    for (int64_t i = 0; i < length; ++i) {
        B[i] = A[i * stride];
    }
#endif
}
