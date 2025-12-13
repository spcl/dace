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
// Scatter: B[idx[i]] = A[i]
//   - double data
//   - int64 indices
// ------------------------------------------------------------
void scatter_double(const double* __restrict__ A,
                    const int64_t* __restrict__ idx,
                    double* __restrict__ B,
                    const int64_t length) {
#if defined(__AVX512F__)
    // True scatter for AVX-512
if (length >= 8){
        for (int64_t i = 0; i < length; i += 8) {
            int64_t idx_buf[8];
            double  val_buf[8];
            int lane_count = 0;

            for (int lane = 0; lane < 8 && (i + lane) < length; ++lane) {
                idx_buf[lane] = idx[i + lane];
                val_buf[lane] = A[i + lane];
                ++lane_count;
            }

            __m512i vindex = _mm512_loadu_si512((const __m512i*)idx_buf);
            __m512d vdata  = _mm512_loadu_pd(val_buf);

            // Scale = 8 bytes (sizeof(double))
            _mm512_i64scatter_pd(B, vindex, vdata, 8);

            // No need for special tail handling, already done in scalar filling.
        }
} else {
    // Scalar fallback
    for (int64_t i = 0; i < length; ++i) {
        B[idx[i]] = A[i];
    }
}
#elif defined(__AVX2__)
    // AVX2 has no scatter; we just scalar-store per lane.
    // (We still group scalar ops in chunks of 4 for structure.)
if (length >= 4){
    for (int64_t i = 0; i < length; i += 4) {
        for (int lane = 0; lane < 4 && (i + lane) < length; ++lane) {
            int64_t j = idx[i + lane];
            B[j] = A[i + lane];
        }
    } 
} else {
    // Scalar fallback
    for (int64_t i = 0; i < length; ++i) {
        B[idx[i]] = A[i];
    }
}

#elif defined(__ARM_FEATURE_SVE)
    // True scatter using SVE
    int64_t i = 0;
    while (i < length) {
        svbool_t pg = svwhilelt_b64(i, length);

        // Load values A[i..]
        svfloat64_t vdata = svld1_f64(pg, &A[i]);

        // Load indices idx[i..]
        svint64_t vindex = svld1_s64(pg, &idx[i]);

        // Scatter: B[idx[i+lane]] = A[i+lane]
        svst1_scatter_s64index_f64(pg, B, vindex, vdata);

        i += svcntd();
    }

#elif defined(__ARM_NEON)
    // NEON has no scatter; pure scalar.
    for (int64_t i = 0; i < length; ++i) {
        B[idx[i]] = A[i];
    }

#else
    // Scalar fallback
    for (int64_t i = 0; i < length; ++i) {
        B[idx[i]] = A[i];
    }
#endif
}
