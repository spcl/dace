#pragma once


#if defined(__AVX512F__) || defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__ARM_FEATURE_SVE)
  #include <arm_sve.h>
#elif defined(__ARM_NEON)
  #include <arm_neon.h>
#endif

#include <stdint.h>

void gather_double(const double *__restrict__ A,
                   const int64_t *__restrict__ idx,
                   double *__restrict__ B,
                   const int64_t length) {
#if defined(__AVX512F__)
    // ---------------------------
    // AVX-512 version (8 doubles)
    // ---------------------------
if (length >= 8){
    for (int64_t i = 0; i < length; i+= 8){
        __m512i vindex = _mm512_loadu_si512((__m512i*)&idx[i]);       // load 8 int64 indices
        __m512d vdata = _mm512_i64gather_pd(vindex, &A[i], 8);        // gather 8 doubles
        _mm512_storeu_pd(&B[i], vdata);                                // store result
    }
} else {
    for (int64_t i = 0; i < length; ++i) {
        B[i] = A[idx[i]];
    }
}

#elif defined(__AVX2__)
    // ---------------------------
    // AVX2 version (4 doubles, 32-bit indices)
    // ---------------------------
    // Convert int64_t indices to int32_t if safe
if (length >= 4){
    for (int64_t i = 0; i < length; i+= 4){
        int32_t idx32[4];
        for (int i = 0; i < 4; ++i) {
            idx32[i] = static_cast<int32_t>(idx[i]);
        }

        __m128i vindex = _mm_loadu_si128((__m128i*)idx32);        // load 4 int32 indices
        __m256d vdata = _mm256_i32gather_pd(&A[i], vindex, 4);        // gather 4 doubles
        _mm256_storeu_pd(&B[i], vdata);                                // store result
    }
} else {
    for (int64_t i = 0; i < length; ++i) {
        B[i] = A[idx[i]];
    }
}
#elif defined(__ARM_FEATURE_SVE)
    // ---------------------------
    // ARM SVE version (true gather)
    // ---------------------------
    int64_t i = 0;
    while (i < length) {
        // Predicate for active lanes (b64 because we use int64/double)
        svbool_t pg = svwhilelt_b64(i, length);

        // Load indices idx[i .. i+vl-1]
        svint64_t vindex = svld1_s64(pg, &idx[i]);

        // Gather: interprets vindex as element indices into A
        svfloat64_t vdata = svld1_gather_s64index_f64(pg, A, vindex);

        // Store back to B[i ..]
        svst1_f64(pg, &B[i], vdata);

        i += svcntd(); // advance by vector-length in doubles
    }

#elif defined(__ARM_NEON)
    // ---------------------------
    // ARM NEON version (emulated gather)
    // NEON has no general indexed gather; we do scalar loads but
    // pack them into vectors in chunks of 2 doubles.
    // ---------------------------
    int64_t i = 0;

    // Process pairs with NEON
    for (; i + 1 < length; i += 2) {
        int64_t i0 = idx[i];
        int64_t i1 = idx[i + 1];

        // Load the two scattered elements
        float64x2_t vdata = { A[i0], A[i1] };
        vst1q_f64(&B[i], vdata);
    }

    // Handle tail element (if length is odd)
    for (; i < length; ++i) {
        B[i] = A[idx[i]];
    }

#else
    // ---------------------------
    // Scalar fallback
    // ---------------------------
    for (int64_t i = 0; i < length; ++i) {
        B[i] = A[idx[i]];
    }
#endif
}
