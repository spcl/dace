#pragma once

#include <immintrin.h>
#include <stdint.h>

void gather_double(const double *__restrict__ A,
                          const int64_t *__restrict__ idx,
                          double *__restrict__ B,
                        const int64_t length) {
#if defined(__AVX512F__)
    // ---------------------------
    // AVX-512 version (8 doubles)
    // ---------------------------
    for (int64_t i = 0; i < length; i+= 8){
        __m512i vindex = _mm512_loadu_si512((__m512i*)&idx[i]);       // load 8 int64 indices
        __m512d vdata = _mm512_i64gather_pd(vindex, &A[i], 8);        // gather 8 doubles
        _mm512_storeu_pd(&B[i], vdata);                                // store result
    }

#elif defined(__AVX2__)
    // ---------------------------
    // AVX2 version (4 doubles, 32-bit indices)
    // ---------------------------
    // Convert int64_t indices to int32_t if safe
    for (int64_t i = 0; i < length; i+= 4){
        int32_t idx32[4];
        for (int i = 0; i < 4; ++i) {
            idx32[i] = static_cast<int32_t>(idx[i]);
        }

        __m128i vindex = _mm_loadu_si128((__m128i*)idx32);        // load 4 int32 indices
        __m256d vdata = _mm256_i32gather_pd(&A[i], vindex, 4);        // gather 4 doubles
        _mm256_storeu_pd(&B[i], vdata);                                // store result
    }
#else

    for (int i = 0; i < length; ++i) B[i] = A[idx[i]];
#endif
}
