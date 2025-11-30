#pragma once

#include <immintrin.h>
#include <stdint.h>

void gather_double_avx512(const double *__restrict__ A,
                          const int64_t *__restrict__ idx,
                          double *__restrict__ B) {
    // Load 8 indices into 512-bit vector
    __m512i vindex = _mm512_loadu_si512((__m512i*)idx);

    // Gather 8 doubles from A using 64-bit indices
    __m512d vdata = _mm512_i64gather_pd(vindex, A, 8);

    // Store result
    _mm512_storeu_pd(B, vdata);
}