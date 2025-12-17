#pragma once

static inline void shift_left(double *__restrict__ dst, const int len, const int shift)
{
    // Forward copy is safe for left shift
    for (int i = 0; i < len - shift; ++i) {
        dst[i] = dst[i + shift];
    }
}


static inline void shift_left_and_assign(double *__restrict__ dst, const double *__restrict__ src, const int len, const int shift)
{
    // Shift left
    for (int i = 0; i < len - shift; ++i) {
        dst[i] = dst[i + shift];
    }

    // Fill newly opened positions
    for (int i = 0; i < shift; ++i) {
        dst[len - shift + i] = src[i];
    }
}


static inline void shift_left_and_assign(double *__restrict__ dst, const double src, const int len, const int shift)
{
    // Shift left
    for (int i = 0; i < len - shift; ++i) {
        dst[i] = dst[i + shift];
    }

    // Fill tail with scalar
    for (int i = 0; i < shift; ++i) {
        dst[len - shift + i] = src;
    }
}
