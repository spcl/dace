#pragma once

#include <stdint.h>
#include <math.h>

#include "dace/arith/exp.h"
#include "dace/arith/log.h"

static inline double dace_pow_d(double x, double y) {
    return dace_exp_d(y * dace_log_d(x));
}

static inline float dace_pow_f(float x, float y) {
    return dace_exp_f(y * dace_log_f(x));
}

static inline double dace_pow_d_int(double x, int y) {
    if (y == 0){
        return 1.0;
    }

    int n = y;
    double result = 1.0;

    // If exponent is negative, invert base and use positive exponent
    if (n < 0) {
        x = 1.0 / x;
        n = -n;
    }

    while (n > 0) {
        if (n & 1)          // If current bit is 1
            result *= x;
        x *= x;             // Square the base
        n >>= 1;            // Shift exponent to process next bit
    }

    return result;
}

static inline float dace_pow_f_int(float x, int y) {
    if (y == 0){
        return 1.0;
    }

    int n = y;
    float result = 1.0;

    // If exponent is negative, invert base and use positive exponent
    if (n < 0) {
        x = 1.0 / x;
        n = -n;
    }

    while (n > 0) {
        if (n & 1)          // If current bit is 1
            result *= x;
        x *= x;             // Square the base
        n >>= 1;            // Shift exponent to process next bit
    }

    return result;
}