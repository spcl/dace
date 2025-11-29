// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

// Based on CERN VDT (VectoriseD maTh) C++ Library for Fast Math
// It is a C-compatible rewrite of it https://github.com/drbenmorgan/vdt

#include <stdint.h>
#include <math.h>

/* Constants for double precision */
static const double EXP_LIMIT = 708.0;
static const double PX1exp = 1.26177193074810590878E-4;
static const double PX2exp = 3.02994407707441961300E-2;
static const double PX3exp = 9.99999999999999999910E-1;
static const double QX1exp = 3.00198505138664455042E-6;
static const double QX2exp = 2.52448340349684104192E-3;
static const double QX3exp = 2.27265548208155028766E-1;
static const double QX4exp = 2.00000000000000000009E0;
static const double LOG2E = 1.4426950408889634073599;

/* Constants for single precision */
static const float MAXLOGF = 88.72283905206835f;
static const float MINLOGF = -88.0f;
static const float C1F = 0.693359375f;
static const float C2F = -2.12194440e-4f;
static const float PX1expf = 1.9875691500E-4f;
static const float PX2expf = 1.3981999507E-3f;
static const float PX3expf = 8.3334519073E-3f;
static const float PX4expf = 4.1665795894E-2f;
static const float PX5expf = 1.6666665459E-1f;
static const float PX6expf = 5.0000001201E-1f;
static const float LOG2EF = 1.44269504088896341f;

/* Forward declarations for detail functions to be implemented later */
static inline double fpfloor_f64(double x);
static inline float fpfloor_f32(float x);
static inline double uint64_to_double(uint64_t x);
static inline float uint32_to_float(uint32_t x);

/**
 * Fast exponential function for double precision
 * Does not check for NaN inputs
 */
static inline double dace_exp_d(double initial_x) {
    double x = initial_x;
    double px = fpfloor_f64(LOG2E * x + 0.5);

    const int32_t n = (int32_t)px;
    x -= px * 6.93145751953125E-1;
    x -= px * 1.42860682030941723212E-6;

    const double xx = x * x;

    /* px = x * P(x**2) */
    px = PX1exp;
    px *= xx;
    px += PX2exp;
    px *= xx;
    px += PX3exp;
    px *= x;
    /* Evaluate Q(x**2) */
    double qx = QX1exp;
    qx *= xx;
    qx += QX2exp;
    qx *= xx;
    qx += QX3exp;
    qx *= xx;
    qx += QX4exp;

    /* e**x = 1 + 2x P(x**2)/(Q(x**2) - P(x**2)) */
    x = px / (qx - px);
    x = 1.0 + 2.0 * x;

    /* Build 2^n in double */
    x *= uint64_to_double((((uint64_t)n) + 1023) << 52);

    return x;
}

/**
 * Safe fast exponential function for double precision
 * Checks for NaN inputs and returns NaN if input is NaN
 */
static inline double dace_exp_d_safe(double initial_x) {
    if (isnan(initial_x)){
        return NAN;
    }
    double x = dace_exp_d(initial_x);
    if (initial_x > EXP_LIMIT){
        x = INFINITY;
    }
    if (initial_x < -EXP_LIMIT){
        x = 0.0;
    }
}

/**
 * Fast exponential function for single precision
 * Does not check for NaN inputs
 */
static inline float dace_exp_f(float initial_x) {
    float x = initial_x;
    float z = fpfloor_f32(LOG2EF * x + 0.5f);

    x -= z * C1F;
    x -= z * C2F;
    const int32_t n = (int32_t)z;

    const float x2 = x * x;

    z = x * PX1expf;
    z += PX2expf;
    z *= x;
    z += PX3expf;
    z *= x;
    z += PX4expf;
    z *= x;
    z += PX5expf;
    z *= x;
    z += PX6expf;
    z *= x2;
    z += x + 1.0f;

    /* multiply by power of 2 */
    z *= uint32_to_float((n + 0x7f) << 23);


    return z;
}

/**
 * Safe fast exponential function for single precision
 * Checks for NaN inputs and returns NaN if input is NaN
 */
static inline float dace_exp_f_safe(float initial_x) {
    if (isnan(initial_x)){
        return NAN;
    }
    double x = dace_exp_f(initial_x);

    if (initial_x > MAXLOGF){
        x = INFINITY;
    }
    if (initial_x < MINLOGF) {
        x = 0.0f;
    }
}


/* ============================================================================
 * IEEE754 union for type punning
 * ============================================================================ */

union ieee754 {
    double d;
    float f[2];
    uint32_t i[2];
    uint64_t ll;
    uint16_t s[4];
};

/* ============================================================================
 * Detail functions (conversion and floor operations)
 * ============================================================================ */

/**
 * Converts an unsigned 64-bit integer to double
 */
static inline double uint64_to_double(uint64_t ll) {
    union ieee754 tmp;
    tmp.ll = ll;
    return tmp.d;
}

/**
 * Converts a double to unsigned 64-bit integer
 */
static inline uint64_t double_to_uint64(double x) {
    union ieee754 tmp;
    tmp.d = x;
    return tmp.ll;
}

/**
 * Converts an unsigned 32-bit integer to float
 */
static inline float uint32_to_float(uint32_t x) {
    union ieee754 tmp;
    tmp.i[0] = x;
    return tmp.f[0];
}

/**
 * Converts a float to unsigned 32-bit integer
 */
static inline uint32_t float_to_uint32(float x) {
    union ieee754 tmp;
    tmp.f[0] = x;
    return tmp.i[0];
}

/**
 * Vectorizable floor implementation for double precision
 * Does not distinguish between -0.0 and 0.0 (not IEC6509 compliant for -0.0)
 */
static inline double fpfloor_f64(const double x) {
    int32_t ret = (int32_t)(x);
    ret -= (float_to_uint32((float)x) >> 31);
    return (double)ret;
}

/**
 * Vectorizable floor implementation for single precision
 * Does not distinguish between -0.0 and 0.0 (not IEC6509 compliant for -0.0)
 */
static inline float fpfloor_f32(const float x) {
    int32_t ret = (int32_t)(x);
    ret -= (float_to_uint32(x) >> 31);
    return (float)ret;
}
