#pragma once

// Based on CERN VDT (VectoriseD maTh) C++ Library for Fast Math
// It is a C-compatible rewrite of it https://github.com/drbenmorgan/vdt

#include <stdint.h>


// ============================================================================
// HELPER FUNCTIONS FOR MANTISSA/EXPONENT EXTRACTION
// ============================================================================

/* Extract mantissa and exponent from double */
static inline double get_mant_exponent_d(double x, double* __restrict__ fe) {
    union {
        double d;
        uint64_t i;
    } u;
    
    u.d = x;
    
    /* Extract exponent */
    int64_t exp = (int64_t)((u.i >> 52) & 0x7ffULL);
    exp -= 1023; /* Remove bias */
    *fe = (double)exp;
    
    /* Set exponent to 0 (bias = 1023) to get mantissa in [0.5, 1) */
    u.i = (u.i & 0x800fffffffffffffULL) | 0x3fe0000000000000ULL;
    
    return u.d;
}

/* Extract mantissa and exponent from float */
static inline float get_mant_exponent_f(float x, float* __restrict__ fe) {
    union {
        float f;
        uint32_t i;
    } u;
    
    u.f = x;
    
    /* Extract exponent */
    int32_t exp = (int32_t)((u.i >> 23) & 0xffU);
    exp -= 127; /* Remove bias */
    *fe = (float)exp;
    
    /* Set exponent to 0 (bias = 127) to get mantissa in [0.5, 1) */
    u.i = (u.i & 0x807fffffU) | 0x3f000000U;
    
    return u.f;
}

// ============================================================================
// DOUBLE PRECISION LOG
// ============================================================================

#define SQRTH 0.70710678118654752440

/* Polynomial P(x) for log */
static inline double get_log_px(double x) {
    const double PX1log = 1.01875663804580931796E-4;
    const double PX2log = 4.97494994976747001425E-1;
    const double PX3log = 4.70579119878881725854E0;
    const double PX4log = 1.44989225341610930846E1;
    const double PX5log = 1.79368678507819816313E1;
    const double PX6log = 7.70838733755885391666E0;
    
    double px = PX1log;
    px *= x;
    px += PX2log;
    px *= x;
    px += PX3log;
    px *= x;
    px += PX4log;
    px *= x;
    px += PX5log;
    px *= x;
    px += PX6log;
    
    return px;
}

/* Polynomial Q(x) for log */
static inline double get_log_qx(double x) {
    const double QX1log = 1.12873587189167450590E1;
    const double QX2log = 4.52279145837532221105E1;
    const double QX3log = 8.29875266912776603211E1;
    const double QX4log = 7.11544750618563894466E1;
    const double QX5log = 2.31251620126765340583E1;
    
    double qx = x;
    qx += QX1log;
    qx *= x;
    qx += QX2log;
    qx *= x;
    qx += QX3log;
    qx *= x;
    qx += QX4log;
    qx *= x;
    qx += QX5log;
    
    return qx;
}

/* Natural logarithm for double precision
   Assumes valid input (x > 0, not NaN, not Inf) */
static inline double dace_log_d(double x) {
    double fe;

    /* Separate mantissa from exponent */
    x = get_mant_exponent_d(x, &fe);
    
    /* Blending */
    if (x > SQRTH) {
        fe += 1.0;
    } else {
        x += x;
    }
    x -= 1.0;
    
    /* Rational form P(x)/Q(x) */
    double px = get_log_px(x);
    
    /* For the final formula */
    const double x2 = x * x;
    px *= x;
    px *= x2;
    
    const double qx = get_log_qx(x);
    
    double res = px / qx;
    
    res -= fe * 2.121944400546905827679e-4;
    res -= 0.5 * x2;
    
    res = x + res;
    res += fe * 0.693359375;
    
    return res;
}

static inline double dace_log_d_safe(double x) {
    double fe;
    const double LOG_UPPER_LIMIT = 1e307;
    const double LOG_LOWER_LIMIT = 0;
    double original_x = x;

    /* Separate mantissa from exponent */
    x = get_mant_exponent_d(x, &fe);
    
    /* Blending */
    if (x > SQRTH) {
        fe += 1.0;
    } else {
        x += x;
    }
    x -= 1.0;
    
    /* Rational form P(x)/Q(x) */
    double px = get_log_px(x);
    
    /* For the final formula */
    const double x2 = x * x;
    px *= x;
    px *= x2;
    
    const double qx = get_log_qx(x);
    
    double res = px / qx;
    
    res -= fe * 2.121944400546905827679e-4;
    res -= 0.5 * x2;
    
    res = x + res;
    res += fe * 0.693359375;

    if (original_x > LOG_UPPER_LIMIT){
        res = INFINITY;
    }
    if (original_x < LOG_LOWER_LIMIT){   /* THIS IS NAN! */
        res = -NAN;
    }

    return res;
}

// ============================================================================
// SINGLE PRECISION LOG
// ============================================================================

#define SQRTHF 0.707106781186547524f

/* Polynomial for logf */
static inline float get_log_poly_f(float x) {
    const float PX1logf = 7.0376836292E-2f;
    const float PX2logf = -1.1514610310E-1f;
    const float PX3logf = 1.1676998740E-1f;
    const float PX4logf = -1.2420140846E-1f;
    const float PX5logf = 1.4249322787E-1f;
    const float PX6logf = -1.6668057665E-1f;
    const float PX7logf = 2.0000714765E-1f;
    const float PX8logf = -2.4999993993E-1f;
    const float PX9logf = 3.3333331174E-1f;
    
    float y = x * PX1logf;
    y += PX2logf;
    y *= x;
    y += PX3logf;
    y *= x;
    y += PX4logf;
    y *= x;
    y += PX5logf;
    y *= x;
    y += PX6logf;
    y *= x;
    y += PX7logf;
    y *= x;
    y += PX8logf;
    y *= x;
    y += PX9logf;
    
    return y;
}

/* Natural logarithm for single precision
   Assumes valid input (x > 0, not NaN, not Inf) */
static inline float dace_log_f(float x) {
    float fe;
    
    x = get_mant_exponent_f(x, &fe);
    
    if (x > SQRTHF) {
        fe += 1.0f;
    } else {
        x += x;
    }
    x -= 1.0f;
    
    const float x2 = x * x;
    
    float res = get_log_poly_f(x);
    res *= x2 * x;

    res += -2.12194440e-4f * fe;
    res += -0.5f * x2;

    res = x + res;

    res += 0.693359375f * fe;

    return res;
}


static inline float dace_log_f_safe(float x) {
    float fe;
    const double original_x = x;

    const double LOG_UPPER_LIMIT = 3.4028234663852885981170418348451692544e38f;
    const double LOG_LOWER_LIMIT = 0;

    x = get_mant_exponent_f(x, &fe);
    
    if (x > SQRTHF) {
        fe += 1.0f;
    } else {
        x += x;
    }
    x -= 1.0f;
    
    const float x2 = x * x;
    
    float res = get_log_poly_f(x);
    res *= x2 * x;
    
    res += -2.12194440e-4f * fe;
    res += -0.5f * x2;
    
    res = x + res;
    
    res += 0.693359375f * fe;

    if (original_x > LOG_UPPER_LIMIT){
        res = INFINITY;
    }
    if (original_x < LOG_LOWER_LIMIT){   /* THIS IS NAN! */
        res = -NAN;
    }

    return res;
}