// Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
#pragma once

#include <cmath>
#include <algorithm>

#define STRINGIZE(x) STRINGIZE_IMPL(x)
#define STRINGIZE_IMPL(x) #x

#if defined(__clang__)
  #define _dace_vectorize(width) _Pragma(STRINGIZE(clang loop vectorize(enable)))
#else
  #define _dace_vectorize(width) _Pragma(STRINGIZE(omp simd))
#endif

// Option F overlay: per-op escape-hatch siblings (vector_<op>_pscalar /
// vector_<op>_av). Arch-independent; one source of truth, included by
// every arch file so the binary always has them regardless of which
// backend the dispatcher selected.
#include "dace/cpu_vectorizable_math_common.h"


template<typename T, int vector_width>
inline void vector_mult(T* __restrict__ c, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        c[i] = a[i] * b[i];
    }
}

template<typename T, int vector_width>
inline void vector_mult_w_scalar(T* __restrict__ b, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        b[i] = a[i] * constant;
    }
}

template<typename T, int vector_width>
inline void vector_add(T* __restrict__ c, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        c[i] = a[i] + b[i];
    }
}

template<typename T, int vector_width>
inline void vector_add_w_scalar(T* __restrict__ b, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        b[i] = a[i] + constant;
    }
}

// Masked add: lanes where mask[i] == false leave out[i] unchanged (RMW).
template<typename T, int vector_width>
inline void vector_add_masked(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
                              const bool* __restrict__ mask) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        if (mask[i]) out[i] = a[i] + b[i];
    }
}

template<typename T, int vector_width>
inline void vector_add_w_scalar_masked(T* __restrict__ out, const T* __restrict__ a, const T constant,
                                       const bool* __restrict__ mask) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        if (mask[i]) out[i] = a[i] + constant;
    }
}

template<typename T, int vector_width>
inline void vector_sub(T* __restrict__ c, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        c[i] = a[i] - b[i];
    }
}

template<typename T, int vector_width>
inline void vector_sub_w_scalar(T* __restrict__ b, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        b[i] = a[i] - constant;
    }
}

template<typename T, int vector_width>
inline void vector_sub_w_scalar_c(T* __restrict__ b, const T constant, const T* __restrict__ a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        b[i] = constant - a[i];
    }
}

template<typename T, int vector_width>
inline void vector_div(T* __restrict__ c, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        c[i] = a[i] / b[i];
    }
}

template<typename T, int vector_width>
inline void vector_div_w_scalar(T* __restrict__ b, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        b[i] = a[i] / constant;
    }
}

template<typename T, int vector_width>
inline void vector_div_w_scalar_c(T* __restrict__ b, const T constant, const T* __restrict__ a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        b[i] = constant / a[i];
    }
}

template<typename T, int vector_width>
inline void vector_copy(T* __restrict__ dst, const T* __restrict__ src) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        dst[i] = src[i];
    }
}

template<typename T, int vector_width>
inline void vector_copy_w_scalar(T* __restrict__ dst, const T a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        dst[i] = a;
    }
}


template<typename T, int vector_width>
inline void vector_exp(T* __restrict__ out, const T* __restrict__ a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = std::exp(a[i]);
    }
}

template<typename T, int vector_width>
inline void vector_log(T* __restrict__ out, const T* __restrict__ a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = std::log(a[i]);
    }
}

template<typename T, int vector_width>
inline void vector_min(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = std::min(a[i], b[i]);
    }
}

template<typename T, int vector_width>
inline void vector_min_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = std::min(a[i], constant);
    }
}

template<typename T, int vector_width>
inline void vector_max(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = std::max(a[i], b[i]);
    }
}

template<typename T, int vector_width>
inline void vector_max_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = std::max(a[i], constant);
    }
}

// ---- Comparison operators ----

template<typename T, int vector_width>
inline void vector_gt(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] > b[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_gt_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] > constant) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_gt_w_scalar_c(T* __restrict__ out, const T constant, const T* __restrict__ a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (constant > a[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_lt(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] < b[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_lt_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] < constant) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_lt_w_scalar_c(T* __restrict__ out, const T constant, const T* __restrict__ a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (constant < a[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_ge(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] >= b[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_ge_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] >= constant) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_ge_w_scalar_c(T* __restrict__ out, const T constant, const T* __restrict__ a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (constant >= a[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_le(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] <= b[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_le_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] <= constant) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_le_w_scalar_c(T* __restrict__ out, const T constant, const T* __restrict__ a) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (constant <= a[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_eq(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] == b[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_eq_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] == constant) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_ne(T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] != b[i]) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width>
inline void vector_ne_w_scalar(T* __restrict__ out, const T* __restrict__ a, const T constant) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = (a[i] != constant) ? 1.0 : 0.0;
    }
}

template<typename T, int vector_width, typename CondT = bool>
inline void vector_select(T* __restrict__ out, const CondT* __restrict__ cond, const T* __restrict__ t,
                          const T* __restrict__ e) {
    _dace_vectorize(vector_width)
    for (int i = 0; i < vector_width; i++) {
        out[i] = cond[i] ? t[i] : e[i];
    }
}
