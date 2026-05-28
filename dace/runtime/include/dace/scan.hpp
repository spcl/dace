// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// dace/scan.hpp -- header-only parallel prefix-scan routines for the DaCe ``Scan``
// library node's CPU expansion. Uses OpenMP 5.0's ``reduction(inscan, op:var)`` +
// ``#pragma omp scan {inclusive,exclusive}(var)`` -- supported by GCC 10+ and
// Clang 11+. Sequential fallback uses ``std::inclusive_scan`` / ``std::exclusive_scan``.
//
// Why a header rather than inline pragmas in the libnode tasklet: the tasklet code
// is wrapped in a block scope, which makes ``#include <numeric>`` etc. fragile and
// pragmas harder to read. Putting the algorithm in a small set of templated
// functions keeps the tasklet to a single function call.
//
// Supported binary ops: ``+``, ``*``, ``min``, ``max`` (all OpenMP-built-in
// reductions, so no user-defined reduction declaration is needed).
//
// Each routine has an inclusive and an exclusive variant. The exclusive variant
// requires the caller to pass the identity element as ``seed`` -- for sum that's
// ``0``, for product ``1``; for min/max it's whatever the caller deems neutral.

#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>

namespace dace { namespace scan {

// --- INCLUSIVE -----------------------------------------------------------------

template <typename It, typename OutIt>
inline void inclusive_sum(It first, It last, OutIt out_first) {
    using T = typename std::iterator_traits<It>::value_type;
    const long n = static_cast<long>(last - first);
    T acc = T(0);
    #pragma omp parallel for simd reduction(inscan, +:acc)
    for (long i = 0; i < n; ++i) {
        acc = acc + first[i];
        #pragma omp scan inclusive(acc)
        out_first[i] = acc;
    }
}

template <typename It, typename OutIt>
inline void inclusive_product(It first, It last, OutIt out_first) {
    using T = typename std::iterator_traits<It>::value_type;
    const long n = static_cast<long>(last - first);
    T acc = T(1);
    #pragma omp parallel for simd reduction(inscan, *:acc)
    for (long i = 0; i < n; ++i) {
        acc = acc * first[i];
        #pragma omp scan inclusive(acc)
        out_first[i] = acc;
    }
}

template <typename It, typename OutIt>
inline void inclusive_min(It first, It last, OutIt out_first) {
    using T = typename std::iterator_traits<It>::value_type;
    const long n = static_cast<long>(last - first);
    if (n <= 0) return;
    T acc = first[0];
    #pragma omp parallel for simd reduction(inscan, min:acc)
    for (long i = 0; i < n; ++i) {
        acc = std::min<T>(acc, first[i]);
        #pragma omp scan inclusive(acc)
        out_first[i] = acc;
    }
}

template <typename It, typename OutIt>
inline void inclusive_max(It first, It last, OutIt out_first) {
    using T = typename std::iterator_traits<It>::value_type;
    const long n = static_cast<long>(last - first);
    if (n <= 0) return;
    T acc = first[0];
    #pragma omp parallel for simd reduction(inscan, max:acc)
    for (long i = 0; i < n; ++i) {
        acc = std::max<T>(acc, first[i]);
        #pragma omp scan inclusive(acc)
        out_first[i] = acc;
    }
}

// --- EXCLUSIVE -----------------------------------------------------------------
// out[0] = seed; out[i] = seed OP in[0] OP ... OP in[i-1]

template <typename It, typename OutIt, typename T>
inline void exclusive_sum(It first, It last, OutIt out_first, T seed) {
    const long n = static_cast<long>(last - first);
    T acc = seed;
    #pragma omp parallel for simd reduction(inscan, +:acc)
    for (long i = 0; i < n; ++i) {
        out_first[i] = acc;
        #pragma omp scan exclusive(acc)
        acc = acc + first[i];
    }
}

template <typename It, typename OutIt, typename T>
inline void exclusive_product(It first, It last, OutIt out_first, T seed) {
    const long n = static_cast<long>(last - first);
    T acc = seed;
    #pragma omp parallel for simd reduction(inscan, *:acc)
    for (long i = 0; i < n; ++i) {
        out_first[i] = acc;
        #pragma omp scan exclusive(acc)
        acc = acc * first[i];
    }
}

template <typename It, typename OutIt, typename T>
inline void exclusive_min(It first, It last, OutIt out_first, T seed) {
    const long n = static_cast<long>(last - first);
    T acc = seed;
    #pragma omp parallel for simd reduction(inscan, min:acc)
    for (long i = 0; i < n; ++i) {
        out_first[i] = acc;
        #pragma omp scan exclusive(acc)
        acc = std::min<T>(acc, first[i]);
    }
}

template <typename It, typename OutIt, typename T>
inline void exclusive_max(It first, It last, OutIt out_first, T seed) {
    const long n = static_cast<long>(last - first);
    T acc = seed;
    #pragma omp parallel for simd reduction(inscan, max:acc)
    for (long i = 0; i < n; ++i) {
        out_first[i] = acc;
        #pragma omp scan exclusive(acc)
        acc = std::max<T>(acc, first[i]);
    }
}

// --- STRIDED INCLUSIVE -------------------------------------------------------
// ``out[i + s] = out[i] OP in[i]`` for stride s > 0 (caller assertion). Each
// residue class k in [0, s) is an independent inclusive scan over the strided
// slice; the s classes run in parallel, the within-class scan stays sequential.
// For s == 1 these are equivalent to the contiguous routines above; callers
// should dispatch on stride to pick the right form.
//
// Contract: ``out`` and ``in`` are 1-D arrays of length ``n``. For each residue
// class k, the scan accumulator starts at the op's identity (0 / 1 / inputs[0]
// / inputs[0] respectively for sum / product / min / max), so the *first*
// scanned value in each class becomes ``out[k]`` directly. Seeds external to
// the scan (if needed) are folded in by the caller via a separate map.

template <typename It, typename OutIt>
inline void strided_inclusive_sum(It first, OutIt out, long n, long s) {
    using T = typename std::iterator_traits<It>::value_type;
    if (s <= 0) std::abort();
    #pragma omp parallel for
    for (long k = 0; k < s; ++k) {
        T acc = T(0);
        for (long j = k; j < n; j += s) {
            acc = acc + first[j];
            out[j] = acc;
        }
    }
}

template <typename It, typename OutIt>
inline void strided_inclusive_product(It first, OutIt out, long n, long s) {
    using T = typename std::iterator_traits<It>::value_type;
    if (s <= 0) std::abort();
    #pragma omp parallel for
    for (long k = 0; k < s; ++k) {
        T acc = T(1);
        for (long j = k; j < n; j += s) {
            acc = acc * first[j];
            out[j] = acc;
        }
    }
}

template <typename It, typename OutIt>
inline void strided_inclusive_min(It first, OutIt out, long n, long s) {
    using T = typename std::iterator_traits<It>::value_type;
    if (s <= 0) std::abort();
    #pragma omp parallel for
    for (long k = 0; k < s; ++k) {
        if (k >= n) continue;
        T acc = first[k];
        out[k] = acc;
        for (long j = k + s; j < n; j += s) {
            acc = std::min<T>(acc, first[j]);
            out[j] = acc;
        }
    }
}

template <typename It, typename OutIt>
inline void strided_inclusive_max(It first, OutIt out, long n, long s) {
    using T = typename std::iterator_traits<It>::value_type;
    if (s <= 0) std::abort();
    #pragma omp parallel for
    for (long k = 0; k < s; ++k) {
        if (k >= n) continue;
        T acc = first[k];
        out[k] = acc;
        for (long j = k + s; j < n; j += s) {
            acc = std::max<T>(acc, first[j]);
            out[j] = acc;
        }
    }
}

}}  // namespace dace::scan
