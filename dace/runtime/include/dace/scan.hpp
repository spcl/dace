// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// dace/scan.hpp -- header-only prefix-scan routines for the DaCe ``Scan`` library
// node's CPU expansion, in the PARALLEL schedule.
//
// The scan lowering has exactly two shapes. This header is the first one: OpenMP
// 5.0's ``reduction(inscan, op:var)`` + ``#pragma omp scan {inclusive,exclusive}
// (var)`` (GCC 10+ / Clang 11+ / ICX 2021+), which the compiler lowers to a
// two-level chunked scan -- per-thread sequential scans, a small prefix over the
// chunk totals, then a parallel offset pass. The second shape is a plain
// sequential loop with no pragma and no call into this header at all; the
// expansion emits it directly for a sequential schedule (see ``ExpandPure`` in
// ``dace/libraries/standard/nodes/scan.py``).
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
//
// TWO PROPERTIES OF THE inscan LOWERING THAT CALLERS MUST KNOW:
//
// 1. Its floating-point association is chunk-wise and implementation-defined, so
//    it does NOT reproduce the sequential left-to-right order, and the result
//    MOVES WITH ``OMP_NUM_THREADS``. Measured on this tree at n=31998, stride 2:
//    a distinct bit pattern at each of 1 / 2 / 4 / 8 / 16 threads, differing from
//    the sequential fold on 74-99% of elements (max |diff| ~3e-13). Short loops
//    (a few dozen elements) are below the chunking threshold and do come out
//    bit-exact, so small-n tests will not catch this. A caller that needs a
//    reproducible scan must use the sequential shape, not this header.
//
// 2. Each unit-stride entry point opens one parallel region. The strided
//    routines below do NOT use ``inscan`` -- an ``inscan`` loop can only span a
//    single residue class, so it would cost one region per class; they open one
//    region for the whole scan and split classes across the team instead.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

// NON-NESTED CONTEXT IS ASSUMED, and there is no runtime check for it: the
// expansion only emits these where the scan is not already inside a parallel
// scope (a scan in a parallel scope lowers to the sequential naked-loop shape
// instead), so the schedule decides statically. An external caller that invokes
// one from inside its own ``omp parallel`` region still gets correct values --
// OpenMP's nested default hands the inner region a one-thread team -- just no
// speedup.

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
// slice. For s == 1 these are equivalent to the contiguous routines above;
// callers should dispatch on stride to pick the right form.
//
// Contract: ``out`` and ``in`` are 1-D arrays of length ``n``. For each residue
// class k, the scan accumulator starts at the op's identity (0 / 1 / inputs[0]
// / inputs[0] respectively for sum / product / min / max), so the *first*
// scanned value in each class becomes ``out[k]`` directly. Seeds external to
// the scan (if needed) are folded in by the caller via a separate map.
//
// SHAPE: ONE parallel region for the whole scan, never one per residue class.
// The classes ARE the parallelism here -- they are independent scans -- so the
// team splits classes and each thread walks its own block. An ``inscan`` loop
// can only span a single class, so using it would open ``s`` regions and ``s``
// strided sweeps; measured at n=31998 / 4 threads that cost 0.75 ms at s=256
// against 0.005 ms for this form (and 11x the regions). The unit-stride
// routines above keep ``inscan`` because there the whole scan IS one class.
//
// TRAVERSAL: row-major, one pass over the array. Walking class-by-class instead
// re-reads every cache line once per class and degrades linearly in s (measured
// 114 ms at s=256 / n=4e6 vs 3.4 ms here). Small strides hold the accumulators
// in registers via a compile-time unroll; wider strides carry through ``out``
// itself, s elements back, which needs no scratch and no allocation.
//
// Thread blocks are cache-line aligned and only used when each thread gets a
// page of classes and the array is large enough to amortize the fork -- a narrow
// block false-shares the lines it straddles and turns one sequential walk into
// one strided walk per thread.
//
// FP: threads split whole classes, never a class, so every element stays the
// left-to-right fold of its own class -- results do not move with the thread
// count. (The unit-stride ``inscan`` routines above do move; see their note.)

namespace detail {

/// Row-major sweep carrying ``S`` accumulators, ``S`` a compile-time constant so
/// the inner loop unrolls into registers.
template <long S, typename It, typename OutIt, typename Seed, typename Op>
inline void strided_scan_unrolled(It first, OutIt out, long n, Seed seed, Op op) {
    using T = typename std::iterator_traits<It>::value_type;
    T acc[S] = {};
    const long head = (S < n) ? S : n;
    for (long k = 0; k < head; ++k) {
        acc[k] = seed(first[k]);
        out[k] = acc[k];
    }
    long j = S;
    for (; j + S <= n; j += S) {
        for (long k = 0; k < S; ++k) {
            acc[k] = op(acc[k], first[j + k]);
            out[j + k] = acc[k];
        }
    }
    for (long k = 0; j + k < n; ++k) {
        acc[k] = op(acc[k], first[j + k]);
        out[j + k] = acc[k];
    }
}

/// Wide strides: the carry lives in ``out`` itself, s elements back. Restricted
/// to residue classes ``[k0, k1)`` so a thread can own a slice of the classes.
template <typename It, typename OutIt, typename Seed, typename Op>
inline void strided_scan_block(It first, OutIt out, long n, long s, long k0, long k1, Seed seed, Op op) {
    if (k0 >= k1) return;
    const long head = (k1 < n) ? k1 : n;
    for (long j = k0; j < head; ++j) out[j] = seed(first[j]);
    for (long base = s; base < n; base += s) {
        const long hi = (base + k1 < n) ? base + k1 : n;
        for (long j = base + k0; j < hi; ++j) out[j] = op(out[j - s], first[j]);
    }
}

/// Residue classes per cache line: a thread block must be a multiple of this or
/// two threads write the same line on every row.
template <typename T>
constexpr long classes_per_line() {
    return (64 / static_cast<long>(sizeof(T))) > 1 ? (64 / static_cast<long>(sizeof(T))) : 1;
}

/// Below this element count the fork costs more than the team saves.
constexpr long PARALLEL_MIN_ELEMENTS = 1L << 16;

/// Minimum bytes of contiguous classes per thread. Measured at n=4e6: 128 B per
/// thread ran 0.33x-0.7x against serial, 16 KB per thread ran 1.1x-1.6x.
constexpr long PARALLEL_MIN_BLOCK_BYTES = 4096;

/// ``seed`` opens a residue class (identity OP first element); ``op`` extends it.
template <typename It, typename OutIt, typename Seed, typename Op>
inline void strided_scan(It first, OutIt out, long n, long s, Seed seed, Op op) {
    if (s <= 0) std::abort();
#ifdef _OPENMP
    using T = typename std::iterator_traits<It>::value_type;
    constexpr long line = classes_per_line<T>();
    constexpr long min_block = PARALLEL_MIN_BLOCK_BYTES / static_cast<long>(sizeof(T));
    const long threads = static_cast<long>(omp_get_max_threads());
    if (threads > 1 && n >= PARALLEL_MIN_ELEMENTS && s >= min_block * threads) {
        #pragma omp parallel num_threads(static_cast<int>(threads))
        {
            const long team = static_cast<long>(omp_get_num_threads());
            const long mine = static_cast<long>(omp_get_thread_num());
            const long lines = (s + line - 1) / line;
            const long per = ((lines + team - 1) / team) * line;
            const long k0 = (mine * per < s) ? mine * per : s;
            const long k1 = (k0 + per < s) ? k0 + per : s;
            strided_scan_block(first, out, n, s, k0, k1, seed, op);
        }
        return;
    }
#endif
    switch (s) {
        case 1: strided_scan_unrolled<1>(first, out, n, seed, op); return;
        case 2: strided_scan_unrolled<2>(first, out, n, seed, op); return;
        case 3: strided_scan_unrolled<3>(first, out, n, seed, op); return;
        case 4: strided_scan_unrolled<4>(first, out, n, seed, op); return;
        case 5: strided_scan_unrolled<5>(first, out, n, seed, op); return;
        case 6: strided_scan_unrolled<6>(first, out, n, seed, op); return;
        case 7: strided_scan_unrolled<7>(first, out, n, seed, op); return;
        case 8: strided_scan_unrolled<8>(first, out, n, seed, op); return;
        default: strided_scan_block(first, out, n, s, 0, s, seed, op); return;
    }
}

}  // namespace detail

template <typename It, typename OutIt>
inline void strided_inclusive_sum(It first, OutIt out, long n, long s) {
    using T = typename std::iterator_traits<It>::value_type;
    detail::strided_scan(first, out, n, s, [](const T& x) { return T(0) + x; },
                         [](const T& a, const T& b) { return a + b; });
}

template <typename It, typename OutIt>
inline void strided_inclusive_product(It first, OutIt out, long n, long s) {
    using T = typename std::iterator_traits<It>::value_type;
    detail::strided_scan(first, out, n, s, [](const T& x) { return T(1) * x; },
                         [](const T& a, const T& b) { return a * b; });
}

template <typename It, typename OutIt>
inline void strided_inclusive_min(It first, OutIt out, long n, long s) {
    using T = typename std::iterator_traits<It>::value_type;
    detail::strided_scan(first, out, n, s, [](const T& x) { return x; },
                         [](const T& a, const T& b) { return std::min<T>(a, b); });
}

template <typename It, typename OutIt>
inline void strided_inclusive_max(It first, OutIt out, long n, long s) {
    using T = typename std::iterator_traits<It>::value_type;
    detail::strided_scan(first, out, n, s, [](const T& x) { return x; },
                         [](const T& a, const T& b) { return std::max<T>(a, b); });
}

}}  // namespace dace::scan
