// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// dace/scan.hpp -- header-only prefix-scan routines for the DaCe ``Scan`` library
// node's CPU expansion, in the PARALLEL schedule.
//
// The scan lowering has exactly two shapes. This header is the first one: a
// hand-blocked three-phase scan over one ``#pragma omp parallel`` region, whose
// per-thread passes are vectorised by OpenMP 5.0's ``simd reduction(inscan,
// op:var)`` + ``#pragma omp scan {inclusive,exclusive}(var)`` (GCC 10+ / Clang 11+
// / ICX 2021+). The second shape is a plain sequential loop with no pragma and no
// call into this header at all; the expansion emits it directly for a sequential
// schedule (see ``ExpandPure`` in ``dace/libraries/standard/nodes/scan.py``).
//
// WHY THE BLOCKING IS HAND-WRITTEN AND NOT ``parallel for simd reduction(inscan)``.
// The composite form looks like the obvious spelling, but GCC lowers it to:
// ``malloc(chunk_bytes)`` per thread, a vectorised local scan from ``in`` INTO
// THAT SCRATCH BUFFER, three ``GOMP_barrier`` calls for the log-P prefix over the
// per-thread totals, a second pass copying scratch + offset into ``out``, then
// ``free``. That is 2R + 2W of DRAM traffic against a sequential scan's 1R + 1W,
// plus a per-call allocation of ``n/P`` elements per thread -- above glibc's
// 128 KB mmap threshold that is an ``mmap``/``munmap`` pair per call, so every
// invocation re-faults its scratch pages. Measured here (fp64 sum, 8 threads,
// GCC 15.2, ratios against a sequential single pass): 0.05x at n=256, 0.62x at
// n=1e6, 0.61x at n=1.6e7 -- i.e. the "parallel" scan lost to the sequential one
// at every size that matters. A ``simd``-only ``inscan`` has none of that: no
// allocation, no barrier, and it writes straight to ``out`` through an in-register
// shift-add network. So this header keeps the ``simd`` part and does the
// cross-thread part itself.
//
// THE SHAPE. One region; the index space is walked in TILES of ``team *
// TILE_ELEMENTS(T)``; within a tile each thread owns one contiguous block, folds
// it read-only, takes its starting offset from the running carry and the totals of
// the blocks before it, then runs a seeded ``simd inscan`` over it writing ``out``.
// Both passes touch the same block, so a tile sized to sit in L2 keeps the second
// read off the memory bus: DRAM traffic is 1R + 1W, the same as a sequential scan,
// at P-way parallelism. Measured against the sequential single pass: 1.7x at
// n=137, 3.8x at n=16384, 6.1x at n=65536, 7.6x at n=262144, 3.4x at n=1e6, 1.3x
// at n=1.6e7 (the last is DRAM-bound and this machine saturates its memory
// controller from one core, so no scan of any shape can gain there).
//
// Supported binary ops: ``+``, ``*``, ``min``, ``max`` (all OpenMP-built-in
// reductions, so no user-defined reduction declaration is needed). Each has an
// inclusive and an exclusive variant, and the inclusive one has a seeded overload
// (``out[k] = seed OP in[0] OP ... OP in[k]``) that ``_scan_init`` maps onto.
//
// TWO PROPERTIES OF THIS LOWERING THAT CALLERS MUST KNOW:
//
// 1. Its floating-point association is block-wise, so it does NOT reproduce the
//    sequential left-to-right order and the result MOVES WITH ``OMP_NUM_THREADS``.
//    Below ``PARALLEL_MIN_ELEMENTS_CONTIGUOUS`` it runs on one thread and does come
//    out stable, so small-n tests will not catch this. A caller that needs a
//    reproducible scan must use the sequential shape, not this header. (Blocking
//    shortens the dependent chains, so it is more accurate, not less: against a
//    long-double reference at n=4e6 it lands 1.1e-13 closer, scale-relative.)
//
// 2. Each unit-stride entry point opens one parallel region. The strided routines
//    below do NOT use ``inscan`` -- an ``inscan`` loop can only span a single
//    residue class, so it would cost one region per class; they open one region
//    for the whole scan and split classes across the team instead.
//
// MULTIPLE INDEPENDENT CHAINS over one index range (``Scan.chains > 1``, the
// cloudsc ``pfsqrf`` / TSVC ``scan_multi_5carry`` shape) are NOT here: OpenMP 5.0
// caps a loop at one ``scan`` directive but not at one list item, so K chains fit
// in a single ``reduction(inscan, op: a0, ..., aK)`` loop -- one fork/join for all
// K. That form cannot be a template, because an ``inscan`` reduction rejects array
// sections (``reduction(inscan, +: acc[0:K])`` is a hard error on GCC 15 / the
// OpenMP 5.0 restriction), so the K accumulator names have to be spelled out.
// ``_multi_chain_tasklet`` in ``dace/libraries/standard/nodes/scan.py`` emits them
// at expansion time, where K is known; it runs the same three-phase blocking and
// borrows ``detail::TeamSlot`` / ``detail::block_span`` from here for it.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>

#include "types.h"

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

namespace detail {

#ifdef _OPENMP
// Complex needs a user-defined reduction (OpenMP has none for a class type), and only
// ``+`` / ``*`` -- see ``is_ordered`` below for why ``min`` / ``max`` are refused instead.
// Declared here (not reused from ``dace::reduce``'s copy in reduction.h) because an OpenMP
// UDR is found by ordinary unqualified lookup from the point of use, and every fold /
// ``inscan`` pass that needs one is a function in THIS namespace.
#pragma omp declare reduction(+ : complex64 : omp_out += omp_in) initializer(omp_priv = complex64(0))
#pragma omp declare reduction(+ : complex128 : omp_out += omp_in) initializer(omp_priv = complex128(0))
#pragma omp declare reduction(* : complex64 : omp_out *= omp_in) initializer(omp_priv = complex64(1))
#pragma omp declare reduction(* : complex128 : omp_out *= omp_in) initializer(omp_priv = complex128(1))
#endif

/// Largest team the blocked scan will ask for. The per-thread block totals live
/// in a stack array of this many cache-line slots (16 KB), so the bound buys an
/// allocation-free publish -- the whole point of not using ``parallel for simd``.
constexpr int MAX_TEAM = 256;

/// One thread's block total, padded to a cache line so publishing it does not
/// invalidate a neighbour's. ``K > 1`` carries one total per scan chain; the
/// multi-chain expansion in ``scan.py`` instantiates it that way.
template <typename T, int K = 1>
struct alignas(64) TeamSlot {
    T v[K];
};

/// Below this element count a scan runs faster on one thread than on a team: the
/// fork plus two barriers costs more than the extra lanes save. Measured (fp64
/// sum, 8 threads, against a sequential single pass) one thread runs 2.5x and a
/// team 2.1x at n=8192; one thread 2.6x and a team 3.8x at n=16384.
constexpr long PARALLEL_MIN_ELEMENTS_CONTIGUOUS = 1L << 14;

/// Bytes per thread per tile. The reduce pass and the scan pass walk the same
/// block back to back, so a tile that fits in L2 keeps the second walk off the
/// memory bus. Measured at n=4e6 and n=1.6e7 the 64 KB - 256 KB range is flat and
/// 1 MB (>= L2) loses 20-30%; 128 KB sits inside the L2 of every target this
/// backend compiles for.
constexpr long TILE_BYTES = 1L << 17;

/// Elements one thread takes per tile. A one-thread team takes the whole range in
/// a single tile: with no block totals to exchange it needs no tiling either, and
/// the scan collapses to one seeded ``simd inscan`` pass.
inline long block_span(long n, long team, long elem_bytes) {
    if (team <= 1) return n;
    const long cap = (elem_bytes > 0 && TILE_BYTES / elem_bytes > 0) ? TILE_BYTES / elem_bytes : 1;
    const long even = (n + team - 1) / team;
    return (cap < even) ? cap : even;
}

/// Team the blocked scan asks for, clamped to the block-total array.
inline int team_size() {
#ifdef _OPENMP
    const int m = omp_get_max_threads();
    return (m < MAX_TEAM) ? m : MAX_TEAM;
#else
    return 1;
#endif
}

/// Size of, and this thread's rank in, the enclosing region. The multi-chain
/// expansion emits calls to these instead of ``omp_get_num_threads`` /
/// ``omp_get_thread_num`` so its tasklet also compiles without OpenMP, where every
/// ``#pragma omp`` in it is ignored and the team is one.
inline long team_count() {
#ifdef _OPENMP
    return static_cast<long>(omp_get_num_threads());
#else
    return 1;
#endif
}

inline long team_rank() {
#ifdef _OPENMP
    return static_cast<long>(omp_get_thread_num());
#else
    return 0;
#endif
}

/// Complex has no ordering: ``min`` / ``max`` are refused by name here rather than
/// through whatever ``std::min``/``numeric_limits`` template noise the missing
/// ``operator<`` would otherwise produce.
template <typename T>
struct is_ordered : std::integral_constant<bool, !std::is_same<T, complex64>::value &&
                                                  !std::is_same<T, complex128>::value> {};

/// The ``min`` / ``max`` neutral elements, matching what OpenMP initialises a
/// ``reduction(min:)`` / ``reduction(max:)`` private copy to.
template <typename T>
constexpr T min_identity() {
    static_assert(is_ordered<T>::value, "dace::scan: min/max needs an ordered element type; "
                  "complex64 / complex128 have none -- only sum and product are defined for them.");
    return std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity()
                                                : std::numeric_limits<T>::max();
}

template <typename T>
constexpr T max_identity() {
    static_assert(is_ordered<T>::value, "dace::scan: min/max needs an ordered element type; "
                  "complex64 / complex128 have none -- only sum and product are defined for them.");
    // ``T(...)``, not a bare ``-infinity()``: unary minus on a low-precision struct converts
    // through ``operator float()`` first, so the ternary's other arm (``lowest()``, type ``T``)
    // would make the two arms ``float`` vs ``T`` and fail to find a common type.
    return std::numeric_limits<T>::has_infinity ? T(-std::numeric_limits<T>::infinity())
                                                : std::numeric_limits<T>::lowest();
}

// --- PER-BLOCK PASSES ----------------------------------------------------------
// One function per op (and per inclusive/exclusive), because an OpenMP reduction
// identifier is not a template parameter -- the op has to be spelled into the
// clause. Each is a single vectorised pass over one thread's block.

template <typename It>
inline typename std::iterator_traits<It>::value_type fold_sum(It f, long lo, long hi) {
    typename std::iterator_traits<It>::value_type s = 0;
    #pragma omp simd reduction(+:s)
    for (long i = lo; i < hi; ++i) s = s + f[i];
    return s;
}

template <typename It>
inline typename std::iterator_traits<It>::value_type fold_product(It f, long lo, long hi) {
    typename std::iterator_traits<It>::value_type s = 1;
    #pragma omp simd reduction(*:s)
    for (long i = lo; i < hi; ++i) s = s * f[i];
    return s;
}

template <typename It>
inline typename std::iterator_traits<It>::value_type fold_min(It f, long lo, long hi) {
    using T = typename std::iterator_traits<It>::value_type;
    T s = min_identity<T>();
    #pragma omp simd reduction(min:s)
    for (long i = lo; i < hi; ++i) s = std::min<T>(s, f[i]);
    return s;
}

template <typename It>
inline typename std::iterator_traits<It>::value_type fold_max(It f, long lo, long hi) {
    using T = typename std::iterator_traits<It>::value_type;
    T s = max_identity<T>();
    #pragma omp simd reduction(max:s)
    for (long i = lo; i < hi; ++i) s = std::max<T>(s, f[i]);
    return s;
}

template <typename It, typename OutIt, typename T>
inline void scan_incl_sum(It f, OutIt o, long lo, long hi, T seed) {
    T acc = seed;
    #pragma omp simd reduction(inscan, +:acc)
    for (long i = lo; i < hi; ++i) {
        acc = acc + f[i];
        #pragma omp scan inclusive(acc)
        o[i] = acc;
    }
}

template <typename It, typename OutIt, typename T>
inline void scan_incl_product(It f, OutIt o, long lo, long hi, T seed) {
    T acc = seed;
    #pragma omp simd reduction(inscan, *:acc)
    for (long i = lo; i < hi; ++i) {
        acc = acc * f[i];
        #pragma omp scan inclusive(acc)
        o[i] = acc;
    }
}

template <typename It, typename OutIt, typename T>
inline void scan_incl_min(It f, OutIt o, long lo, long hi, T seed) {
    T acc = seed;
    #pragma omp simd reduction(inscan, min:acc)
    for (long i = lo; i < hi; ++i) {
        acc = std::min<T>(acc, f[i]);
        #pragma omp scan inclusive(acc)
        o[i] = acc;
    }
}

template <typename It, typename OutIt, typename T>
inline void scan_incl_max(It f, OutIt o, long lo, long hi, T seed) {
    T acc = seed;
    #pragma omp simd reduction(inscan, max:acc)
    for (long i = lo; i < hi; ++i) {
        acc = std::max<T>(acc, f[i]);
        #pragma omp scan inclusive(acc)
        o[i] = acc;
    }
}

template <typename It, typename OutIt, typename T>
inline void scan_excl_sum(It f, OutIt o, long lo, long hi, T seed) {
    T acc = seed;
    #pragma omp simd reduction(inscan, +:acc)
    for (long i = lo; i < hi; ++i) {
        o[i] = acc;
        #pragma omp scan exclusive(acc)
        acc = acc + f[i];
    }
}

template <typename It, typename OutIt, typename T>
inline void scan_excl_product(It f, OutIt o, long lo, long hi, T seed) {
    T acc = seed;
    #pragma omp simd reduction(inscan, *:acc)
    for (long i = lo; i < hi; ++i) {
        o[i] = acc;
        #pragma omp scan exclusive(acc)
        acc = acc * f[i];
    }
}

template <typename It, typename OutIt, typename T>
inline void scan_excl_min(It f, OutIt o, long lo, long hi, T seed) {
    T acc = seed;
    #pragma omp simd reduction(inscan, min:acc)
    for (long i = lo; i < hi; ++i) {
        o[i] = acc;
        #pragma omp scan exclusive(acc)
        acc = std::min<T>(acc, f[i]);
    }
}

template <typename It, typename OutIt, typename T>
inline void scan_excl_max(It f, OutIt o, long lo, long hi, T seed) {
    T acc = seed;
    #pragma omp simd reduction(inscan, max:acc)
    for (long i = lo; i < hi; ++i) {
        o[i] = acc;
        #pragma omp scan exclusive(acc)
        acc = std::max<T>(acc, f[i]);
    }
}

/// Three-phase blocked scan over one parallel region. ``reduce(lo, hi)`` folds a
/// block (the op identity on an empty one), ``scan(lo, hi, off)`` writes
/// ``out[lo:hi]`` seeded with ``off``, and ``combine`` is the same binary op over
/// two folds. ``T`` is the fold type; the tile loop, the barriers and the
/// carry are the only things that are not op-specific, so they live here alone.
template <typename T, typename Reduce, typename Scan, typename Combine>
inline void blocked_scan(long n, long elem_bytes, T seed, Reduce reduce, Scan scan, Combine combine) {
    if (n <= 0) return;
#ifdef _OPENMP
    const int want = team_size();
    // The size test has to sit OUTSIDE the pragma rather than in an ``if`` clause:
    // a serialized region is not free -- an empty ``omp parallel if(0)`` measures
    // 691 ns against the 90 ns a 137-element scan takes, so the cloudsc-sized
    // scans would pay 8x their own cost just to enter and leave a one-thread team.
    if (want > 1 && n >= PARALLEL_MIN_ELEMENTS_CONTIGUOUS) {
        TeamSlot<T> totals[MAX_TEAM];
        #pragma omp parallel num_threads(want)
        {
            const long team = static_cast<long>(omp_get_num_threads());
            const long me = static_cast<long>(omp_get_thread_num());
            const long per = block_span(n, team, elem_bytes);
            const long tile = per * team;
            T carry = seed;
            for (long base = 0; base < n; base += tile) {
                const long end = (base + tile < n) ? base + tile : n;
                const long lo = (base + me * per < end) ? base + me * per : end;
                const long hi = (lo + per < end) ? lo + per : end;
                if (team > 1) {
                    totals[me].v[0] = reduce(lo, hi);
                    #pragma omp barrier
                    T off = carry, all = carry;
                    for (long q = 0; q < team; ++q) {
                        if (q == me) off = all;
                        all = combine(all, totals[q].v[0]);
                    }
                    scan(lo, hi, off);
                    carry = all;
                    #pragma omp barrier
                } else {
                    scan(lo, hi, carry);
                }
            }
        }
        return;
    }
#endif
    scan(0, n, seed);
}

}  // namespace detail

// --- INCLUSIVE -----------------------------------------------------------------
// The seeded overload is ``out[k] = seed OP in[0] OP ... OP in[k]``; the plain one
// seeds with the op's identity, which for min/max is the neutral element rather
// than ``in[0]`` (same result, and it keeps an empty range from reading ``in``).

template <typename It, typename OutIt, typename T>
inline void inclusive_sum(It first, It last, OutIt out_first, T seed) {
    using E = typename std::iterator_traits<It>::value_type;
    detail::blocked_scan<E>(
        static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
        [first](long lo, long hi) { return detail::fold_sum(first, lo, hi); },
        [first, out_first](long lo, long hi, E off) { detail::scan_incl_sum(first, out_first, lo, hi, off); },
        [](E a, E b) { return a + b; });
}

template <typename It, typename OutIt>
inline void inclusive_sum(It first, It last, OutIt out_first) {
    inclusive_sum(first, last, out_first, typename std::iterator_traits<It>::value_type(0));
}

template <typename It, typename OutIt, typename T>
inline void inclusive_product(It first, It last, OutIt out_first, T seed) {
    using E = typename std::iterator_traits<It>::value_type;
    detail::blocked_scan<E>(
        static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
        [first](long lo, long hi) { return detail::fold_product(first, lo, hi); },
        [first, out_first](long lo, long hi, E off) { detail::scan_incl_product(first, out_first, lo, hi, off); },
        [](E a, E b) { return a * b; });
}

template <typename It, typename OutIt>
inline void inclusive_product(It first, It last, OutIt out_first) {
    inclusive_product(first, last, out_first, typename std::iterator_traits<It>::value_type(1));
}

template <typename It, typename OutIt, typename T>
inline void inclusive_min(It first, It last, OutIt out_first, T seed) {
    using E = typename std::iterator_traits<It>::value_type;
    detail::blocked_scan<E>(
        static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
        [first](long lo, long hi) { return detail::fold_min(first, lo, hi); },
        [first, out_first](long lo, long hi, E off) { detail::scan_incl_min(first, out_first, lo, hi, off); },
        [](E a, E b) { return std::min<E>(a, b); });
}

template <typename It, typename OutIt>
inline void inclusive_min(It first, It last, OutIt out_first) {
    inclusive_min(first, last, out_first, detail::min_identity<typename std::iterator_traits<It>::value_type>());
}

template <typename It, typename OutIt, typename T>
inline void inclusive_max(It first, It last, OutIt out_first, T seed) {
    using E = typename std::iterator_traits<It>::value_type;
    detail::blocked_scan<E>(
        static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
        [first](long lo, long hi) { return detail::fold_max(first, lo, hi); },
        [first, out_first](long lo, long hi, E off) { detail::scan_incl_max(first, out_first, lo, hi, off); },
        [](E a, E b) { return std::max<E>(a, b); });
}

template <typename It, typename OutIt>
inline void inclusive_max(It first, It last, OutIt out_first) {
    inclusive_max(first, last, out_first, detail::max_identity<typename std::iterator_traits<It>::value_type>());
}

// --- EXCLUSIVE -----------------------------------------------------------------
// out[0] = seed; out[i] = seed OP in[0] OP ... OP in[i-1]

template <typename It, typename OutIt, typename T>
inline void exclusive_sum(It first, It last, OutIt out_first, T seed) {
    using E = typename std::iterator_traits<It>::value_type;
    detail::blocked_scan<E>(
        static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
        [first](long lo, long hi) { return detail::fold_sum(first, lo, hi); },
        [first, out_first](long lo, long hi, E off) { detail::scan_excl_sum(first, out_first, lo, hi, off); },
        [](E a, E b) { return a + b; });
}

template <typename It, typename OutIt, typename T>
inline void exclusive_product(It first, It last, OutIt out_first, T seed) {
    using E = typename std::iterator_traits<It>::value_type;
    detail::blocked_scan<E>(
        static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
        [first](long lo, long hi) { return detail::fold_product(first, lo, hi); },
        [first, out_first](long lo, long hi, E off) { detail::scan_excl_product(first, out_first, lo, hi, off); },
        [](E a, E b) { return a * b; });
}

template <typename It, typename OutIt, typename T>
inline void exclusive_min(It first, It last, OutIt out_first, T seed) {
    using E = typename std::iterator_traits<It>::value_type;
    detail::blocked_scan<E>(
        static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
        [first](long lo, long hi) { return detail::fold_min(first, lo, hi); },
        [first, out_first](long lo, long hi, E off) { detail::scan_excl_min(first, out_first, lo, hi, off); },
        [](E a, E b) { return std::min<E>(a, b); });
}

template <typename It, typename OutIt, typename T>
inline void exclusive_max(It first, It last, OutIt out_first, T seed) {
    using E = typename std::iterator_traits<It>::value_type;
    detail::blocked_scan<E>(
        static_cast<long>(last - first), static_cast<long>(sizeof(E)), static_cast<E>(seed),
        [first](long lo, long hi) { return detail::fold_max(first, lo, hi); },
        [first, out_first](long lo, long hi, E off) { detail::scan_excl_max(first, out_first, lo, hi, off); },
        [](E a, E b) { return std::max<E>(a, b); });
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
