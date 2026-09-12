// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_DETECT_H
#define __DACE_DETECT_H

// Runtime detection primitives that library-node expansions call instead of emitting
// their own loop: duplicate detection for a scatter index, and a short-circuiting
// find-first over a predicate. Both are OpenMP-parallel and vectorized; keeping them
// here rather than in generated text means one implementation to tune and one place
// where the pragmas are reviewed.

#include <cmath>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "types.h"

namespace dace {

//: Elements one SIMD block of find_first_index covers. The block is the early-exit
//: granularity: a vectorized loop cannot break, so the scan vectorizes WITHIN a block
//: and tests between blocks. Wide enough that the reduction epilogue is amortized,
//: short enough that the overshoot past a hit stays negligible.
static constexpr long long FIND_FIRST_SIMD_BLOCK = 64;

//: Chunk size of find_first_index, as a multiple of sqrt(span). The chunk is the parallel and
//: the cancellation granularity, and the two costs it balances pull opposite ways: a chunk that
//: is too big scans past the answer on one thread, one that is too small pays dispatch on every
//: chunk the answer makes dead. Both are linear in the wrong direction, so the optimum grows as
//: sqrt(span) -- and it measures that way: the best fixed chunk count on this machine was 64 at
//: 1e5 elements, 256 at 1e6 and 1024 at 1e7, i.e. a chunk of 1.6k / 3.9k / 9.8k against the
//: 2.5k / 8k / 25k this rule picks.
static constexpr double FIND_FIRST_CHUNK_SCALE = 8.0;

//: Floor on how many chunks each thread gets, which only binds below ~64k elements -- there
//: sqrt(span) alone would hand a whole thread's share to one chunk and leave the rest idle.
static constexpr long long FIND_FIRST_CHUNKS_PER_THREAD = 4;

/**
 * Chunk size :cpp:func:`find_first_index` splits a ``span``-element range into.
 *
 * See :cvar:`FIND_FIRST_CHUNK_SCALE` for why it grows as the square root of the span, and
 * :cvar:`FIND_FIRST_CHUNKS_PER_THREAD` for the small-span floor.
 */
inline long long find_first_chunk(long long span, bool parallel) {
    long long chunk = (long long)(FIND_FIRST_CHUNK_SCALE * std::sqrt((double)span));
    long long threads = 1;
#ifdef _OPENMP
    if (parallel) threads = (long long)omp_get_max_threads();
#endif
    long long ceiling = span / (FIND_FIRST_CHUNKS_PER_THREAD * threads);
    if (ceiling < 1) ceiling = 1;
    if (chunk > ceiling) chunk = ceiling;
    if (chunk < 1) chunk = 1;
    return chunk;
}

/**
 * Duplicate detection over a scatter index: tagged-write + verify, O(n), no sort.
 *
 * Pass 1 writes ``owner[idx[i]] = i``; pass 2 OR-reduces ``owner[idx[i]] != i``. If two
 * i collide on a slot only one wins pass 1, so the loser reads back a different i and the
 * OR trips. Pass 2 reads only slots pass 1 wrote, so ``owner`` needs no initialization and
 * no clearing between calls -- a caller may hand in one persistent buffer. The
 * last-writer-wins race in pass 1 is benign: any winner is fine and the OR is monotonic.
 *
 * Two passes is the floor. Verify can only read a slot once every writer has had its turn,
 * so folding it into pass 1 would compare against a half-written tag array; a single-pass
 * form needs an atomic read-modify-write per element plus a pre-cleared or epoch-stamped
 * tag array, which is an extra sweep over a domain at least as wide as ``idx``.
 *
 * Values outside ``[0, capacity)`` are skipped in both passes. Such a value is never a slot
 * the guarded scatter writes, so it can neither collide with nor mask a real conflict, and
 * skipping keeps the tag write in bounds when ``idx`` carries entries the scatter does not
 * use (a strided scatter reads only part of ``idx``).
 *
 * @param idx       The scatter index.
 * @param n         Elements of ``idx`` to check.
 * @param owner     Tag buffer spanning the scattered array's domain, any integer type wide enough for
 *                  ``n``; needs no initialization.
 * @param capacity  Elements of ``owner``.
 * @param parallel  Run both passes under OpenMP.
 * @return 1 if any duplicate was found, 0 otherwise.
 */
template <typename T, typename TagT>
inline long long detect_collision(const T *idx, long long n, TagT *owner, long long capacity, bool parallel = true) {
#pragma omp parallel for if (parallel : parallel)
    for (long long i = 0; i < n; ++i) {
        const long long v = static_cast<long long>(idx[i]);
        if (v >= 0 && v < capacity) owner[v] = static_cast<TagT>(i);
    }
    long long c = 0;
    // Bitwise-or is simd-safe (see reduction.h), so the verify pass carries simd as well.
#pragma omp parallel for simd if (parallel : parallel) reduction(| : c)
    for (long long i = 0; i < n; ++i) {
        const long long v = static_cast<long long>(idx[i]);
        if (v >= 0 && v < capacity) c |= (static_cast<long long>(owner[v]) != i) ? 1LL : 0LL;
    }
    return c;
}

/**
 * :func:`detect_collision` sizing its own tag buffer from ``max(idx)``.
 *
 * Costs one extra reduction pass over ``idx`` plus an allocation, so prefer the overload
 * taking a caller-owned buffer wherever the scattered array's domain is known.
 */
template <typename T>
inline long long detect_collision(const T *idx, long long n, bool parallel = true) {
    long long mx = 0;
#pragma omp parallel for simd if (parallel : parallel) reduction(max : mx)
    for (long long i = 0; i < n; ++i) {
        const long long v = static_cast<long long>(idx[i]);
        mx = v > mx ? v : mx;
    }
    std::unique_ptr<long long[]> owner(new long long[static_cast<size_t>(mx) + 1]);
    return detect_collision(idx, n, owner.get(), mx + 1, parallel);
}

/**
 * Whether every element of ``a`` is strictly positive: 1 if all are, 0 if any is not.
 *
 * A per-element 0/1 flag folded by a ``min`` reduction, so the sweep is both parallel and
 * vectorized -- an early ``abort()`` inside the loop would force it serial for no gain, since a
 * precondition check that trips is a program bug and never the hot path. The caller decides what
 * to do with a 0; nothing here traps.
 */
template <typename T>
inline long long detect_all_positive(const T *a, long long n, bool parallel = true) {
    long long ok = 1;
#pragma omp parallel for simd if (parallel : parallel) reduction(min : ok)
    for (long long i = 0; i < n; ++i) {
        const long long flag = a[i] > 0 ? 1 : 0;
        ok = flag < ok ? flag : ok;
    }
    return ok;
}

/**
 * The smallest ``i`` in ``[begin, end)`` for which ``pred(i)`` holds, or ``end`` if none does.
 *
 * A short-circuiting parallel argmin over the firing indices. The range is split into
 * :cpp:func:`find_first_chunk`-sized chunks handed out ``schedule(dynamic, 1)``, so chunks are
 * claimed roughly in index order; every chunk reads a shared best-so-far hint and clamps its own
 * upper bound to it, which both skips a chunk that starts past the hint and truncates the one that
 * straddles it. Within a chunk the scan runs a ``simd`` min-reduction over one
 * :cvar:`FIND_FIRST_SIMD_BLOCK` at a time and stops at the first block that fires: a vectorized
 * loop cannot break, so the block is the early-exit granularity.
 *
 * The schedule is ``dynamic, 1`` and NOT ``guided`` or a block ``static``, which is a measured
 * choice rather than a default: both of those hand a contiguous prefix of the range to a single
 * thread, so an answer inside that prefix is found by a serial scan -- 5x slower than this on a
 * hit 10% into a 1e7-element range (500us against 95us, 8 threads). ``static, 1`` interleaves and
 * is competitive, but it cannot rebalance when the predicate's cost varies across the range.
 *
 * The answer and the hint are two different variables on purpose. The ANSWER is an OpenMP
 * ``reduction(min:)``, so it is exact. The HINT is shared and races by design: its
 * read-compare-write is not atomic as a whole, so a smaller value can be overwritten by a larger
 * one -- but every value it ever takes is either ``end`` or a real firing index, hence never below
 * the answer, so a lost update costs pruning and never correctness. Folding the two into one
 * shared word is exactly the lost-update bug, and it only shows up under load.
 *
 * @param begin     First index to test.
 * @param end       One past the last index to test; also the no-hit sentinel.
 * @param pred      Callable ``bool(long long)``, side-effect free.
 * @param parallel  Hand the chunks to OpenMP. Serial still keeps the blocked simd scan.
 */
template <typename Pred>
inline long long find_first_index(long long begin, long long end, Pred pred, bool parallel = true) {
    if (begin >= end) return end;
    const long long span = end - begin;
    const long long chunk = find_first_chunk(span, parallel);
    const long long nchunks = (span + chunk - 1) / chunk;
    long long best = end;
    long long hint = end;

#pragma omp parallel for schedule(dynamic, 1) if (parallel : parallel) reduction(min : best)
    for (long long c = 0; c < nchunks; ++c) {
        long long seen;
#pragma omp atomic read
        seen = hint;
        const long long lo = begin + c * chunk;
        if (lo >= seen) continue;  // this chunk cannot hold the minimum
        long long hi = lo + chunk;
        if (hi > end) hi = end;
        if (hi > seen) hi = seen;  // nothing at or past the hint can win

        long long found = end;
        for (long long b = lo; b < hi; b += FIND_FIRST_SIMD_BLOCK) {
            long long be = b + FIND_FIRST_SIMD_BLOCK;
            if (be > hi) be = hi;
            long long block = end;
#pragma omp simd reduction(min : block)
            for (long long i = b; i < be; ++i) {
                const long long v = pred(i) ? i : end;
                block = v < block ? v : block;
            }
            if (block < end) {
                found = block;
                break;
            }
        }
        if (found < end) {
            if (found < best) best = found;  // the ANSWER: a reduction, so no update can be lost
            long long cur;
#pragma omp atomic read
            cur = hint;
            if (found < cur) {
                // The HINT, advisory only. This read-compare-write is not atomic as a whole, so a
                // concurrent smaller write can be overwritten by a larger one -- which costs
                // pruning and nothing else, because every value the hint takes is a real firing
                // index and therefore never below the answer.
#pragma omp atomic write
                hint = found;
            }
        }
    }
    return best;
}

}  // namespace dace

#endif  // __DACE_DETECT_H
