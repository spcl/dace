// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_DETECT_H
#define __DACE_DETECT_H

// Duplicate detection and find-first helpers shared by library-node expansions.

#include <cmath>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "types.h"

namespace dace {

//: Elements one SIMD block of find_first_index scans before checking for an early exit.
static constexpr long long FIND_FIRST_SIMD_BLOCK = 64;

//: Chunk size of find_first_index, as a multiple of sqrt(span).
static constexpr double FIND_FIRST_CHUNK_SCALE = 8.0;

//: Floor on how many chunks each thread gets.
static constexpr long long FIND_FIRST_CHUNKS_PER_THREAD = 4;

// Chunk size find_first_index splits a span-element range into.
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
 * Returns 1 if any duplicate was found, 0 otherwise. ``owner`` needs no initialization;
 * values outside ``[0, capacity)`` are skipped.
 */
template <typename T, typename TagT>
inline long long detect_collision(const T* idx, long long n, TagT* owner, long long capacity, bool parallel = true) {
#pragma omp parallel for if (parallel : parallel)
  for (long long i = 0; i < n; ++i) {
    const long long v = static_cast<long long>(idx[i]);
    if (v >= 0 && v < capacity) owner[v] = static_cast<TagT>(i);
  }
  long long c = 0;
#pragma omp parallel for simd if (parallel : parallel) reduction(| : c)
  for (long long i = 0; i < n; ++i) {
    const long long v = static_cast<long long>(idx[i]);
    if (v >= 0 && v < capacity) c |= (static_cast<long long>(owner[v]) != i) ? 1LL : 0LL;
  }
  return c;
}

// detect_collision sizing its own tag buffer from max(idx); costs one extra pass plus an
// allocation, so prefer the overload above when the scattered array's domain is known.
template <typename T>
inline long long detect_collision(const T* idx, long long n, bool parallel = true) {
  long long mx = 0;
#pragma omp parallel for simd if (parallel : parallel) reduction(max : mx)
  for (long long i = 0; i < n; ++i) {
    const long long v = static_cast<long long>(idx[i]);
    mx = v > mx ? v : mx;
  }
  std::unique_ptr<long long[]> owner(new long long[static_cast<size_t>(mx) + 1]);
  return detect_collision(idx, n, owner.get(), mx + 1, parallel);
}

// Whether every element of `a` is strictly positive: 1 if all are, 0 if any is not.
template <typename T>
inline long long detect_all_positive(const T* a, long long n, bool parallel = true) {
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
 * ``pred`` must be side-effect free. The shared hint races by design but never goes below
 * the answer, so a lost update only costs pruning, never correctness.
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
        // advisory hint, races by design (see function doc)
#pragma omp atomic write
        hint = found;
      }
    }
  }
  return best;
}

}  // namespace dace

#endif  // __DACE_DETECT_H
