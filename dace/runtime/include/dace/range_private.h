// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Range-privatized scatter reductions on the CPU (``Map.omp_range_privatize``): each thread of a statically
// scheduled parallel loop accumulates into a private copy of only the rows its chunk writes, and the copies are
// folded into the shared array by row owner, between two barriers.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "reduction.h"

namespace dace {
namespace range_private {

// The value a private row starts at: the identity of ``R`` on ``T``.
template <ReductionType R, typename T>
inline T identity() {
  if constexpr (R == ReductionType::Sum || R == ReductionType::Logical_Or || R == ReductionType::Bitwise_Or ||
                R == ReductionType::Bitwise_Xor) {
    return T(0);
  } else if constexpr (R == ReductionType::Product || R == ReductionType::Logical_And) {
    return T(1);
  } else if constexpr (R == ReductionType::Bitwise_And) {
    static_assert(std::is_integral<T>::value, "bitwise reduction of a non-integral type");
    return T(~T(0));
  } else if constexpr (R == ReductionType::Min) {
    static_assert(std::numeric_limits<T>::is_specialized, "min reduction of an unordered type");
    return std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
  } else {
    static_assert(R == ReductionType::Max, "operator without a range-private identity");
    static_assert(std::numeric_limits<T>::is_specialized, "max reduction of an unordered type");
    return std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity()
                                                : std::numeric_limits<T>::lowest();
  }
}

// A thread updates the shared rows atomically instead of privatizing its span when the span has more than this
// many rows per update: filling and folding a row then costs more than the atomic updates it saves (fitted on
// MI300A to random indices, 24 and 48 threads; sorted sparse indices would break even nearer one row per update).
constexpr int64_t kRowsPerUpdate = 6;

// One accumulator of ``rows`` rows of ``row`` elements, shared by a team of at most ``nthreads`` threads.
// ``kAtomic``: the body can update the shared rows atomically instead, which a thread does when its span is long.
template <ReductionType R, typename T, bool kAtomic>
class Accumulator {
 public:
  Accumulator(int nthreads, int64_t rows, int64_t row) : spans_(nthreads), rows_(rows), row_(row) {}

  // Thread ``t``, which writes rows ``[lo, hi]`` with ``updates`` updates: returns the base its body accumulates
  // through, ``shared`` itself when ``atomic`` is set, else a private copy of those rows filled with the identity.
  T* open(int t, T* shared, int64_t lo, int64_t hi, int64_t updates, bool& atomic) {
    lo = std::max<int64_t>(lo, 0);
    hi = std::min<int64_t>(hi, rows_ - 1);
    atomic = kAtomic && (hi < lo || hi - lo + 1 > kRowsPerUpdate * updates);
    Span& span = spans_[t];
    if (atomic || hi < lo) {
      span.lo = 1;
      span.hi = 0;
      return shared;
    }
    const int64_t size = (hi - lo + 1) * row_;
    span.data.reset(new T[size]);
    std::fill_n(span.data.get(), size, identity<R, T>());
    span.lo = lo;
    span.hi = hi;
    return span.data.get() - lo * row_;
  }

  // Thread ``t`` of ``nthreads``, once every thread's body is done: folds its private rows into ``shared`` and
  // frees them; the spans stay published for the other threads' folds. The union of all private spans is cut into one block per thread, each folded under its own lock;
  // a thread starts at its own block, so the folds run in parallel wherever the spans overlap.
  void combine(int t, int nthreads, T* shared) {
    Span& mine = spans_[t];
    if (mine.hi < mine.lo) {
      return;
    }
    int64_t first = std::numeric_limits<int64_t>::max(), last = std::numeric_limits<int64_t>::min();
    for (int s = 0; s < nthreads; ++s) {
      if (spans_[s].hi >= spans_[s].lo) {
        first = std::min(first, spans_[s].lo);
        last = std::max(last, spans_[s].hi);
      }
    }
    const int64_t total = last - first + 1;
    const T* src = mine.data.get() - mine.lo * row_;
    for (int k = 0; k < nthreads; ++k) {
      const int b = (t + k) % nthreads;
      const int64_t lo = std::max(mine.lo, first + total * b / nthreads);
      const int64_t hi = std::min(mine.hi + 1, first + total * (b + 1) / nthreads);
      if (lo >= hi) {
        continue;
      }
      Span& owner = spans_[b];
      while (owner.lock.exchange(true, std::memory_order_acquire)) {
        while (owner.lock.load(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(__i386__)
          __builtin_ia32_pause();
#endif
        }
      }
      for (int64_t i = lo * row_; i < hi * row_; ++i) {
        wcr_fixed<R, T>::reduce(shared + i, src[i]);
      }
      owner.lock.store(false, std::memory_order_release);
    }
    mine.data.reset();
  }

 private:
  struct alignas(64) Span {
    int64_t lo = 1, hi = 0;
    std::unique_ptr<T[]> data;
    std::atomic<bool> lock{false};
  };
  std::vector<Span> spans_;
  const int64_t rows_, row_;
};

}  // namespace range_private
}  // namespace dace
