// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Range-privatized scatter reductions on the CPU (``Map.omp_range_privatize``): each thread of a parallel loop takes
// one contiguous chunk of its iterations (``Order``: in index order, or sorted by the first row each writes when that
// narrows the threads' spans enough to pay), accumulates into a private copy of only the rows its chunk writes, and
// the copies are folded into the shared array by row owner in a fixed thread order.
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

// Sorting the iterations pays when the threads' private rows exceed this many bytes per thread unsorted.
constexpr int64_t kSortBytesPerThread = int64_t(1) << 20;
// ... and sorting narrows them at least this many times.
constexpr int64_t kSortGain = 2;
// Most buckets the iterations are counting-sorted into.
constexpr int64_t kSortBuckets = 4096;

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
  // frees them. The union of all spans is cut into one block per thread; in round ``r`` thread ``t`` folds into
  // block ``(t + r) % nthreads`` once that block's previous folder is done, so every block takes its threads'
  // rows in one fixed order and the result does not depend on the schedule.
  void combine(int t, int nthreads, T* shared) {
    int64_t first = std::numeric_limits<int64_t>::max(), last = std::numeric_limits<int64_t>::min();
    for (int s = 0; s < nthreads; ++s) {
      if (spans_[s].hi >= spans_[s].lo) {
        first = std::min(first, spans_[s].lo);
        last = std::max(last, spans_[s].hi);
      }
    }
    Span& mine = spans_[t];
    const int64_t total = last - first + 1;
    const T* src = mine.hi >= mine.lo ? mine.data.get() - mine.lo * row_ : nullptr;
    for (int r = 0; r < nthreads && last >= first; ++r) {
      const int b = (t + r) % nthreads;
      Span& owner = spans_[b];
      while (owner.turn.load(std::memory_order_acquire) != r) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
      }
      const int64_t lo = std::max(mine.lo, first + total * b / nthreads);
      const int64_t hi = std::min(mine.hi + 1, first + total * (b + 1) / nthreads);
      for (int64_t i = lo * row_; i < hi * row_; ++i) {
        wcr_fixed<R, T>::reduce(shared + i, src[i]);
      }
      owner.turn.store(r + 1, std::memory_order_release);
    }
    mine.data.reset();
  }

 private:
  struct alignas(64) Span {
    int64_t lo = 1, hi = 0;
    std::unique_ptr<T[]> data;
    std::atomic<int> turn{0};
  };
  std::vector<Span> spans_;
  const int64_t rows_, row_;
};

// The iterations ``[0, n)`` of a range-privatized map over a team, one contiguous chunk per thread, and the rows each
// thread's chunk writes per group of accumulators sharing their write sites. The pre-pass ``note``s every iteration
// in identity order; ``arrange`` then keeps that order or counting-sorts the iterations by the first row each
// writes, whichever the private rows favour, and sets the spans of the chunks in the chosen order.
class Order {
 public:
  static constexpr int64_t kNone = std::numeric_limits<int64_t>::max();

  Order(int nthreads, int64_t n, int groups)
      : n_(std::max<int64_t>(n, 0)), groups_(groups), stats_(nthreads), keys_(new int64_t[std::max<int64_t>(n, 1)]) {
    for (Stats& s : stats_) {
      s.lo.assign(groups, kNone);
      s.hi.assign(groups, std::numeric_limits<int64_t>::min());
      s.updates.assign(groups, 0);
      s.dlo.assign(groups, kNone);
      s.dhi.assign(groups, std::numeric_limits<int64_t>::min());
    }
  }

  // The first iteration of chunk ``t`` of ``nthreads``.
  int64_t first(int t, int nthreads) const { return n_ * t / nthreads; }

  // The iteration at position ``k`` of the chosen order.
  int64_t operator[](int64_t k) const { return perm_ ? perm_[k] : k; }

  // Thread ``t``'s iteration ``k`` writes rows ``[lo[g], hi[g]]`` of group ``g`` (none where ``lo[g] > hi[g]``),
  // ``updates[g]`` times.
  void note(int t, int64_t k, const int64_t* lo, const int64_t* hi, const int64_t* updates) {
    Stats& s = stats_[t];
    int64_t key = kNone;
    for (int g = 0; g < groups_; ++g) {
      s.updates[g] += updates[g];
      if (lo[g] > hi[g]) {
        continue;
      }
      key = std::min(key, lo[g]);
      s.lo[g] = std::min(s.lo[g], lo[g]);
      s.hi[g] = std::max(s.hi[g], hi[g]);
    }
    keys_[k] = key;
    if (key == kNone) {
      return;
    }
    s.kmin = std::min(s.kmin, key);
    s.kmax = std::max(s.kmax, key);
    for (int g = 0; g < groups_; ++g) {
      if (lo[g] <= hi[g]) {
        s.dlo[g] = std::min(s.dlo[g], lo[g] - key);
        s.dhi[g] = std::max(s.dhi[g], hi[g] - key);
      }
    }
  }

  // Every thread of the team, after its pre-pass: sorts when the private rows of all threads unsorted exceed
  // ``kSortBytesPerThread`` per thread and ``kSortGain`` times their estimate sorted (``row_bytes[g]``: the bytes of
  // a row of every accumulator of group ``g``). Every thread decides alike from the published spans.
  void arrange(int t, int nthreads, const int64_t* row_bytes) {
#pragma omp barrier
    int64_t kmin = kNone, kmax = std::numeric_limits<int64_t>::min(), unsorted = 0, sorted = 0;
    for (int s = 0; s < nthreads; ++s) {
      kmin = std::min(kmin, stats_[s].kmin);
      kmax = std::max(kmax, stats_[s].kmax);
      for (int g = 0; g < groups_; ++g) {
        if (stats_[s].hi[g] >= stats_[s].lo[g]) {
          unsorted += (stats_[s].hi[g] - stats_[s].lo[g] + 1) * row_bytes[g];
        }
      }
    }
    if (kmax < kmin || unsorted <= kSortBytesPerThread * nthreads) {
      return;
    }
    const int64_t keys = kmax - kmin + 1;
    std::vector<int64_t> dlo(groups_, kNone), dhi(groups_, std::numeric_limits<int64_t>::min());
    for (int g = 0; g < groups_; ++g) {
      for (int s = 0; s < nthreads; ++s) {
        dlo[g] = std::min(dlo[g], stats_[s].dlo[g]);
        dhi[g] = std::max(dhi[g], stats_[s].dhi[g]);
      }
      if (dhi[g] >= dlo[g]) {
        sorted += nthreads * ((keys + nthreads - 1) / nthreads + dhi[g] - dlo[g] + 1) * row_bytes[g];
      }
    }
    if (unsorted <= kSortGain * sorted) {
      return;
    }
    sort(t, nthreads, kmin, keys);
    int64_t lo = kNone, hi = std::numeric_limits<int64_t>::min();
    for (int64_t k = first(t, nthreads); k < first(t + 1, nthreads); ++k) {
      const int64_t key = keys_[perm_[k]];
      if (key != kNone) {
        lo = std::min(lo, key);
        hi = std::max(hi, key);
      }
    }
    Stats& mine = stats_[t];
    for (int g = 0; g < groups_; ++g) {
      int64_t updates = 0;
      for (int s = 0; s < nthreads; ++s) {
        updates += stats_[s].updates[g];
      }
      mine.sorted_updates.push_back(updates / nthreads);
      mine.sorted_lo.push_back(hi < lo || dhi[g] < dlo[g] ? 1 : lo + dlo[g]);
      mine.sorted_hi.push_back(hi < lo || dhi[g] < dlo[g] ? 0 : hi + dhi[g]);
    }
#pragma omp barrier
    mine.lo.swap(mine.sorted_lo);
    mine.hi.swap(mine.sorted_hi);
    mine.updates.swap(mine.sorted_updates);
  }

  // Thread ``t``'s rows and updates of group ``g`` in the chosen order, once ``arrange``d.
  int64_t lo(int t, int g) const { return stats_[t].lo[g]; }
  int64_t hi(int t, int g) const { return stats_[t].hi[g]; }
  int64_t updates(int t, int g) const { return stats_[t].updates[g]; }

 private:
  // Stable counting sort of the iterations by key into at most ``kSortBuckets`` buckets, the keyless ones last.
  void sort(int t, int nthreads, int64_t kmin, int64_t keys) {
    const int64_t buckets = std::min(keys, kSortBuckets);
    auto bucket = [&](int64_t k) {
      const int64_t key = keys_[k];
      return key == kNone ? buckets : (key - kmin) * buckets / keys;
    };
    std::vector<int64_t>& count = stats_[t].count;
    count.assign(buckets + 1, 0);
    for (int64_t k = first(t, nthreads); k < first(t + 1, nthreads); ++k) {
      ++count[bucket(k)];
    }
#pragma omp single
    perm_.reset(new int64_t[std::max<int64_t>(n_, 1)]);
    std::vector<int64_t> offset(buckets + 1);
    int64_t at = 0;
    for (int64_t b = 0; b <= buckets; ++b) {
      for (int s = 0; s < nthreads; ++s) {
        if (s == t) {
          offset[b] = at;
        }
        at += stats_[s].count[b];
      }
    }
    for (int64_t k = first(t, nthreads); k < first(t + 1, nthreads); ++k) {
      perm_[offset[bucket(k)]++] = k;
    }
#pragma omp barrier
  }

  struct alignas(64) Stats {
    int64_t kmin = kNone, kmax = std::numeric_limits<int64_t>::min();
    std::vector<int64_t> lo, hi, updates, dlo, dhi, count, sorted_lo, sorted_hi, sorted_updates;
  };
  const int64_t n_;
  const int groups_;
  std::vector<Stats> stats_;
  std::unique_ptr<int64_t[]> keys_, perm_;
};

}  // namespace range_private
}  // namespace dace
