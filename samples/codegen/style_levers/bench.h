// Shared timing harness for the codegen-pessimization reproducers.
//
// Rules it enforces (CODEGEN_STYLE_PERFORMANCE.md sections 1 and 3):
//   * median of >= 15 reps, never a single shot
//   * every reproducer carries a TWIN: a byte-identical copy of the FAST form under a
//     different symbol name. twin-vs-fast is the intra-binary noise+layout floor. Any
//     delta not comfortably above it is an ordinary draw, not a finding.
//   * results must be bit-identical across forms or the "lever" is a bug.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

static inline double now_sec() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
}

struct Stat {
    double median, min, p25, p75;
};

template <class F>
static Stat bench(F &&f, int reps = 21) {
    std::vector<double> t;
    t.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        double a = now_sec();
        f();
        double b = now_sec();
        t.push_back(b - a);
    }
    std::sort(t.begin(), t.end());
    Stat s;
    s.median = t[t.size() / 2];
    s.min = t.front();
    s.p25 = t[t.size() / 4];
    s.p75 = t[(3 * t.size()) / 4];
    return s;
}

// Spread of the timing distribution itself, as a fraction of the median.
static inline double iqr_frac(const Stat &s) {
    return (s.p75 - s.p25) / s.median;
}

static inline void row(const char *name, const Stat &s, double baseline_median) {
    std::printf("  %-28s median %9.3f ms   IQR %5.1f%%   ratio-vs-fast %6.2fx\n", name, s.median * 1e3,
                iqr_frac(s) * 100.0, s.median / baseline_median);
}

// Bit-exact comparison. A "fast" form that changes results is a bug, not a lever.
static inline bool bitsame(const void *a, const void *b, size_t n) {
    return std::memcmp(a, b, n) == 0;
}

#define CHECK_BITSAME(a, b, n, what)                                                                                   \
    do {                                                                                                               \
        if (!bitsame((a), (b), (n))) {                                                                                 \
            std::printf("  !! MISMATCH (%s) -- forms are NOT semantically identical, result invalid\n", (what));        \
            return 2;                                                                                                  \
        }                                                                                                              \
    } while (0)
