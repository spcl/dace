// LEVER 1 -- accumulator materialization: reduce into MEMORY vs into a LOCAL.
//
// This is the shape DaCe actually chooses between: a WCR edge onto an output array can be
// emitted as a read-modify-write of the destination element (cpu.py write_and_resolve_expr,
// line 1242) or the accumulator can be held in a local and stored once at map exit.
//
// MECHANISM (named): LICM's scalar promotion -- LLVM LICMPass ->
// promoteLoopAccessesToScalars(); GCC's equivalent is -ftree-loop-im / store-motion
// (pass_lim). Promotion is only legal if the loop's stores cannot alias its loads. All three
// pointers here are `double*`/`int64_t*`, so TBAA gives no disambiguation and BasicAA cannot
// prove distinctness of two function parameters. Promotion therefore FAILS, the accumulator
// stays in memory, and:
//   (a) every iteration pays a store->load forward on the critical path, and
//   (b) LoopVectorize never sees a reduction PHI, so the loop cannot vectorize at all.
//
// Two computations on purpose, to show the lever's magnitude is set by what promotion UNLOCKS:
//   int64 sum  -- integer add is associative, so promotion unlocks full vectorization.
//   fp64 dot   -- without -ffast-math the FP reduction is ordered, so promotion unlocks only
//                 the store->load forward removal. Same lever, much smaller delta.
//
// All forms are bit-identical: same operations, same order, in every variant.
#include "bench.h"

// ---------------------------------------------------------------- int64 sum
__attribute__((noinline)) void sum_mem(const int64_t *a, int64_t *out, int n) {
    for (int i = 0; i < n; ++i)
        out[0] += a[i];
}

__attribute__((noinline)) void sum_mem_restrict(const int64_t *__restrict__ a, int64_t *__restrict__ out, int n) {
    for (int i = 0; i < n; ++i)
        out[0] += a[i];
}

__attribute__((noinline)) void sum_reg(const int64_t *a, int64_t *out, int n) {
    int64_t s = out[0];
    for (int i = 0; i < n; ++i)
        s += a[i];
    out[0] = s;
}

// TWIN of sum_reg: byte-identical body, different symbol. Measures noise+layout floor.
__attribute__((noinline)) void sum_reg_twin(const int64_t *a, int64_t *out, int n) {
    int64_t s = out[0];
    for (int i = 0; i < n; ++i)
        s += a[i];
    out[0] = s;
}

// ---------------------------------------------------------------- fp64 dot
__attribute__((noinline)) void dot_mem(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i)
        out[0] += a[i] * b[i];
}

__attribute__((noinline)) void dot_reg(const double *a, const double *b, double *out, int n) {
    double s = out[0];
    for (int i = 0; i < n; ++i)
        s += a[i] * b[i];
    out[0] = s;
}

int main() {
    const int N = 1 << 16;
    const int OUTER = 200;

    std::vector<int64_t> ia(N);
    std::vector<double> da(N), db(N);
    for (int i = 0; i < N; ++i) {
        ia[i] = int64_t(i * 2654435761u) >> 7;
        da[i] = double((i % 97) + 1) * 0.5;
        db[i] = double((i % 89) + 1) * 0.25;
    }

    int64_t r_mem = 0, r_memr = 0, r_reg = 0, r_twin = 0;
    double d_mem = 0, d_reg = 0;

    auto run = [&](auto &&f, auto *slot) {
        return [&, slot] {
            *slot = 0;
            for (int k = 0; k < OUTER; ++k)
                f(slot);
        };
    };

    Stat s_reg = bench(run([&](int64_t *o) { sum_reg(ia.data(), o, N); }, &r_reg));
    Stat s_twin = bench(run([&](int64_t *o) { sum_reg_twin(ia.data(), o, N); }, &r_twin));
    Stat s_memr = bench(run([&](int64_t *o) { sum_mem_restrict(ia.data(), o, N); }, &r_memr));
    Stat s_mem = bench(run([&](int64_t *o) { sum_mem(ia.data(), o, N); }, &r_mem));

    Stat d_s_reg = bench(run([&](double *o) { dot_reg(da.data(), db.data(), o, N); }, &d_reg));
    Stat d_s_mem = bench(run([&](double *o) { dot_mem(da.data(), db.data(), o, N); }, &d_mem));

    CHECK_BITSAME(&r_reg, &r_twin, sizeof(int64_t), "sum_reg vs twin");
    CHECK_BITSAME(&r_reg, &r_mem, sizeof(int64_t), "sum_reg vs sum_mem");
    CHECK_BITSAME(&r_reg, &r_memr, sizeof(int64_t), "sum_reg vs sum_mem_restrict");
    CHECK_BITSAME(&d_reg, &d_mem, sizeof(double), "dot_reg vs dot_mem");

    std::printf("LEVER 1: accumulator materialization  (N=%d, outer=%d)\n", N, OUTER);
    std::printf(" int64 sum (promotion unlocks vectorization):\n");
    row("sum_reg      [FAST]", s_reg, s_reg.median);
    row("sum_reg_twin [FLOOR]", s_twin, s_reg.median);
    row("sum_mem_restrict", s_memr, s_reg.median);
    row("sum_mem      [SLOW]", s_mem, s_reg.median);
    std::printf(" fp64 dot (ordered FP: promotion unlocks only store->load removal):\n");
    row("dot_reg      [FAST]", d_s_reg, d_s_reg.median);
    row("dot_mem      [SLOW]", d_s_mem, d_s_reg.median);
    std::printf(" checksums: int64=%lld  fp64=%.6f\n", (long long)r_reg, d_reg);
    return 0;
}
