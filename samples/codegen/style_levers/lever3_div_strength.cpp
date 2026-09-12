// LEVER 3 -- integer division/modulo strength reduction: constant divisor vs laundered runtime.
//
// Index wrapping is everywhere in generated code: `a[i % M]`, `row = idx / W`, periodic
// boundaries, hash/scatter addressing. When the modulus/divisor is a COMPILE-TIME CONSTANT the
// backend strength-reduces the division to a multiply-high + shift (magic number). When the SAME
// value is only known at RUNTIME the backend must emit a hardware `idiv`/`div`, ~20-40 cycles on
// Zen4 and not pipelined.
//
// A code generator picks which: DaCe's sym2cpp can emit a known extent as a C++ literal (fast)
// or route it through a state-struct member / runtime symbol (slow). Same value, same result,
// worst-case codegen. The doc already establishes this axis is real and bidirectional -- section
// 4, "making an operand compile-time-constant made GCC WORSE on TSVC S276" is the OTHER
// direction of the very same lever.
//
// MECHANISM (named): division-by-constant strength reduction -- GCC expand_divmod() /
// choose_multiplier(); LLVM instcombine + SelectionDAG BuildUDIV/BuildSDIV (Granlund-Montgomery
// magic numbers). It fires ONLY on a constant operand; a runtime operand is a genuine `divl`.
// The compiler cannot re-derive the constant, so this lever does not get normalized away.
//
// Bit-identical: g_M holds exactly M, so a[i] % M == a[i] % g_M for every i.
#include "bench.h"

static volatile uint32_t g_M = 1000;  // laundered: same value as the constant, but runtime.

// `bias` varies per outer iteration so the result is NOT loop-invariant -> the pure-function
// call cannot be hoisted out of the timing loop, and both forms do the full N*OUTER work.

// SLOW: runtime modulus -> hardware divide every iteration.
__attribute__((noinline)) uint64_t histish_runtime(const uint32_t *a, int n, uint32_t M, uint32_t bias) {
    uint64_t acc = 0;
    for (int i = 0; i < n; ++i)
        acc += (a[i] + bias) % M;
    return acc;
}

// FAST: compile-time-constant modulus -> multiply-high + shift.
__attribute__((noinline)) uint64_t histish_const(const uint32_t *a, int n, uint32_t bias) {
    uint64_t acc = 0;
    for (int i = 0; i < n; ++i)
        acc += (a[i] + bias) % 1000u;
    return acc;
}

// TWIN of the fast form -> noise/layout floor.
__attribute__((noinline)) uint64_t histish_const_twin(const uint32_t *a, int n, uint32_t bias) {
    uint64_t acc = 0;
    for (int i = 0; i < n; ++i)
        acc += (a[i] + bias) % 1000u;
    return acc;
}

int main() {
    const int N = 1 << 16;
    const int OUTER = 300;
    std::vector<uint32_t> a(N);
    for (int i = 0; i < N; ++i)
        a[i] = uint32_t(i * 2654435761u);
    g_M = 1000;

    uint64_t rc = 0, rt = 0, rr = 0;
    auto run = [&](auto &&f, uint64_t *slot) {
        return [&, slot] {
            uint64_t s = 0;
            for (int k = 0; k < OUTER; ++k)
                s += f((uint32_t)k);
            *slot = s;
        };
    };

    Stat s_const = bench(run([&](uint32_t bias) { return histish_const(a.data(), N, bias); }, &rc));
    Stat s_twin = bench(run([&](uint32_t bias) { return histish_const_twin(a.data(), N, bias); }, &rt));
    Stat s_run = bench(run([&](uint32_t bias) { return histish_runtime(a.data(), N, (uint32_t)g_M, bias); }, &rr));

    CHECK_BITSAME(&rc, &rr, sizeof(uint64_t), "const vs runtime modulus");
    CHECK_BITSAME(&rc, &rt, sizeof(uint64_t), "const vs twin");

    std::printf("LEVER 3: integer modulo strength reduction (constant vs laundered runtime)  (N=%d, outer=%d)\n", N,
                OUTER);
    row("histish_const      [FAST]", s_const, s_const.median);
    row("histish_const_twin [FLOOR]", s_twin, s_const.median);
    row("histish_runtime    [SLOW]", s_run, s_const.median);
    std::printf("  checksum=%llu\n", (unsigned long long)rc);
    return 0;
}
