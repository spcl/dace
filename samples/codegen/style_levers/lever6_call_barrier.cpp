// LEVER 6 -- the same elementwise op as an INLINED tasklet body vs an out-of-line CALL.
//
// DaCe lowers an elementwise operation either by inlining its tasklet body into the map loop
// (InlineTaskletConnectors -> `out[i] = a[i]*c + b[i];`) or by emitting a call to a library-node
// / helper function. When that helper lives in another translation unit and LTO is off, the call
// is an OPTIMIZATION BARRIER: the compiler cannot vectorize across it, cannot hoist anything out
// of it, and pays call/spill overhead once per element.
//
//   FAST: `out[i] = a[i] * C + b[i];` inlined -> 8-wide `vfmadd...pd`.
//   SLOW: `out[i] = fma_op(a[i], C, b[i]);` where fma_op is in lever6_helper.cpp (separate TU,
//     no LTO) -> `call fma_op@plt` per element, scalar, ABI spills.
//
// Bit-identical: fma_op computes exactly `x*a+b` with the same rounding as the inlined form
// (both are a single FP multiply + add; no FMA-contraction is applied because the default here
// keeps them as separate ops on both sides -- verified by the bit check).
//
// Non-normalizable: with LTO off there is no way for the compiler to recover the inlined form
// from the call. This is a genuine barrier, not a legality flag the compiler can version around.
//
// MECHANISM (named): the inliner (LLVM Inliner / GCC IPA-inline) NOT firing across a TU boundary,
// which in turn blocks LoopVectorize (a call with unknown side effects cannot be vectorized) and
// LICM. The lever is inline-vs-call, exactly the tasklet-inline decision.
#include "bench.h"

double fma_op(double x, double a, double b);  // defined in lever6_helper.cpp (separate TU)

static const double C = 2.5;

// SLOW: out-of-line call per element.
__attribute__((noinline)) void axpy_call(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = fma_op(a[i], C, b[i]);
}

// FAST: inlined body.
__attribute__((noinline)) void axpy_inline(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * C + b[i];
}

// TWIN of the fast form -> noise/layout floor.
__attribute__((noinline)) void axpy_inline_twin(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = a[i] * C + b[i];
}

int main() {
    const int N = 1 << 12;   // L1/L2 resident -> compute/call-overhead bound
    const int OUTER = 5000;
    std::vector<double> a(N), b(N), o1(N), o2(N), o3(N);
    for (int i = 0; i < N; ++i) {
        a[i] = double((i % 97) + 1) * 0.5;
        b[i] = double((i % 89) + 1) * 0.25;
    }

    auto run = [&](auto &&f, std::vector<double> &out) {
        return [&] {
            for (int k = 0; k < OUTER; ++k)
                f(a.data(), b.data(), out.data(), N);
        };
    };

    Stat s_inl = bench(run(axpy_inline, o1));
    Stat s_twin = bench(run(axpy_inline_twin, o2));
    Stat s_call = bench(run(axpy_call, o3));

    CHECK_BITSAME(o1.data(), o3.data(), N * 8, "inline vs call");
    CHECK_BITSAME(o1.data(), o2.data(), N * 8, "inline vs twin");

    std::printf("LEVER 6: inlined tasklet body vs out-of-line library-call  (N=%d, outer=%d)\n", N, OUTER);
    row("axpy_inline      [FAST]", s_inl, s_inl.median);
    row("axpy_inline_twin [FLOOR]", s_twin, s_inl.median);
    row("axpy_call        [SLOW]", s_call, s_inl.median);
    std::printf("  checksum=%.6f\n", o1[N - 1]);
    return 0;
}
