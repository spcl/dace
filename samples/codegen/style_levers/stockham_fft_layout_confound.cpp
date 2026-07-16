// stockham_fft "readable-codegen ~1.3x regression" -- REFUTED as a source-form lever; it is the
// LAYOUT CONFOUND (CODEGEN_STYLE_PERFORMANCE.md section 3), same verdict as heat3d_layout_confound.cpp.
// NOT a leverN: a documented negative result, in the spirit of heat3d_layout_confound.cpp and lever2.
//
// stockham_fft is reported ~1.3x slower under the experimental readable CPU codegen at single core.
// The butterfly itself is a `cblas_cgemm` -- BYTE-IDENTICAL in both codegens -- so it cannot be the
// cause. The only per-element work that differs legacy vs experimental is two streaming loops over
// N = R^K complex64 (paper preset R=4,K=10 -> N=2^20): the twiddle-multiply and the stride-permute
// (TensorTranspose). This file carries BOTH, in both source forms + a twin + one hybrid per axis,
// ALL IN ONE BINARY, single-thread. The forms are dispatched THROUGH FUNCTION POINTERS on purpose:
// in the real kernel R and K arrive as runtime `int64_t` parameters, so the trip counts / strides
// (`dace::math::ipow(R, ...)`) are RUNTIME values -- calling by pointer keeps them runtime here too,
// instead of letting the compiler constant-fold literals it would never get to fold in production.
//
// The source-form axes that differ legacy vs experimental (each flipped one at a time):
//   B  copy-in named locals  { cf __in1=..; __in2=..; __out; __out=(__in1*__in2); out[i]=__out; }  (legacy)
//        vs inlined single store  out[i] = a[i]*b[i]                                                (exp)
//   C  a[a_idx(i)] constexpr index helper (exp)  vs  a[i] inline index (legacy)
//   E  cf* __restrict__ heap temps (exp)  vs  plain cf* (legacy)
//   (loop index type int vs long long -- also checked, irrelevant)
//
// WHY IT COLLAPSES (the section-6 criterion "a knob the compiler can re-derive is not a knob"):
//   * twiddle-multiply: MEMORY-BANDWIDTH bound (3x8 MB streamed). Every axis, both compilers, 1.00x.
//   * TensorTranspose: a PURE COPY with a unit-stride inner run. With RUNTIME R,K the inner trip
//     count is a runtime register, so g++'s memcpy/loop-distribute idiom (-ftree-loop-distribute-
//     patterns) does NOT fire for EITHER source form -- both compile to the same scalar per-element
//     complex copy. Verified in the REAL generated .s: the legacy transpose omp_fn and the
//     experimental one (_omp_fn.5) are BOTH scalar (0 `bl memcpy`, 0 ld1/st1, ~145 vs ~141 insns) --
//     same instruction selection, different block order. So axis B/C/E all land 1.00x here.
//   * A tempting FALSE POSITIVE to be aware of: a standalone microbench that calls the transpose
//     with LITERAL R=4,K=10,i=5 lets g++ .constprop the trip count to 256 and THEN fire memcpy on
//     the direct form but not through the named local -> a clean 1.27x (g++ only; clang never fires
//     it, ~27x slower for all forms). That fast form is reachable ONLY under constant-folded dims,
//     which the runtime-symbol kernel never has -- so it is not a knob the generator can pull, and
//     it points the WRONG way anyway (experimental already emits the direct form).
//
// KERNEL-LEVEL (the actual claim): twiddle flat + transpose flat => the two codegens tie. Controlled
// interleaved single-process measurement of the real compiled kernel, paper preset, single core
// (Neoverse-V2 Grace):
//     g++    experimental/legacy = 0.999x      clang  experimental/legacy = 0.995x
//     (recorded S-preset sweep: g++ 0.995x, clang 0.996x) -- the reported ~1.3x is inside the layout
//     floor (heat_3d's layout_probe.cpp: 5 byte-identical twins span 1.28x on g++), and its sign is
//     not fixed (separate-invocation runs flip which codegen looks faster). No semantic cause survives.
//
// No codegen_params flag is warranted: there is no fast form for the generator to emit.
#include "bench.h"
#include <complex>
using cf = std::complex<float>;

// dace's ipow: runtime loop, constexpr+forceinline; loop-invariant exponents here.
static inline constexpr long long mb_ipow(long long a, unsigned b) {
    long long r = 1;
    for (unsigned k = 0; k < b; ++k) r *= a;
    return r;
}
// experimental "readable" codegen's constexpr index helpers (verbatim shapes).
static inline constexpr long long id1(long long d0) { return d0; }
static inline constexpr long long yv_idx(long long d0, long long d1, long long d2, long long K, long long R, long long i) {
    return (R * d0 * mb_ipow(R, K - i - 1) + d1 * mb_ipow(R, K - i - 1) + d2);
}
static inline constexpr long long tv_idx(long long d0, long long d1, long long d2, long long K, long long R, long long i) {
    return (d0 * mb_ipow(R, i) * mb_ipow(R, K - i - 1) + d1 * mb_ipow(R, K - i - 1) + d2);
}

// ============================================================ TWIDDLE-MULTIPLY (memory-bound)
// EXPERIMENTAL: restrict + index-fn + inlined store.
__attribute__((noinline)) void tw_exp(cf *__restrict__ o, cf *__restrict__ a, cf *__restrict__ b, long long n) {
    for (int i = 0; i < n; ++i) o[id1(i)] = (a[id1(i)] * b[id1(i)]);
}
__attribute__((noinline)) void tw_exp_twin(cf *__restrict__ o, cf *__restrict__ a, cf *__restrict__ b, long long n) {
    for (int i = 0; i < n; ++i) o[id1(i)] = (a[id1(i)] * b[id1(i)]);
}
// LEGACY: non-restrict + copy-in named locals + plain index.
__attribute__((noinline)) void tw_legacy(cf *o, cf *a, cf *b, long long n) {
    for (int i = 0; i < n; ++i) {
        cf __in1 = a[i], __in2 = b[i], __out;
        __out = (__in1 * __in2);
        o[i] = __out;
    }
}

// ============================================================ TENSOR-TRANSPOSE (pure copy, unit-stride inner)
// EXPERIMENTAL: restrict + index-fns + DIRECT store (memcpy idiom fires ONLY if dims fold to constants).
__attribute__((noinline)) void tr_exp(const cf *__restrict__ in, cf *__restrict__ out, long long K, long long R, long long i) {
    for (long long a = 0; a < mb_ipow(R, i); ++a)
        for (long long b = 0; b < R; ++b)
            for (long long c = 0; c < mb_ipow(R, K - i - 1); ++c)
                out[tv_idx(b, a, c, K, R, i)] = in[yv_idx(a, b, c, K, R, i)];
}
__attribute__((noinline)) void tr_exp_twin(const cf *__restrict__ in, cf *__restrict__ out, long long K, long long R, long long i) {
    for (long long a = 0; a < mb_ipow(R, i); ++a)
        for (long long b = 0; b < R; ++b)
            for (long long c = 0; c < mb_ipow(R, K - i - 1); ++c)
                out[tv_idx(b, a, c, K, R, i)] = in[yv_idx(a, b, c, K, R, i)];
}
// LEGACY: restrict (it is an out-of-line helper with restrict args) + INLINE index + copy-in locals.
__attribute__((noinline)) void tr_legacy(const cf *__restrict__ in, cf *__restrict__ out, long long K, long long R, long long i) {
    for (long long a = 0; a < mb_ipow(R, i); ++a)
        for (long long b = 0; b < R; ++b)
            for (long long c = 0; c < mb_ipow(R, K - i - 1); ++c) {
                cf _inp = in[(R * a * mb_ipow(R, K - i - 1) + b * mb_ipow(R, K - i - 1) + c)];
                cf _out;
                _out = _inp;
                out[(a * mb_ipow(R, K - i - 1) + b * mb_ipow(R, i) * mb_ipow(R, K - i - 1) + c)] = _out;
            }
}
// Axis B hybrid: experimental index-fns but WITH copy-in locals (flips ONLY axis B toward legacy).
__attribute__((noinline)) void tr_exp_locals(const cf *__restrict__ in, cf *__restrict__ out, long long K, long long R, long long i) {
    for (long long a = 0; a < mb_ipow(R, i); ++a)
        for (long long b = 0; b < R; ++b)
            for (long long c = 0; c < mb_ipow(R, K - i - 1); ++c) {
                cf _inp = in[yv_idx(a, b, c, K, R, i)];
                cf _out;
                _out = _inp;
                out[tv_idx(b, a, c, K, R, i)] = _out;
            }
}
// Axis C hybrid: legacy INLINE index but DIRECT store (flips ONLY axis B toward exp; index still inline).
__attribute__((noinline)) void tr_legacy_direct(const cf *__restrict__ in, cf *__restrict__ out, long long K, long long R, long long i) {
    for (long long a = 0; a < mb_ipow(R, i); ++a)
        for (long long b = 0; b < R; ++b)
            for (long long c = 0; c < mb_ipow(R, K - i - 1); ++c)
                out[(a * mb_ipow(R, K - i - 1) + b * mb_ipow(R, i) * mb_ipow(R, K - i - 1) + c)] =
                    in[(R * a * mb_ipow(R, K - i - 1) + b * mb_ipow(R, K - i - 1) + c)];
}

int main() {
    // Sourced through `volatile` so R,K,i stay RUNTIME even under -fipa-cp devirtualization --
    // matching the real kernel (R,K are runtime int64 params). Constant dims would let g++ fold the
    // trip count and fire the memcpy idiom, a fast form the runtime-symbol kernel can never reach.
    volatile long long vK = 10, vR = 4, vi = 5;
    const long long K = vK, R = vR, i = vi;
    const long long N = R * mb_ipow(R, i) * mb_ipow(R, K - i - 1); // 2^20
    const int ROUNDS = 41;

    std::vector<cf> A(N), B(N), In(N), o(N), oref(N);
    for (long long j = 0; j < N; ++j) {
        A[j] = cf(float((j % 97) + 1) * 0.5f, float((j % 53) + 1) * 0.25f);
        B[j] = cf(float((j % 89) + 1) * 0.25f, float((j % 61) + 1) * 0.5f);
        In[j] = A[j];
    }

    // ---- TWIDDLE group (baseline = tw_exp) ----
    struct TwForm { const char *name; void (*fn)(cf *, cf *, cf *, long long); };
    TwForm tw[] = {{"tw_exp      [EXP]", (void (*)(cf *, cf *, cf *, long long))tw_exp},
                   {"tw_exp_twin [FLOOR]", (void (*)(cf *, cf *, cf *, long long))tw_exp_twin},
                   {"tw_legacy   [LEG]", tw_legacy}};
    const int NTW = 3;
    tw_exp(oref.data(), A.data(), B.data(), N); // reference for twiddle
    std::vector<cf> twref(oref);
    std::vector<std::vector<double>> tt(NTW);
    for (int w = 0; w < 3; ++w)
        for (int f = 0; f < NTW; ++f) tw[f].fn(o.data(), A.data(), B.data(), N);
    for (int r = 0; r < ROUNDS; ++r)
        for (int f = 0; f < NTW; ++f) {
            double a = now_sec();
            tw[f].fn(o.data(), A.data(), B.data(), N);
            double b = now_sec();
            tt[f].push_back(b - a);
            if (r == 0) CHECK_BITSAME(twref.data(), o.data(), N * sizeof(cf), tw[f].name);
        }

    // ---- TRANSPOSE group (baseline = tr_exp) ----
    struct TrForm { const char *name; void (*fn)(const cf *, cf *, long long, long long, long long); };
    TrForm tr[] = {{"tr_exp        [EXP: direct]", tr_exp},
                   {"tr_exp_twin   [FLOOR]", tr_exp_twin},
                   {"tr_exp_locals (axisB->leg)", tr_exp_locals},
                   {"tr_legacy_direct (axisC)", tr_legacy_direct},
                   {"tr_legacy     [LEG: locals]", tr_legacy}};
    const int NTR = 5;
    tr_exp(In.data(), oref.data(), K, R, i);
    std::vector<cf> trref(oref);
    std::vector<std::vector<double>> rt(NTR);
    for (int w = 0; w < 3; ++w)
        for (int f = 0; f < NTR; ++f) tr[f].fn(In.data(), o.data(), K, R, i);
    for (int r = 0; r < ROUNDS; ++r)
        for (int f = 0; f < NTR; ++f) {
            double a = now_sec();
            tr[f].fn(In.data(), o.data(), K, R, i);
            double b = now_sec();
            rt[f].push_back(b - a);
            if (r == 0) CHECK_BITSAME(trref.data(), o.data(), N * sizeof(cf), tr[f].name);
        }

    auto stat = [](std::vector<double> &v) {
        std::sort(v.begin(), v.end());
        Stat s;
        s.median = v[v.size() / 2];
        s.min = v.front();
        s.p25 = v[v.size() / 4];
        s.p75 = v[(3 * v.size()) / 4];
        return s;
    };

    std::printf("stockham_fft source-form bisection  (N=%lld complex64, rounds=%d, interleaved, single-thread)\n", N,
                ROUNDS);
    std::printf(" TWIDDLE-MULTIPLY (memory-bound; baseline = tw_exp):\n");
    Stat b0 = stat(tt[0]);
    for (int f = 0; f < NTW; ++f) {
        Stat s = stat(tt[f]);
        row(tw[f].name, s, b0.median);
    }
    std::printf(" TENSOR-TRANSPOSE (pure copy, stage i=%lld; baseline = tr_exp):\n", i);
    Stat r0 = stat(rt[0]);
    for (int f = 0; f < NTR; ++f) {
        Stat s = stat(rt[f]);
        row(tr[f].name, s, r0.median);
    }
    std::printf(" NOTE: with RUNTIME R,K (as in the real kernel) every form ties -- the transpose is scalar for\n");
    std::printf("       BOTH source forms (memcpy idiom needs a constant-folded trip count the kernel never has).\n");
    return 0;
}
