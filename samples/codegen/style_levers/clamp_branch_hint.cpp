// CLAMP BRANCH HINT -- C++20 [[unlikely]] on the clamp pattern, vs no hint.
//
// The candidate: scientific/weather stencils are full of the clamp `if (x < lo) x = lo;`.
// If such a branch survived to codegen, it fires rarely, so [[unlikely]] should pay.
//
// REFUTED. The premise fails at step one: THE BRANCH DOES NOT SURVIVE -O3. This file is kept
// as the reproducer for that, per the directory's selection criterion -- "a lever survives
// only when the fast form is not recoverable by a runtime guard." Here the compiler does not
// even need a guard: the clamp is if-converted unconditionally.
//
// The forms below are the shapes the DaCe CPU generators ACTUALLY emit for a clamp (dumped
// from `sdfg.generate_code()[0].clean_code`): a tasklet computing `bool __tmp0 = (r0 < r1);`
// then branching on it, const-ref inputs, ref output, called from the map loop.
//
// MECHANISM (named), verified with objdump -d on gcc 15.2 / clang 21.1 / nvc++ 26.3. icpx is NOT
// installed on the reference box, so every icpx claim below is ASSUMED (it is clang-based), not
// measured -- re-run there before relying on it:
//
//   * VALUE-SELECT form (`if (c) w = lo; else w = x;` -- DaCe's clamp_ifelse): if-converted to
//     vmaxsd, then vectorized to vmaxpd. This is GCC's ifcvt / LLVM's SimplifyCFG
//     speculation + InstCombine's `select(fcmp olt x, lo, lo, x)` -> `maxnum` idiom. It happens
//     WITHOUT -ffast-math, because x86 maxsd's NaN/-0.0 tie-break is chosen to match the C
//     semantics of the source `if` exactly. There is no branch left to hint, at any -O level
//     that vectorizes, and NONE even with vectorization disabled (still a scalar vmaxsd).
//
//   * CONDITIONAL-STORE form (`if (c) w = lo;` -- DaCe's clamp_map): vectorized to
//     `vcmpgtpd -> k1` + a masked store `vmovupd %zmm,(%rax){%k1}`. Also branchless. GCC adds a
//     `kortestb %k1,%k1 / je` store-skip guard, but that is GCC's own per-VECTOR (8-lane)
//     invention, not the source's per-element `if`, and the hint does not steer it: the vector
//     body is BYTE-IDENTICAL with and without [[unlikely]].
//
// So [[unlikely]] can only reach the SCALAR REMAINDER (< 8 iterations of an N-element loop --
// O(1) work in an O(N) loop). Measured effect there: gcc moves the cold block out of line;
// clang PESSIMIZES, replacing its branchless epilogue vmaxsd with vucomisd+ja. Both are noise
// at the loop level. See the README table for numbers.
//
// The third pair (`rec_*`) is the ONE shape where the hint does something real, and it is
// included to show why it is still not this lever: with the clamp on a SERIAL RECURRENCE's
// critical path, clang's hinted form drops vmaxsd out of the dependency chain and speculates
// past the compare instead (vfmadd+vmaxsd ~6.5 cyc/iter -> vfmadd ~4 cyc/iter). That is a
// dependency-chain LATENCY effect, not a branch-probability one -- which is why it gets LARGER
// at a 50% hit rate, the opposite of what a branch hint would do. It is the already-rejected
// branch-vs-branchless axis (README "Why most candidates are missing"), reached from the other
// side, and gcc does not reproduce it (1.00x). DaCe emits clamps inside data-parallel maps,
// which vectorize, so this shape does not arise from a clamp in practice.
//
// Both hit rates below come from the SAME binary and the SAME forms -- only the input
// distribution differs, so 1% vs 50% is a pure branch-predictability contrast. The actual
// measured hit rate is printed, not just the intended one.
//
// !! nvc++ 26.3 MISCOMPILES [[unlikely]] -- this file reproduces it, and CHECK_BITSAME catches
// !! it (exit 2, "MISMATCH (rec_unlikely)"). That is the harness working as designed, not a
// !! defect in this file. C++20 [dcl.attr.likelihood] makes [[likely]]/[[unlikely]] a pure hint
// !! with NO semantic effect, so a compiler that changes results is emitting wrong code.
// !! Reduced: a serial recurrence `b[i] = (b[i-1]*0.5 + a[i] < lo) ? lo : b[i-1]*0.5 + a[i]`
// !! with a[i] == 1.0 and lo == 0.0 -- the clamp condition is ALWAYS false, yet from i=2 on
// !! nvc++ stores `lo` unconditionally (16382/16384 elements wrong). Bisected on this box:
// !!   * deleting the attribute is the ONLY change needed to make it correct
// !!   * -O0/-O1 correct; -O2/-O3/-O4 wrong;  -O3 -Mnovect correct  => it is in the vectorizer
// !!   * [[likely]] on the else arm instead is correct
// !! gcc 15.2 and clang 21.1 agree bit-for-bit on the same source (icpx not installed here;
// !! assumed correct, being clang-based -- unverified).
// !! This is the decisive argument against emitting the attribute from a code generator: on one
// !! of the four target compilers it is not a no-op, it is a wrong-code bug at the -O level DaCe
// !! actually ships (-O3).
//
// Build (needs C++20 for [[unlikely]]; DaCe's compiler.cpp_standard defaults to "20"):
//   g++     -std=c++20 -O3 -march=native -ffast-math -o /tmp/cbh_gcc   clamp_branch_hint.cpp && /tmp/cbh_gcc
//   clang++ -std=c++20 -O3 -march=native -ffast-math -o /tmp/cbh_clang clamp_branch_hint.cpp && /tmp/cbh_clang
//   # nvc++ spells fast-math -fast, and EXITS 2 on the rec_* pair -- see the miscompile note above:
//   nvc++   -std=c++20 -O3 -tp=native -fast -o /tmp/cbh_nvc clamp_branch_hint.cpp && /tmp/cbh_nvc
#include "bench.h"

static const int N = 1 << 14;  // 128 KiB of double -- fits L2, so this is not a bandwidth test

// ---------------------------------------------------------------- value-select (DaCe clamp_ifelse)
// `plain` and `twin` are byte-identical: twin-vs-plain is the intra-binary noise+layout floor.
#define IE_BODY                                                                                                        \
    for (int i = 0; i < N; ++i) {                                                                                      \
        bool __tmp0 = (a[i] < lo);                                                                                     \
        if (__tmp0) {                                                                                                  \
            b[i] = lo;                                                                                                 \
        } else {                                                                                                       \
            b[i] = a[i];                                                                                               \
        }                                                                                                              \
    }

__attribute__((noinline)) void ie_plain(const double *__restrict__ a, double *__restrict__ b, double lo) { IE_BODY }
__attribute__((noinline)) void ie_twin(const double *__restrict__ a, double *__restrict__ b, double lo) { IE_BODY }

__attribute__((noinline)) void ie_unlikely(const double *__restrict__ a, double *__restrict__ b, double lo) {
    for (int i = 0; i < N; ++i) {
        bool __tmp0 = (a[i] < lo);
        if (__tmp0) [[unlikely]] {
            b[i] = lo;
        } else [[likely]] {
            b[i] = a[i];
        }
    }
}

// ---------------------------------------------------------------- conditional store (DaCe clamp_map)
// Out-of-place so that re-running does not change the hit rate (an in-place clamp is idempotent
// after the first rep, which would silently make every later rep a 0% run).
#define CS_BODY                                                                                                        \
    for (int i = 0; i < N; ++i) {                                                                                      \
        bool __tmp0 = (a[i] < lo);                                                                                     \
        if (__tmp0) {                                                                                                  \
            b[i] = lo;                                                                                                 \
        }                                                                                                              \
    }

__attribute__((noinline)) void cs_plain(const double *__restrict__ a, double *__restrict__ b, double lo) { CS_BODY }
__attribute__((noinline)) void cs_twin(const double *__restrict__ a, double *__restrict__ b, double lo) { CS_BODY }

__attribute__((noinline)) void cs_unlikely(const double *__restrict__ a, double *__restrict__ b, double lo) {
    for (int i = 0; i < N; ++i) {
        bool __tmp0 = (a[i] < lo);
        if (__tmp0) [[unlikely]] {
            b[i] = lo;
        }
    }
}

// ---------------------------------------------------------------- clamp on a serial recurrence
// b[i] depends on b[i-1], so this cannot vectorize for a REAL reason (not a pragma), and the
// clamp lands on the critical path. The one shape where the hint changes the hot loop.
#define REC_BODY                                                                                                       \
    b[0] = a[0];                                                                                                       \
    for (int i = 1; i < N; ++i) {                                                                                      \
        double x = b[i - 1] * 0.5 + a[i];                                                                              \
        bool __tmp0 = (x < lo);                                                                                        \
        if (__tmp0) {                                                                                                  \
            b[i] = lo;                                                                                                 \
        } else {                                                                                                       \
            b[i] = x;                                                                                                  \
        }                                                                                                              \
    }

__attribute__((noinline)) void rec_plain(const double *__restrict__ a, double *__restrict__ b, double lo) { REC_BODY }
__attribute__((noinline)) void rec_twin(const double *__restrict__ a, double *__restrict__ b, double lo) { REC_BODY }

__attribute__((noinline)) void rec_unlikely(const double *__restrict__ a, double *__restrict__ b, double lo) {
    b[0] = a[0];
    for (int i = 1; i < N; ++i) {
        double x = b[i - 1] * 0.5 + a[i];
        bool __tmp0 = (x < lo);
        if (__tmp0) [[unlikely]] {
            b[i] = lo;
        } else [[likely]] {
            b[i] = x;
        }
    }
}

// ============================================================================================
typedef void (*fn_t)(const double *__restrict__, double *__restrict__, double);

// Deterministic (fixed seed, xorshift64): `frac` of the elements are negative, so with lo=0.0
// the clamp condition is true for ~frac of them. Same code, different data -- that is the
// whole 1%-vs-50% contrast.
static std::vector<double> make_input(double frac) {
    std::vector<double> a(N);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < N; ++i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        double u = double(s >> 11) * (1.0 / 9007199254740992.0);  // [0,1)
        a[i] = (u < frac) ? -1.0 - u : 1.0 + u;
    }
    return a;
}

struct Form {
    const char *name;
    fn_t fn;
};

// Count how often the branch condition is ACTUALLY true, for the shape being timed.
static double measured_rate(const std::vector<double> &a, double lo, bool recurrence) {
    long hits = 0;
    if (recurrence) {
        std::vector<double> b(N);
        b[0] = a[0];
        for (int i = 1; i < N; ++i) {
            double x = b[i - 1] * 0.5 + a[i];
            if (x < lo) {
                ++hits;
                b[i] = lo;
            } else {
                b[i] = x;
            }
        }
        return double(hits) / double(N - 1);
    }
    for (int i = 0; i < N; ++i)
        if (a[i] < lo) ++hits;
    return double(hits) / double(N);
}

static int run_case(const char *label, const Form *forms, int nforms, const std::vector<double> &a, bool recurrence) {
    const double lo = 0.0;

    // Bit-identical outputs, or the "lever" is a bug.
    std::vector<std::vector<double>> outs(nforms, std::vector<double>(N, 0.0));
    for (int f = 0; f < nforms; ++f)
        forms[f].fn(a.data(), outs[f].data(), lo);
    for (int f = 1; f < nforms; ++f)
        CHECK_BITSAME(outs[0].data(), outs[f].data(), N * sizeof(double), forms[f].name);

    std::printf("\n%s -- branch actually taken on %.2f%% of iterations (N=%d)\n", label,
                measured_rate(a, lo, recurrence) * 100.0, N);

    std::vector<double> b(N, 0.0);
    for (int f = 0; f < nforms; ++f)  // warm caches + predictor for EVERY form before timing ANY
        for (int w = 0; w < 50; ++w) forms[f].fn(a.data(), b.data(), lo);

    Stat base{};
    for (int f = 0; f < nforms; ++f) {
        fn_t fn = forms[f].fn;
        const double *ap = a.data();
        double *bp = b.data();
        Stat s = bench([&]() { for (int k = 0; k < 32; ++k) fn(ap, bp, lo); }, 100);
        if (f == 0) base = s;
        row(forms[f].name, s, base.median);
    }
    return 0;
}

int main() {
    const Form ie[] = {{"ie_plain", ie_plain}, {"ie_twin (layout floor)", ie_twin}, {"ie_unlikely", ie_unlikely}};
    const Form cs[] = {{"cs_plain", cs_plain}, {"cs_twin (layout floor)", cs_twin}, {"cs_unlikely", cs_unlikely}};
    const Form rec[] = {{"rec_plain", rec_plain}, {"rec_twin (layout floor)", rec_twin}, {"rec_unlikely", rec_unlikely}};

    std::printf("clamp branch hint -- [[unlikely]] vs nothing. median of n=100, ratio vs the un-hinted form.\n");
    std::printf("reportable threshold ~1.2x; twin-vs-plain is the noise+layout floor.\n");

    for (double frac : {0.01, 0.50}) {
        std::vector<double> a = make_input(frac);
        std::printf("\n================ input: %.0f%% of elements below the clamp ================\n", frac * 100.0);
        if (run_case("[value-select]  DaCe clamp_ifelse -- if-converted to vmaxpd, no branch", ie, 3, a, false)) return 2;
        if (run_case("[cond. store]   DaCe clamp_map -- vcmpgtpd + masked store, no branch", cs, 3, a, false)) return 2;
        if (run_case("[recurrence]    clamp on a serial critical path -- branch survives", rec, 3, a, true)) return 2;
    }
    return 0;
}
