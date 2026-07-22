// heat_3d "readable-codegen 1.30x regression" -- REFUTED as a source-form lever; it is the
// LAYOUT CONFOUND (CODEGEN_STYLE_PERFORMANCE.md section 3). NOT a leverN: kept as a documented
// negative result, in the spirit of divisor_spellings.cpp and lever2's "REFUTED here" row.
//
// heat_3d runs ~1.30x slower under the experimental readable CPU codegen than legacy at
// single core. This file carries the EXACT fused first-sweep inner-loop body of DaCe's
// heat_3d, written verbatim in BOTH the legacy and the experimental-readable source forms,
// plus a byte-identical TWIN of each and one hybrid per source-form axis -- ALL IN ONE BINARY,
// single-thread. One binary == one code layout, so this removes the cross-binary/cross-TU
// placement draw that the real measurement (two separately compiled .cpp -> two layout samples)
// cannot. Any surviving delta would be the source form; there is none.
//
// The four source-form axes that differ legacy vs experimental (each flipped one at a time):
//   A  const double t[1] = {expr}  (exp)  vs  double t[1]; t[0]=expr;  (legacy)     -> sweep_exp_mutabletemp
//   B  inlined single-expression   (exp)  vs  { double __in=..; __out=..; t[0]=__out; } brace+named-locals (legacy)
//   C  A[A_idx(i,j,k,N)] index fn   (exp)  vs  A[N*j + i*ipow(N,2) + k] inline flat index (legacy) -> sweep_exp_inlineidx
//   E  double* __restrict__ heap temps (exp) vs plain double* (legacy)              -> sweep_exp_norestrict
//
// WHY IT COLLAPSES (the section-6 criterion "a knob the compiler can re-derive is not a knob"):
// every axis is erased before the vectorizer/scheduler sees it. SROA+mem2reg promote the len-1
// `[1]` temp arrays to SSA registers (kills A and B); the DACE_HDFI=`inline constexpr` A_idx
// helper is fully inlined and GVN commons the arithmetic (kills C, 0 `bl` to it in either .s);
// A/B are already `__restrict__` in BOTH signatures so E only toggles alias facts on write-only
// output temps (kills E). Confirmed by objdump: the two generated compute TUs have IDENTICAL
// FP instruction selection (g++ fadd 17/17 fmul 4/4 fsub 6/6 fmadd 6/6, NO stencil SVE; clang
// fadd 22/22 fmul 9/9 fsub 9/9 + identical tail-memcpy SVE ld1d 9/9) -- same instructions,
// different block order.
//
// MEASURED (Neoverse-V2 Grace, g++16.1 / clang22.1, -O3 -march=native, single core):
//   Controlled (interleaved round-robin + ONE shared dst buffer -> removes DVFS-drift and
//   destination-placement bias): every form, both endpoints AND all axis flips, lands in
//     g++ 1.00-1.08x   clang 0.98-1.02x   -- twins move as much as variants; nothing clears 1.2x.
//   Layout floor (layout_probe.cpp: 5 BYTE-IDENTICAL twins, sequential + per-twin buffers):
//     g++ spread 1.28x (default) -> 1.17x (-falign-functions=64 -falign-loops=32); clang 1.03x.
//   i.e. identical code varies by ~1.3x from placement alone -- the reported 1.30x is INSIDE it,
//   and its sign is not even fixed (uncontrolled sequential timing made LEGACY look 1.5x slower
//   on g++ and split byte-identical clang twins by 2.7x). No semantic cause survives.
//
// No codegen_params flag is warranted: there is no fast form for the generator to emit.
#include "bench.h"

// ---- index helpers (both spellings compute the same flat offset) ----
static inline long long A_idx(long long d0, long long d1, long long d2, long long N) {
    return N * d1 + d0 * (N * N) + d2;
}
static inline long long O_idx(long long d0, long long d1, long long d2, long long N) {
    return d0 * ((N - 2) * (N - 2)) + d1 * (N - 2) + d2;
}
#define IDX_FN(i, j, k)  A_idx((i), (j), (k), N)
#define IDX_INL(i, j, k) (N * (long long)(j) + (long long)(i) * ((long long)N * N) + (long long)(k))
#define O_INL(i, j, k)   ((long long)(i) * ((long long)(N - 2) * (N - 2)) + (long long)(j) * (N - 2) + (long long)(k))

// ======================================================================
// V1  LEGACY: mutable double[1] temps declared up front, per-tasklet brace
//     scope with copy-in __in/__out named locals, INLINE flat index, no restrict.
// ======================================================================
__attribute__((noinline))
void sweep_legacy(double *A, double *B, double *tmp10, double *tmp13, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N - 2; ++i)
      for (int j = 0; j < N - 2; ++j)
        for (int k = 0; k < N - 2; ++k) {
            double t5[1], am2_1[1], a2_1[1], t6[1], t2[1], am2_0[1], a2_0[1], t3[1];
            double t0[1], am2[1], a2[1], t1[1], t4[1], t7[1], Bv[1], B0[1];
            { double __in2 = A[IDX_INL(i+1,j+1,k+1)]; double __out; __out = (2.0*__in2); t5[0]=__out; }
            { double __in2 = t5[0]; double __in1 = A[IDX_INL(i+1,j+1,k+2)]; double __out; __out=(__in1-__in2); am2_1[0]=__out; }
            { double __in1 = am2_1[0]; double __in2 = A[IDX_INL(i+1,j+1,k)]; double __out; __out=(__in1+__in2); a2_1[0]=__out; }
            { double __in2 = a2_1[0]; double __out; __out=(0.125*__in2); t6[0]=__out; }
            { double __in2 = A[IDX_INL(i+1,j+1,k+1)]; double __out; __out=(2.0*__in2); t2[0]=__out; }
            { double __in2 = t2[0]; double __in1 = A[IDX_INL(i+1,j+2,k+1)]; double __out; __out=(__in1-__in2); am2_0[0]=__out; }
            { double __in1 = am2_0[0]; double __in2 = A[IDX_INL(i+1,j,k+1)]; double __out; __out=(__in1+__in2); a2_0[0]=__out; }
            { double __in2 = a2_0[0]; double __out; __out=(0.125*__in2); t3[0]=__out; }
            { double __in2 = A[IDX_INL(i+1,j+1,k+1)]; double __out; __out=(2.0*__in2); t0[0]=__out; }
            { double __in2 = t0[0]; double __in1 = A[IDX_INL(i+2,j+1,k+1)]; double __out; __out=(__in1-__in2); am2[0]=__out; }
            { double __in1 = am2[0]; double __in2 = A[IDX_INL(i,j+1,k+1)]; double __out; __out=(__in1+__in2); a2[0]=__out; }
            { double __in2 = a2[0]; double __out; __out=(0.125*__in2); t1[0]=__out; }
            { double __in1 = t1[0]; double __in2 = t3[0]; double __out; __out=(__in1+__in2); t4[0]=__out; }
            { double __in1 = t4[0]; double __in2 = t6[0]; double __out; __out=(__in1+__in2); t7[0]=__out; }
            { double __in1 = t7[0]; double __in2 = A[IDX_INL(i+1,j+1,k+1)]; double __out; __out=(__in1+__in2); Bv[0]=__out; }
            B0[0] = Bv[0];
            { double __in2 = Bv[0]; double __out; __out=(2.0*__in2); tmp10[O_INL(i,j,k)]=__out; }
            B0[0] = B[IDX_INL(i+1,j+1,k+1)];
            { double __in2 = B0[0]; double __out; __out=(2.0*__in2); tmp13[O_INL(i,j,k)]=__out; }
        }
}
// twin of V1 (byte-identical body, different symbol) -> layout/noise floor
__attribute__((noinline))
void sweep_legacy_twin(double *A, double *B, double *tmp10, double *tmp13, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N - 2; ++i)
      for (int j = 0; j < N - 2; ++j)
        for (int k = 0; k < N - 2; ++k) {
            double t5[1], am2_1[1], a2_1[1], t6[1], t2[1], am2_0[1], a2_0[1], t3[1];
            double t0[1], am2[1], a2[1], t1[1], t4[1], t7[1], Bv[1], B0[1];
            { double __in2 = A[IDX_INL(i+1,j+1,k+1)]; double __out; __out = (2.0*__in2); t5[0]=__out; }
            { double __in2 = t5[0]; double __in1 = A[IDX_INL(i+1,j+1,k+2)]; double __out; __out=(__in1-__in2); am2_1[0]=__out; }
            { double __in1 = am2_1[0]; double __in2 = A[IDX_INL(i+1,j+1,k)]; double __out; __out=(__in1+__in2); a2_1[0]=__out; }
            { double __in2 = a2_1[0]; double __out; __out=(0.125*__in2); t6[0]=__out; }
            { double __in2 = A[IDX_INL(i+1,j+1,k+1)]; double __out; __out=(2.0*__in2); t2[0]=__out; }
            { double __in2 = t2[0]; double __in1 = A[IDX_INL(i+1,j+2,k+1)]; double __out; __out=(__in1-__in2); am2_0[0]=__out; }
            { double __in1 = am2_0[0]; double __in2 = A[IDX_INL(i+1,j,k+1)]; double __out; __out=(__in1+__in2); a2_0[0]=__out; }
            { double __in2 = a2_0[0]; double __out; __out=(0.125*__in2); t3[0]=__out; }
            { double __in2 = A[IDX_INL(i+1,j+1,k+1)]; double __out; __out=(2.0*__in2); t0[0]=__out; }
            { double __in2 = t0[0]; double __in1 = A[IDX_INL(i+2,j+1,k+1)]; double __out; __out=(__in1-__in2); am2[0]=__out; }
            { double __in1 = am2[0]; double __in2 = A[IDX_INL(i,j+1,k+1)]; double __out; __out=(__in1+__in2); a2[0]=__out; }
            { double __in2 = a2[0]; double __out; __out=(0.125*__in2); t1[0]=__out; }
            { double __in1 = t1[0]; double __in2 = t3[0]; double __out; __out=(__in1+__in2); t4[0]=__out; }
            { double __in1 = t4[0]; double __in2 = t6[0]; double __out; __out=(__in1+__in2); t7[0]=__out; }
            { double __in1 = t7[0]; double __in2 = A[IDX_INL(i+1,j+1,k+1)]; double __out; __out=(__in1+__in2); Bv[0]=__out; }
            B0[0] = Bv[0];
            { double __in2 = Bv[0]; double __out; __out=(2.0*__in2); tmp10[O_INL(i,j,k)]=__out; }
            B0[0] = B[IDX_INL(i+1,j+1,k+1)];
            { double __in2 = B0[0]; double __out; __out=(2.0*__in2); tmp13[O_INL(i,j,k)]=__out; }
        }
}

// ======================================================================
// V3  EXPERIMENTAL-READABLE: const double[1]={expr} temps initialized inline,
//     no brace/named-locals, INDEX-FUNCTION loads, __restrict__ heap temps.
// ======================================================================
__attribute__((noinline))
void sweep_exp(double *__restrict__ A, double *__restrict__ B,
               double *__restrict__ tmp10, double *__restrict__ tmp13, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N - 2; ++i)
      for (int j = 0; j < N - 2; ++j)
        for (int k = 0; k < N - 2; ++k) {
            double B0[1];
            const double t5[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2_1[1] = {(double)((A[IDX_FN(i+1,j+1,k+2)] - t5[0]))};
            const double a2_1[1]  = {(double)((am2_1[0] + A[IDX_FN(i+1,j+1,k)]))};
            const double t6[1]    = {(double)((0.125 * a2_1[0]))};
            const double t2[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2_0[1] = {(double)((A[IDX_FN(i+1,j+2,k+1)] - t2[0]))};
            const double a2_0[1]  = {(double)((am2_0[0] + A[IDX_FN(i+1,j,k+1)]))};
            const double t3[1]    = {(double)((0.125 * a2_0[0]))};
            const double t0[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2[1]   = {(double)((A[IDX_FN(i+2,j+1,k+1)] - t0[0]))};
            const double a2[1]    = {(double)((am2[0] + A[IDX_FN(i,j+1,k+1)]))};
            const double t1[1]    = {(double)((0.125 * a2[0]))};
            const double t4[1]    = {(double)((t1[0] + t3[0]))};
            const double t7[1]    = {(double)((t4[0] + t6[0]))};
            const double Bv[1]    = {(double)((t7[0] + A[IDX_FN(i+1,j+1,k+1)]))};
            B0[0] = Bv[0];
            tmp10[O_idx(i,j,k,N)] = (2.0 * Bv[0]);
            B0[0] = B[IDX_FN(i+1,j+1,k+1)];
            tmp13[O_idx(i,j,k,N)] = (2.0 * B0[0]);
        }
}
// twin of V3
__attribute__((noinline))
void sweep_exp_twin(double *__restrict__ A, double *__restrict__ B,
                    double *__restrict__ tmp10, double *__restrict__ tmp13, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N - 2; ++i)
      for (int j = 0; j < N - 2; ++j)
        for (int k = 0; k < N - 2; ++k) {
            double B0[1];
            const double t5[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2_1[1] = {(double)((A[IDX_FN(i+1,j+1,k+2)] - t5[0]))};
            const double a2_1[1]  = {(double)((am2_1[0] + A[IDX_FN(i+1,j+1,k)]))};
            const double t6[1]    = {(double)((0.125 * a2_1[0]))};
            const double t2[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2_0[1] = {(double)((A[IDX_FN(i+1,j+2,k+1)] - t2[0]))};
            const double a2_0[1]  = {(double)((am2_0[0] + A[IDX_FN(i+1,j,k+1)]))};
            const double t3[1]    = {(double)((0.125 * a2_0[0]))};
            const double t0[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2[1]   = {(double)((A[IDX_FN(i+2,j+1,k+1)] - t0[0]))};
            const double a2[1]    = {(double)((am2[0] + A[IDX_FN(i,j+1,k+1)]))};
            const double t1[1]    = {(double)((0.125 * a2[0]))};
            const double t4[1]    = {(double)((t1[0] + t3[0]))};
            const double t7[1]    = {(double)((t4[0] + t6[0]))};
            const double Bv[1]    = {(double)((t7[0] + A[IDX_FN(i+1,j+1,k+1)]))};
            B0[0] = Bv[0];
            tmp10[O_idx(i,j,k,N)] = (2.0 * Bv[0]);
            B0[0] = B[IDX_FN(i+1,j+1,k+1)];
            tmp13[O_idx(i,j,k,N)] = (2.0 * B0[0]);
        }
}

// ======================================================================
// AXIS HYBRIDS (flip exactly one axis away from V3/exp)
// ======================================================================
// Axis C: exp but INLINE index (isolates index-function vs inline flat index)
__attribute__((noinline))
void sweep_exp_inlineidx(double *__restrict__ A, double *__restrict__ B,
                         double *__restrict__ tmp10, double *__restrict__ tmp13, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N - 2; ++i)
      for (int j = 0; j < N - 2; ++j)
        for (int k = 0; k < N - 2; ++k) {
            double B0[1];
            const double t5[1]    = {(double)((2.0 * A[IDX_INL(i+1,j+1,k+1)]))};
            const double am2_1[1] = {(double)((A[IDX_INL(i+1,j+1,k+2)] - t5[0]))};
            const double a2_1[1]  = {(double)((am2_1[0] + A[IDX_INL(i+1,j+1,k)]))};
            const double t6[1]    = {(double)((0.125 * a2_1[0]))};
            const double t2[1]    = {(double)((2.0 * A[IDX_INL(i+1,j+1,k+1)]))};
            const double am2_0[1] = {(double)((A[IDX_INL(i+1,j+2,k+1)] - t2[0]))};
            const double a2_0[1]  = {(double)((am2_0[0] + A[IDX_INL(i+1,j,k+1)]))};
            const double t3[1]    = {(double)((0.125 * a2_0[0]))};
            const double t0[1]    = {(double)((2.0 * A[IDX_INL(i+1,j+1,k+1)]))};
            const double am2[1]   = {(double)((A[IDX_INL(i+2,j+1,k+1)] - t0[0]))};
            const double a2[1]    = {(double)((am2[0] + A[IDX_INL(i,j+1,k+1)]))};
            const double t1[1]    = {(double)((0.125 * a2[0]))};
            const double t4[1]    = {(double)((t1[0] + t3[0]))};
            const double t7[1]    = {(double)((t4[0] + t6[0]))};
            const double Bv[1]    = {(double)((t7[0] + A[IDX_INL(i+1,j+1,k+1)]))};
            B0[0] = Bv[0];
            tmp10[O_INL(i,j,k)] = (2.0 * Bv[0]);
            B0[0] = B[IDX_INL(i+1,j+1,k+1)];
            tmp13[O_INL(i,j,k)] = (2.0 * B0[0]);
        }
}
// Axis A: exp but MUTABLE temp + separate store (isolates const-init {expr} vs mutable[1]+assign)
__attribute__((noinline))
void sweep_exp_mutabletemp(double *__restrict__ A, double *__restrict__ B,
                           double *__restrict__ tmp10, double *__restrict__ tmp13, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N - 2; ++i)
      for (int j = 0; j < N - 2; ++j)
        for (int k = 0; k < N - 2; ++k) {
            double t5[1], am2_1[1], a2_1[1], t6[1], t2[1], am2_0[1], a2_0[1], t3[1];
            double t0[1], am2[1], a2[1], t1[1], t4[1], t7[1], Bv[1], B0[1];
            t5[0]    = (2.0 * A[IDX_FN(i+1,j+1,k+1)]);
            am2_1[0] = (A[IDX_FN(i+1,j+1,k+2)] - t5[0]);
            a2_1[0]  = (am2_1[0] + A[IDX_FN(i+1,j+1,k)]);
            t6[0]    = (0.125 * a2_1[0]);
            t2[0]    = (2.0 * A[IDX_FN(i+1,j+1,k+1)]);
            am2_0[0] = (A[IDX_FN(i+1,j+2,k+1)] - t2[0]);
            a2_0[0]  = (am2_0[0] + A[IDX_FN(i+1,j,k+1)]);
            t3[0]    = (0.125 * a2_0[0]);
            t0[0]    = (2.0 * A[IDX_FN(i+1,j+1,k+1)]);
            am2[0]   = (A[IDX_FN(i+2,j+1,k+1)] - t0[0]);
            a2[0]    = (am2[0] + A[IDX_FN(i,j+1,k+1)]);
            t1[0]    = (0.125 * a2[0]);
            t4[0]    = (t1[0] + t3[0]);
            t7[0]    = (t4[0] + t6[0]);
            Bv[0]    = (t7[0] + A[IDX_FN(i+1,j+1,k+1)]);
            B0[0] = Bv[0];
            tmp10[O_idx(i,j,k,N)] = (2.0 * Bv[0]);
            B0[0] = B[IDX_FN(i+1,j+1,k+1)];
            tmp13[O_idx(i,j,k,N)] = (2.0 * B0[0]);
        }
}
// Axis E: exp but NO __restrict__ on any pointer (isolates restrict on heap temps)
__attribute__((noinline))
void sweep_exp_norestrict(double *A, double *B, double *tmp10, double *tmp13, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N - 2; ++i)
      for (int j = 0; j < N - 2; ++j)
        for (int k = 0; k < N - 2; ++k) {
            double B0[1];
            const double t5[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2_1[1] = {(double)((A[IDX_FN(i+1,j+1,k+2)] - t5[0]))};
            const double a2_1[1]  = {(double)((am2_1[0] + A[IDX_FN(i+1,j+1,k)]))};
            const double t6[1]    = {(double)((0.125 * a2_1[0]))};
            const double t2[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2_0[1] = {(double)((A[IDX_FN(i+1,j+2,k+1)] - t2[0]))};
            const double a2_0[1]  = {(double)((am2_0[0] + A[IDX_FN(i+1,j,k+1)]))};
            const double t3[1]    = {(double)((0.125 * a2_0[0]))};
            const double t0[1]    = {(double)((2.0 * A[IDX_FN(i+1,j+1,k+1)]))};
            const double am2[1]   = {(double)((A[IDX_FN(i+2,j+1,k+1)] - t0[0]))};
            const double a2[1]    = {(double)((am2[0] + A[IDX_FN(i,j+1,k+1)]))};
            const double t1[1]    = {(double)((0.125 * a2[0]))};
            const double t4[1]    = {(double)((t1[0] + t3[0]))};
            const double t7[1]    = {(double)((t4[0] + t6[0]))};
            const double Bv[1]    = {(double)((t7[0] + A[IDX_FN(i+1,j+1,k+1)]))};
            B0[0] = Bv[0];
            tmp10[O_idx(i,j,k,N)] = (2.0 * Bv[0]);
            B0[0] = B[IDX_FN(i+1,j+1,k+1)];
            tmp13[O_idx(i,j,k,N)] = (2.0 * B0[0]);
        }
}

static double checksum(const double *p, long long n) {
    double s = 0;
    for (long long i = 0; i < n; ++i) s += p[i];
    return s;
}

typedef void (*sweep_fn)(double *, double *, double *, double *, int);

int main() {
    const int N = 120;
    const long long NN = (long long)N * N * N;
    const long long OO = (long long)(N - 2) * (N - 2) * (N - 2);
    const int ROUNDS = 41;

    std::vector<double> A(NN), B(NN);
    for (long long i = 0; i < NN; ++i) {
        A[i] = double((i % 97) + 1) * 0.5;
        B[i] = double((i % 89) + 1) * 0.25;
    }
    // ONE shared pair of output buffers for every form -> removes the destination-placement
    // (conflict-miss / TLB) confound. Reference is computed once and saved.
    std::vector<double> o(OO), d(OO), oref(OO), dref(OO);

    struct Form { const char *name; sweep_fn fn; };
    Form forms[] = {
        {"sweep_exp        [EXP]",   sweep_exp},
        {"sweep_exp_twin   [FLOOR]", sweep_exp_twin},
        {"sweep_legacy     [LEG]",   sweep_legacy},
        {"sweep_legacy_twin[FLOOR]", sweep_legacy_twin},
        {"exp_inlineidx (axisC)",    sweep_exp_inlineidx},
        {"exp_mutabletemp(axisA)",   sweep_exp_mutabletemp},
        {"exp_norestrict (axisE)",   sweep_exp_norestrict},
    };
    const int NF = sizeof(forms) / sizeof(forms[0]);

    // reference + warmup (spins up DVFS before any timing)
    sweep_legacy(A.data(), B.data(), oref.data(), dref.data(), N);
    for (int w = 0; w < 5; ++w)
        for (int f = 0; f < NF; ++f) forms[f].fn(A.data(), B.data(), o.data(), d.data(), N);

    // INTERLEAVED round-robin timing: each round times every form once, so DVFS/thermal
    // drift is shared equally across forms. Median over rounds per form.
    std::vector<std::vector<double>> t(NF);
    for (int r = 0; r < ROUNDS; ++r)
        for (int f = 0; f < NF; ++f) {
            double a = now_sec();
            forms[f].fn(A.data(), B.data(), o.data(), d.data(), N);
            double b = now_sec();
            t[f].push_back(b - a);
            if (r == 0) {  // bit-identical check on first pass
                CHECK_BITSAME(oref.data(), o.data(), OO*8, forms[f].name);
                CHECK_BITSAME(dref.data(), d.data(), OO*8, forms[f].name);
            }
        }

    Stat st[16];
    for (int f = 0; f < NF; ++f) {
        std::sort(t[f].begin(), t[f].end());
        st[f].median = t[f][t[f].size()/2];
        st[f].min = t[f].front();
        st[f].p25 = t[f][t[f].size()/4];
        st[f].p75 = t[f][(3*t[f].size())/4];
    }
    double base = st[0].median;  // sweep_exp

    std::printf("heat_3d source-form bisection  (N=%d, interior=%lld, rounds=%d, interleaved, single-thread)\n",
                N, OO, ROUNDS);
    std::printf(" baseline = sweep_exp (the experimental form); shared o/d buffers for all forms\n");
    for (int f = 0; f < NF; ++f) row(forms[f].name, st[f], base);
    std::printf(" checksum(oref)=%.6f  checksum(dref)=%.6f\n",
                checksum(oref.data(), OO), checksum(dref.data(), OO));
    return 0;
}
