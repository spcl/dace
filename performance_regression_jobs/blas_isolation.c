/* Layer-isolation for the OpenBLAS thread-scaling collapse (both pthreads and
 * openmp spack builds collapse ~linearly in thread count on Grace, while DaCe's
 * own OpenMP regions scale fine on the same nodes).  Pure C, no Python/DaCe:
 *   mode "gemm":  cblas_dgemm 2048^3, threads swept by the caller via env.
 *   mode "triad": plain '#pragma omp parallel for' streaming triad as a
 *                 control -- if THIS scales but gemm does not, the problem is
 *                 inside OpenBLAS, not in libgomp/binding/cgroup.
 * Build twice, once against each spack OpenBLAS (pthreads / openmp).       */
#include <cblas.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

extern int openblas_get_num_threads(void);
extern int openblas_get_parallel(void);

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "gemm";
    const int n = argc > 2 ? atoi(argv[2]) : 2048;
    const int reps = 3;

    if (strcmp(mode, "gemm") == 0) {
        double *A = malloc((size_t)n * n * sizeof(double));
        double *B = malloc((size_t)n * n * sizeof(double));
        double *C = malloc((size_t)n * n * sizeof(double));
        for (long i = 0; i < (long)n * n; i++) { A[i] = 0.5 + i % 7; B[i] = 0.25 + i % 5; C[i] = 0; }
        /* warmup */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now_s();
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("GEMM n=%d openblas_threads=%d parallel_mode=%d best_ms=%.2f gflops=%.1f\n", n,
               openblas_get_num_threads(), openblas_get_parallel(), best * 1e3,
               2.0 * n * n * (double)n / best / 1e9);
        /* keep C observable so the call cannot be dead-code-eliminated */
        fprintf(stderr, "checksum %.3e\n", C[0] + C[(size_t)n * n - 1]);
    } else { /* triad control: pure libgomp, no BLAS */
        const long len = 1L << 27; /* 128M doubles = 1 GiB per array x3 */
        double *a = malloc(len * sizeof(double)), *b = malloc(len * sizeof(double)),
               *c = malloc(len * sizeof(double));
#pragma omp parallel for schedule(static) /* first-touch init on the owning thread */
        for (long i = 0; i < len; i++) { b[i] = 1.5; c[i] = 2.5; a[i] = 0; }
        double best = 1e30;
        for (int r = 0; r < reps + 1; r++) { /* first rep is warmup */
            double t0 = now_s();
#pragma omp parallel for schedule(static)
            for (long i = 0; i < len; i++) a[i] = b[i] + 3.0 * c[i];
            double dt = now_s() - t0;
            if (r > 0 && dt < best) best = dt;
        }
        printf("TRIAD len=%ld omp_max_threads=%d best_ms=%.2f gbps=%.1f\n", len, omp_get_max_threads(),
               best * 1e3, 3.0 * len * sizeof(double) / best / 1e9);
        fprintf(stderr, "checksum %.3e\n", a[0] + a[len - 1]);
    }
    return 0;
}
