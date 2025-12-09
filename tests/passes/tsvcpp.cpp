#include <chrono>
#include <cstdint>
#include <cmath>
using clock_highres = std::chrono::high_resolution_clock;

extern "C" {

// s000: a[i] = b[i] + 1
void s000_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 2 * iterations; ++nl) {
            for (int i = 0; i < len_1d; ++i) {
                a[i] = b[i] + 1.0;
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s111: a[i] = a[i-1] + b[i] for odd i
void s111_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 2 * iterations; ++nl) {
            for (int i = 1; i < len_1d; i += 2) {
                a[i] = a[i - 1] + b[i];
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s1111: a[2*i] = c[i]*b[i] + d[i]*b[i] + c[i]*c[i] + d[i]*b[i] + d[i]*c[i]
void s1111_run_timed(
    double * a,
    const double * b,
    const double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        const int half = len_1d / 2;
        for (int nl = 0; nl < 2 * iterations; ++nl) {
            for (int i = 0; i < half; ++i) {
                const double bi = b[i];
                const double ci = c[i];
                const double di = d[i];
                a[2 * i] =
                    ci * bi + di * bi + ci * ci + di * bi + di * ci;
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s112: reversed loop, a[i+1] = a[i] + b[i]
void s112_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 3 * iterations; ++nl) {
            for (int i = len_1d - 2; i >= 0; --i) {
                a[i + 1] = a[i] + b[i];
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s1112: reversed loop, a[i] = b[i] + 1
void s1112_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 3 * iterations; ++nl) {
            for (int i = len_1d - 1; i >= 0; --i) {
                a[i] = b[i] + 1.0;
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s113: a(i)=a(1) but no actual dependence cycle
void s113_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 4*iterations; nl++) {
            for (int i = 1; i < len_1d; i++) {
                a[i] = a[0] + b[i];
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s1113: one iteration dependency on a(LEN_1D/2) but still vectorizable
void s1113_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 2*iterations; nl++) {
            for (int i = 0; i < len_1d; i++) {
                a[i] = a[len_1d/2] + b[i];
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s114: transpose vectorization - Jump in data access
void s114_run_timed(
    double * aa,
    const double * bb,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 200*(iterations/len_2d); nl++) {
            for (int i = 0; i < len_2d; i++) {
                for (int j = 0; j < i; j++) {
                    aa[i*len_2d + j] = aa[j*len_2d + i] + bb[i*len_2d + j];
                }
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s115: triangular saxpy loop
void s115_run_timed(
    double * a,
    const double * aa,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 1000*(iterations/len_2d); nl++) {
            for (int j = 0; j < len_2d; j++) {
                for (int i = j+1; i < len_2d; i++) {
                    a[i] -= aa[j*len_2d + i] * a[j];
                }
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s1115: triangular saxpy loop variant
void s1115_run_timed(
    double * aa,
    const double * bb,
    const double * cc,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 100*(iterations/len_2d); nl++) {
            for (int i = 0; i < len_2d; i++) {
                for (int j = 0; j < len_2d; j++) {
                    aa[i*len_2d + j] = aa[i*len_2d + j] * cc[j*len_2d + i] + bb[i*len_2d + j];
                }
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}



// ------------------------------------------------------------
// s116: unrolled recurrence, stride-5
// ------------------------------------------------------------
void s116_run_timed(
    double * a,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations * 10; ++nl) {
            for (int i = 0; i < len_1d - 5; i += 5) {
                a[i]     = a[i + 1] * a[i];
                a[i + 1] = a[i + 2] * a[i + 1];
                a[i + 2] = a[i + 3] * a[i + 2];
                a[i + 3] = a[i + 4] * a[i + 3];
                a[i + 4] = a[i + 5] * a[i + 4];
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// ------------------------------------------------------------
// s118: potential dot-product-like recursion on a[], uses bb[j][i]
// ------------------------------------------------------------
void s118_run_timed(
    double * a,
    const double * bb,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        const int outer = 4 * (iterations);
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 1; i < len_2d; ++i) {
                for (int j = 0; j <= i - 1; ++j) {
                    const int idx_bb = j * len_2d + i;  // bb[j][i]
                    a[i] += bb[idx_bb] * a[i - j - 1];
                }
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// ------------------------------------------------------------
// s119: 2D recurrence over aa, reads bb
// aa[i][j] = aa[i-1][j-1] + bb[i][j]
// ------------------------------------------------------------
void s119_run_timed(
    double * aa,
    const double * bb,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        const int outer = 200 * (iterations / len_2d);
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 1; i < len_2d; ++i) {
                for (int j = 1; j < len_2d; ++j) {
                    const int idx_ij   = i * len_2d + j;         // [i][j]
                    const int idx_im1j = (i - 1) * len_2d + (j - 1); // [i-1][j-1]
                    aa[idx_ij] = aa[idx_im1j] + bb[idx_ij];
                }
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}


// ------------------------------------------------------------
// s121: j = i+1, a[i] = a[j] + b[i]
// ------------------------------------------------------------
void s121_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int j;
        for (int nl = 0; nl < 3 * iterations; ++nl) {
            for (int i = 0; i < len_1d - 1; ++i) {
                j = i + 1;
                a[i] = a[j] + b[i];
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// ------------------------------------------------------------
// s122: variable lower/upper bound + stride, reverse/jumped access
// a[i] += b[len_1d - k]
// ------------------------------------------------------------
void s122_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    const int n1,
    const int n3,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int j, k;
        for (int nl = 0; nl < iterations; ++nl) {
            j = 1;
            k = 0;
            for (int i = n1 - 1; i < len_1d; i += n3) {
                k += j;
                a[i] += b[len_1d - k];
            }
        }
    }
    auto t2 = clock::now();

    std::int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}


// s123: induction variable under an if
void s123_run_timed(
    double * a,
    const double * b,
    const double * c,
    const double * d,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int j;
        for (int nl = 0; nl < iterations; nl++) {
            j = -1;
            for (int i = 0; i < (len_1d/2); i++) {
                j++;
                a[j] = b[i] + d[i] * e[i];
                if (c[i] > 0.0) {
                    j++;
                    a[j] = c[i] + d[i] * e[i];
                }
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s124: induction variable under both sides of if (same value)
void s124_run_timed(
    double * a,
    const double * b,
    const double * c,
    const double * d,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int j;
        for (int nl = 0; nl < iterations; nl++) {
            j = -1;
            for (int i = 0; i < len_1d; i++) {
                if (b[i] > 0.0) {
                    j++;
                    a[j] = b[i] + d[i] * e[i];
                } else {
                    j++;
                    a[j] = c[i] + d[i] * e[i];
                }
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s125: induction variable in two loops; collapsing possible
void s125_run_timed(
    const double * aa,
    const double * bb,
    const double * cc,
    double * flat_2d_array,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int k;
        for (int nl = 0; nl < 100*(iterations/len_2d); nl++) {
            k = -1;
            for (int i = 0; i < len_2d; i++) {
                for (int j = 0; j < len_2d; j++) {
                    k++;
                    flat_2d_array[k] = aa[i*len_2d + j] + bb[i*len_2d + j] * cc[i*len_2d + j];
                }
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}


// s126: induction variable in two loops; recurrence in inner loop
void s126_run_timed(
    double * bb,
    const double * cc,
    const double * flat_2d_array,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int k;
        for (int nl = 0; nl < 10*(iterations/len_2d); nl++) {
            k = 1;
            for (int i = 0; i < len_2d; i++) {
                for (int j = 1; j < len_2d; j++) {
                    bb[j*len_2d + i] = bb[(j-1)*len_2d + i] + flat_2d_array[k-1] * cc[j*len_2d + i];
                    ++k;
                }
                ++k;
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s127: induction variable with multiple increments
void s127_run_timed(
    double * a,
    const double * b,
    const double * c,
    const double * d,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int j;
        for (int nl = 0; nl < 2*iterations; nl++) {
            j = -1;
            for (int i = 0; i < len_1d/2; i++) {
                j++;
                a[j] = b[i] + c[i] * d[i];
                j++;
                a[j] = b[i] + d[i] * e[i];
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s128: coupled induction variables - jump in data access
void s128_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int j, k;
        for (int nl = 0; nl < 2*iterations; nl++) {
            j = -1;
            for (int i = 0; i < len_1d/2; i++) {
                k = j + 1;
                a[i] = b[k] - d[i];
                j = k + 1;
                b[k] = a[i] + c[k];
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}

// s131: forward substitution
void s131_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int m = 1;
        for (int nl = 0; nl < 5*iterations; nl++) {
            for (int i = 0; i < len_1d - 1; i++) {
                a[i] = a[i + m] + b[i];
            }
        }
    }
    auto t2 = clock::now();
    std::int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    time_ns[0] = ns;
}


// ------------------------------------------------------------
// s132
// aa[j][i] = aa[k][i-1] + b[i] * c[1]
// j = 0, k = 1
// ------------------------------------------------------------
void s132_run_timed(
    double * aa,
    const double * b,
    const double * c,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    const int j = 0;
    const int k = 1;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 400 * iterations; ++nl) {
            for (int i = 1; i < len_2d; ++i) {
                aa[j * len_2d + i] =
                    aa[k * len_2d + (i - 1)] + b[i] * c[1];
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s141
// packed symmetric row: flat_2d_array[k]
// ------------------------------------------------------------
void s141_run_timed(
    const double * bb,
    double * flat_2d_array,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        const int outer = 200 * (iterations / len_2d);
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 0; i < len_2d; ++i) {
                int k = (i + 1) * (i) / 2 + (i);
                for (int j = i; j < len_2d; ++j) {
                    flat_2d_array[k] += bb[j * len_2d + i];
                    k += (j + 1);
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s151s + s151
// ------------------------------------------------------------
static inline void s151s_kernel(
    double * a,
    const double * b,
    const int len_1d,
    const int m
){
    for (int i = 0; i < len_1d - 1; ++i) {
        a[i] = a[i + m] + b[i];
    }
}

void s151_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 5 * iterations; ++nl) {
            s151s_kernel(a, b, len_1d, 1);
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s152s + s152
// ------------------------------------------------------------
static inline void s152s_kernel(
    double * a,
    const double * b,
    const double * c,
    const int i
){
    a[i] += b[i] * c[i];
}

void s152_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 0; i < len_1d; ++i) {
                b[i] = d[i] * e[i];
                s152s_kernel(a, b, c, i);
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s161
// ------------------------------------------------------------
void s161_run_timed(
    double * a,
    const double * b,
    double * c,
    const double * d,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations / 2; ++nl) {
            for (int i = 0; i < len_1d; ++i) {

                if (b[i] < 0.0) {
                    // L20
                    c[i + 1] = a[i] + d[i] * d[i];
                } else {
                    // main branch
                    a[i] = c[i] + d[i] * e[i];
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s1161
// ------------------------------------------------------------
void s1161_run_timed(
    double * a,
    double * b,
    double * c,
    const double * d,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 0; i < len_1d; ++i) {
                if (c[i] < 0.0) {
                    b[i] = a[i] + d[i] * d[i];
                } else {
                    a[i] = c[i] + d[i] * e[i];
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s162
// ------------------------------------------------------------
void s162_run_timed(
    double * a,
    const double * b,
    const double * c,
    const int iterations,
    const int k,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            if (k > 0) {
                for (int i = 0; i < len_1d - k; ++i) {
                    a[i] = a[i + k] + b[i] * c[i];
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s171
// ------------------------------------------------------------
void s171_run_timed(
    double * a,
    const double * b,
    const int inc,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 0; i < len_1d; ++i) {
                a[i * inc] += b[i];
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s172
// ------------------------------------------------------------
void s172_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    const int n1,
    const int n3,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = n1 - 1; i < len_1d; i += n3) {
                a[i] += b[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s173
// ------------------------------------------------------------
void s173_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    int k = len_1d / 2;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 10 * iterations; ++nl) {
            for (int i = 0; i < len_1d / 2; ++i) {
                a[i + k] = a[i] + b[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s174
// ------------------------------------------------------------
void s174_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    const int M,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 10 * iterations; ++nl) {
            for (int i = 0; i < M; ++i) {
                a[i + M] = a[i] + b[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s175
// ------------------------------------------------------------
void s175_run_timed(
    double * a,
    const double * b,
    const int inc,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 0; i < len_1d - inc; i += inc) {
                a[i] = a[i + inc] + b[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s176  (convolution)
// ============================================================================
void s176_run_timed(
    double * a,
    const double * b,
    const double * c,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    int m = len_1d / 2;

    auto t1 = clock::now();
    {
        int outer = 4 * (iterations / len_1d);
        for (int nl = 0; nl < outer; ++nl) {
            for (int j = 0; j < (len_1d / 2); ++j) {
                for (int i = 0; i < m; ++i) {
                    a[i] += b[i + m - j - 1] * c[j];
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s211  (statement reordering)
// ============================================================================
void s211_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 1; i < len_1d - 1; ++i) {
                a[i] = b[i - 1] + c[i] * d[i];
                b[i] = b[i + 1] - e[i] * d[i];
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s212  (needs temporary)
// ============================================================================
void s212_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 0; i < len_1d - 1; ++i) {
                a[i] *= c[i];
                b[i] += a[i + 1] * d[i];
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s1213
// ============================================================================
void s1213_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 1; i < len_1d - 1; ++i) {
                a[i] = b[i - 1] + c[i];
                b[i] = a[i + 1] * d[i];
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s221  (recursive update in same loop)
// ============================================================================
void s221_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = iterations / 2;
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 1; i < len_1d; ++i) {
                a[i] += c[i] * d[i];
                b[i] = b[i - 1] + a[i] + d[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s1221  (runtime symbolic resolution)
// ============================================================================
void s1221_run_timed(
    double * a,
    double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 4; i < len_1d; ++i) {
                b[i] = b[i - 4] + a[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s222  (recurrence in middle of vectorizable ops)
// ============================================================================
void s222_run_timed(
    double * a,
    double * b,
    const double * c,
    double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = iterations / 2;
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 1; i < len_1d; ++i) {
                a[i] += b[i] * c[i];
                e[i] = e[i - 1] * e[i - 1];
                a[i] -= b[i] * c[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s231  (loop interchange, column recursion)
// ============================================================================
void s231_run_timed(
    double * aa,
    const double * bb,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int outer = 100 * (iterations / len_2d);
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 0; i < len_2d; ++i) {
                for (int j = 1; j < len_2d; ++j) {
                    aa[j * len_2d + i] =
                        aa[(j - 1) * len_2d + i] + bb[j * len_2d + i];
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s232  (triangular loop interchange)
// ============================================================================
void s232_run_timed(
    double * aa,
    const double * bb,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int outer = 2 * (iterations);
        for (int nl = 0; nl < outer; ++nl) {
            for (int j = 1; j < len_2d; ++j) {
                for (int i = 1; i <= j; ++i) {
                    aa[j * len_2d + i] =
                        aa[j * len_2d + (i - 1)] *
                        aa[j * len_2d + (i - 1)]
                        + bb[j * len_2d + i];
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s1232
// ============================================================================
void s1232_run_timed(
    double * aa,
    const double * bb,
    const double * cc,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();
    {
        int outer = 100 * (iterations / len_2d);
        for (int nl = 0; nl < outer; ++nl) {
            for (int j = 0; j < len_2d; ++j) {
                for (int i = j; i < len_2d; ++i) {
                    aa[i * len_2d + j] =
                        bb[i * len_2d + j] + cc[i * len_2d + j];
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s233
// ============================================================================
void s233_run_timed(
    double * aa,
    double * bb,
    const double * cc,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 100 * (iterations / len_2d);
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 1; i < len_2d; ++i) {

                for (int j = 1; j < len_2d; ++j) {
                    aa[j * len_2d + i] =
                        aa[(j - 1) * len_2d + i] + cc[j * len_2d + i];
                }

                for (int j = 1; j < len_2d; ++j) {
                    bb[j * len_2d + i] =
                        bb[j * len_2d + (i - 1)] + cc[j * len_2d + i];
                }
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================================
// s2233
// ============================================================================
void s2233_run_timed(
    double * aa,
    double * bb,
    const double * cc,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 100 * (iterations / len_2d);
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 1; i < len_2d; ++i) {

                for (int j = 1; j < len_2d; ++j) {
                    aa[j * len_2d + i] =
                        aa[(j - 1) * len_2d + i] + cc[j * len_2d + i];
                }

                for (int j = 1; j < len_2d; ++j) {
                    bb[i * len_2d + j] =
                        bb[(i - 1) * len_2d + j] + cc[i * len_2d + j];
                }
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

void s235_run_timed(
    double * a,
    double * aa,
    const double * b,
    const double * bb,
    const double * c,
    const int iterations,
    const int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    int outer = 200 * (iterations / len_2d);

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 0; i < len_2d; ++i) {
                a[i] += b[i] * c[i];
                for (int j = 1; j < len_2d; ++j) {
                    aa[j * len_2d + i] =
                        aa[(j - 1) * len_2d + i] + bb[j * len_2d + i] * a[i];
                }
            }
        }
    }
    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s241
// ============================================================================
void s241_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 2 * iterations; ++nl) {
            for (int i = 0; i < len_1d - 1; ++i) {
                a[i] = b[i] * c[i] * d[i];
                b[i] = a[i] * a[i + 1] * d[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s242
// ============================================================================
void s242_run_timed(
    double * a,
    const double * b,
    const double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    const double s1,
    const double s2,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    int outer = iterations / 4;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < outer; ++nl) {
            for (int i = 1; i < len_1d; ++i) {
                a[i] = a[i - 1] + s1 + s2 + b[i] + c[i] + d[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s243
// ============================================================================
void s243_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 0; i < len_1d - 1; ++i) {
                a[i] = b[i] + c[i] * d[i];
                b[i] = a[i] + d[i] * e[i];
                a[i] = b[i] + a[i + 1] * d[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s244
// ============================================================================
void s244_run_timed(
    double * a,
    double * b,
    const double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 0; i < len_1d - 1; ++i) {
                a[i] = b[i] + c[i] * d[i];
                b[i] = c[i] + b[i];
                a[i + 1] = b[i] + a[i + 1] * d[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s2244
// ============================================================================
void s2244_run_timed(
    double * a,
    const double * b,
    const double * c,
    const double * e,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            for (int i = 0; i < len_1d - 1; ++i) {
                a[i + 1] = b[i] + e[i];
                a[i] = b[i] + c[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}




// ============================================================================
// s252
// ============================================================================
void s252_run_timed(
    double * a,
    const double * b,
    const double * c,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            double t = 0.0;
            for (int i = 0; i < len_1d; ++i) {
                double s = b[i] * c[i];
                a[i] = s + t;
                t = s;
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s253
// ============================================================================
void s253_run_timed(
    double * a,
    double * b,
    double * c,
    const double * d,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; ++nl) {
            double s = 0.0;
            for (int i = 0; i < len_1d; ++i) {
                if (a[i] > b[i]) {
                    s = a[i] - b[i] * d[i];
                    c[i] += s;
                    a[i] = s;
                }
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================================
// s254
// ============================================================================
void s254_run_timed(
    double * a,
    const double * b,
    const int iterations,
    const int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 4 * iterations; ++nl) {
            double x = b[len_1d - 1];
            for (int i = 0; i < len_1d; ++i) {
                a[i] = 0.5 * (b[i] + x);
                x = b[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}





// ------------------------------------------------------------
// s1244
// ------------------------------------------------------------
void s1244_run_timed(
    double* a,
    const double* b,
    const double* c,
    double* d,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            for (int i = 0; i < len_1d - 1; i++) {
                a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
                d[i] = a[i] + a[i+1];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s251
// ------------------------------------------------------------
void s251_run_timed(
    double* a,
    const double* b,
    const double* c,
    const double* d,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < 4 * iterations; nl++) {
            for (int i = 0; i < len_1d; i++) {
                double s = b[i] + c[i] * d[i];
                a[i] = s * s;
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s1251
// ------------------------------------------------------------
void s1251_run_timed(
    double* a,
    double* b,
    const double* c,
    const double* d,
    const double* e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < 4 * iterations; nl++) {
            for (int i = 0; i < len_1d; i++) {
                double s = b[i] + c[i];
                b[i] = a[i] + d[i];
                a[i] = s * e[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s2251
// ------------------------------------------------------------
void s2251_run_timed(
    double* a,
    double* b,
    const double* c,
    const double* d,
    const double* e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            double s = 0.0;
            for (int i = 0; i < len_1d; i++) {
                a[i] = s * e[i];
                s = b[i] + c[i];
                b[i] = a[i] + d[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s3251
// ------------------------------------------------------------
void s3251_run_timed(
    double* a,
    double* b,
    const double* c,
    double* d,
    const double* e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            for (int i = 0; i < len_1d - 1; i++) {
                a[i+1] = b[i] + c[i];
                b[i]   = c[i] * e[i];
                d[i]   = a[i] * e[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s255
// ------------------------------------------------------------
void s255_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            double x = b[len_1d - 1];
            double y = b[len_1d - 2];
            for (int i = 0; i < len_1d; i++) {
                a[i] = (b[i] + x + y) * 0.333;
                y = x;
                x = b[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s256
// ------------------------------------------------------------
void s256_run_timed(
    double* a,
    double* aa,
    const double* bb,
    const double* d,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        int outer = 10 * (iterations / len_2d);
        for (int nl = 0; nl < outer; nl++) {
            for (int i = 0; i < len_2d; i++) {
                for (int j = 1; j < len_2d; j++) {
                    a[j] = 1.0 - a[j - 1];
                    aa[j * len_2d + i] =
                        a[j] + bb[j * len_2d + i] * d[j];
                }
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s257
// ------------------------------------------------------------
void s257_run_timed(
    double* a,
    double* aa,
    const double* bb,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        int outer = 10 * (iterations / len_2d);
        for (int nl = 0; nl < outer; nl++) {
            for (int i = 1; i < len_2d; i++) {
                for (int j = 0; j < len_2d; j++) {
                    a[i] = aa[j * len_2d + i] - a[i - 1];
                    aa[j * len_2d + i] = a[i] + bb[j * len_2d + i];
                }
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s258
// ------------------------------------------------------------
void s258_run_timed(
    double* a,
    const double* aa,
    double* b,
    const double* c,
    const double* d,
    double* e,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            double s = 0.0;
            for (int i = 0; i < len_2d; i++) {
                if (a[i] > 0.0)
                    s = d[i] * d[i];

                b[i] = s * c[i] + d[i];
                e[i] = (s + 1.0) * aa[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s261
// ------------------------------------------------------------
void s261_run_timed(
    double* a,
    double* b,
    double* c,
    const double* d,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            for (int i = 1; i < len_1d; i++) {
                double t = a[i] + b[i];
                a[i] = t + c[i - 1];
                c[i] = c[i] * d[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s271
// ------------------------------------------------------------
void s271_run_timed(
    double* a,
    const double* b,
    const double* c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < 4 * iterations; nl++) {
            for (int i = 0; i < len_1d; i++) {
                if (b[i] > 0.0)
                    a[i] += b[i] * c[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s272
// ------------------------------------------------------------
void s272_run_timed(
    double* a,
    double* b,
    const double* c,
    const double* d,
    const double* e,
    int iterations,
    int len_1d,
    int threshold,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            for (int i = 0; i < len_1d; i++) {
                if (e[i] >= threshold) {
                    a[i] += c[i] * d[i];
                    b[i] += c[i] * c[i];
                }
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s273
// ------------------------------------------------------------
void s273_run_timed(
    double* a,
    double* b,
    double* c,
    const double* d,
    const double* e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            for (int i = 0; i < len_1d; i++) {
                a[i] += d[i] * e[i];
                if (a[i] < 0.0)
                    b[i] += d[i] * e[i];
                c[i] += a[i] * d[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s274
// ------------------------------------------------------------
void s274_run_timed(
    double* a,
    double* b,
    const double* c,
    const double* d,
    const double* e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        for (int nl = 0; nl < iterations; nl++) {
            for (int i = 0; i < len_1d; i++) {
                a[i] = c[i] + e[i] * d[i];
                if (a[i] > 0.0)
                    b[i] = a[i] + b[i];
                else
                    a[i] = d[i] * e[i];
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s275
// ------------------------------------------------------------
void s275_run_timed(
    double* aa,
    const double* bb,
    const double* cc,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    {
        int outer = 10 * (iterations / len_2d);
        for (int nl = 0; nl < outer; nl++) {
            for (int i = 0; i < len_2d; i++) {
                if (aa[i] > 0.0) {
                    for (int j = 1; j < len_2d; j++) {
                        aa[j * len_2d + i] =
                            aa[(j - 1) * len_2d + i]
                            + bb[j * len_2d + i] * cc[j * len_2d + i];
                    }
                }
            }
        }
    }

    auto t2 = clock::now();
    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}



// ------------------------------------------------------------
// s281
// ------------------------------------------------------------
void s281_run_timed(
    double* a,
    double* b,
    const double* c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; nl++) {
            for (int i = 0; i < len_1d; i++) {
                double x = a[len_1d - i - 1] + b[i] * c[i];
                a[i] = x - 1.0;
                b[i] = x;
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s1281
// ------------------------------------------------------------
void s1281_run_timed(
    double* a,
    double* b,
    const double* c,
    const double* d,
    const double* e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 4 * iterations;
        for (int nl = 0; nl < outer; nl++) {
            for (int i = 0; i < len_1d; i++) {
                double x = b[i] * c[i] + a[i] * d[i] + e[i];
                a[i] = x - 1.0;
                b[i] = x;
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s291
// ------------------------------------------------------------
void s291_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 2 * iterations;
        for (int nl = 0; nl < outer; nl++) {
            int im1 = len_1d - 1;
            for (int i = 0; i < len_1d; i++) {
                a[i] = (b[i] + b[im1]) * 0.5;
                im1 = i;
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s292
// ------------------------------------------------------------
void s292_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < iterations; nl++) {
            int im1 = len_1d - 1;
            int im2 = len_1d - 2;
            for (int i = 0; i < len_1d; i++) {
                a[i] = (b[i] + b[im1] + b[im2]) * 0.333;
                im2 = im1;
                im1 = i;
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s293
// ------------------------------------------------------------
void s293_run_timed(
    double* a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 4 * iterations;
        double a0 = a[0];
        for (int nl = 0; nl < outer; nl++) {
            for (int i = 0; i < len_1d; i++) {
                a[i] = a0;
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s2101
// ------------------------------------------------------------
void s2101_run_timed(
    double* aa,
    const double* bb,
    const double* cc,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 10 * iterations;
        for (int nl = 0; nl < outer; nl++) {
            for (int i = 0; i < len_2d; i++) {
                aa[i * len_2d + i] +=
                    bb[i * len_2d + i] * cc[i * len_2d + i];
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s2102
// ------------------------------------------------------------
void s2102_run_timed(
    double* aa,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 100 * (iterations / len_2d);
        for (int nl = 0; nl < outer; nl++) {
            for (int i = 0; i < len_2d; i++) {
                for (int j = 0; j < len_2d; j++) {
                    aa[j * len_2d + i] = 0.0;
                }
                aa[i * len_2d + i] = 1.0;
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s2111
// ------------------------------------------------------------
void s2111_run_timed(
    double* aa,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 100 * (iterations / len_2d);
        for (int nl = 0; nl < outer; nl++) {
            for (int j = 1; j < len_2d; j++) {
                for (int i = 1; i < len_2d; i++) {
                    double left  = aa[j * len_2d + (i - 1)];
                    double upper = aa[(j - 1) * len_2d + i];
                    aa[j * len_2d + i] = (left + upper) / 1.9;
                }
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// s311
// ------------------------------------------------------------
void s311_run_timed(
    double* a,
    double* sum_out,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 10 * iterations;
        for (int nl = 0; nl < outer; nl++) {
            sum_out[0] = 0.0;
            for (int i = 0; i < len_1d; i++) {
                sum_out[0] += a[i];
            }
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------------------------------------------------------
// helper test() (used by s31111)
// ------------------------------------------------------------
double s31111_test(const double* A)
{
    double s = 0.0;
    for (int i = 0; i < 4; i++)
        s += A[i];
    return s;
}


// ------------------------------------------------------------
// s31111
// ------------------------------------------------------------
void s31111_run_timed(
    double* a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int outer = 2000 * iterations;
        for (int nl = 0; nl < outer; nl++) {
            double sum = 0.0;
            for (int base = 0; base < len_1d; base += 4)
                sum += s31111_test(&a[base]);

            //asm volatile("" :: "r,m"(sum));
        }
    }
    auto t2 = clock::now();

    *time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// s2275: uses a, b, c, d, aa, bb, cc
void s2275_run_timed(
    double *a,
    double *aa,
    const double *b,
    const double *bb,
    const double *c,
    const double *cc,
    const double *d,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    int nl_max = 100 * (iterations / len_2d);
    for (int nl = 0; nl < nl_max; ++nl) {
        for (int i = 0; i < len_2d; ++i) {
            for (int j = 0; j < len_2d; ++j) {
                int idx = j * len_2d + i;
                aa[idx] = aa[idx] + bb[idx] * cc[idx];
            }
            a[i] = b[i] + c[i] * d[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s276: uses a, b, c, d
void s276_run_timed(
    double *a,
    const double *b,
    const double *c,
    const double *d,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    int mid = len_1d / 2;
    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (i + 1 < mid) {
                a[i] += b[i] * c[i];
            } else {
                a[i] += b[i] * d[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s277: uses a, b, c, d, e
void s277_run_timed(
    double *a,
    double *b,
    const double *c,
    const double *d,
    const double *e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d - 1; ++i) {
            if (a[i] >= 0.0) {
                goto L20;
            }
            if (b[i] >= 0.0) {
                goto L30;
            }
            a[i] += c[i] * d[i];
        L30:
            b[i + 1] = c[i] + d[i] * e[i];
        L20:
            ;
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s278: uses a, b, c, d, e
void s278_run_timed(
    double *a,
    double *b,
    double *c,
    const double *d,
    const double *e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (a[i] > 0.0) {
                goto L20;
            }
            b[i] = -b[i] + d[i] * e[i];
            goto L30;
        L20:
            c[i] = -c[i] + d[i] * e[i];
        L30:
            a[i] = b[i] + c[i] * d[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s279: uses a, b, c, d, e
void s279_run_timed(
    double *a,
    double *b,
    double *c,
    const double *d,
    const double *e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations / 2; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (a[i] > 0.0) {
                goto L20;
            }
            b[i] = -b[i] + d[i] * d[i];
            if (b[i] <= a[i]) {
                goto L30;
            }
            c[i] += d[i] * e[i];
            goto L30;
        L20:
            c[i] = -c[i] + e[i] * e[i];
        L30:
            a[i] = b[i] + c[i] * d[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s1279: uses a, b, c, d, e
void s1279_run_timed(
    const double *a,
    const double *b,
    double *c,
    const double *d,
    const double *e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (a[i] < 0.0) {
                if (b[i] > a[i]) {
                    c[i] += d[i] * e[i];
                }
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s2710: uses a, b, c, d, e and scalar x
void s2710_run_timed(
    double *a,
    double *b,
    double *c,
    const double *d,
    const double *e,
    const double *x,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations / 2; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (a[i] > b[i]) {
                a[i] += b[i] * d[i];
                if (len_1d > 10) {
                    c[i] += d[i] * d[i];
                } else {
                    c[i] = d[i] * e[i] + 1.0;
                }
            } else {
                b[i] = a[i] + e[i] * e[i];
                if (x[0] > 0.0) {
                    c[i] = a[i] + d[i] * d[i];
                } else {
                    c[i] += e[i] * e[i];
                }
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s2711: uses a, b, c
void s2711_run_timed(
    double *a,
    const double *b,
    const double *c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (b[i] != 0.0) {
                a[i] += b[i] * c[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s2712: uses a, b, c
void s2712_run_timed(
    double *a,
    const double *b,
    const double *c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (a[i] > b[i]) {
                a[i] += b[i] * c[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}



// ------------------------------------------------------------
// s312: product reduction over a
// ------------------------------------------------------------
void s312_run_timed(
    double *a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        double prod;
        for (int nl = 0; nl < 10 * iterations; ++nl) {
            prod = 1.0;
            for (int i = 0; i < len_1d; ++i) {
                prod *= a[i];
            }
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s313: dot product a·b
// ------------------------------------------------------------
void s313_run_timed(
    const double *a,
    const double *b,
    double *dot,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 5 * iterations; ++nl) {
            dot[0] = 0.0;
            for (int i = 0; i < len_1d; ++i) {
                dot[0] += a[i] * b[i];
            }
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s314: max reduction over a
// ------------------------------------------------------------
void s314_run_timed(
    const double *a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        double x;
        for (int nl = 0; nl < 5 * iterations; ++nl) {
            x = a[0];
            for (int i = 0; i < len_1d; ++i) {
                if (a[i] > x) {
                    x = a[i];
                }
            }
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s315: max reduction with index (1D)
// ------------------------------------------------------------
void s315_run_timed(
    double *a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        // Initial permutation of a (inside timed region)
        for (int i = 0; i < len_1d; ++i) {
            a[i] = static_cast<double>((i * 7) % len_1d);
        }

        double x;
        int index;
        for (int nl = 0; nl < iterations; ++nl) {
            x = a[0];
            index = 0;
            for (int i = 0; i < len_1d; ++i) {
                if (a[i] > x) {
                    x = a[i];
                    index = i;
                }
            }
            volatile double chksum = x + static_cast<double>(index);
            (void)chksum;
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s316: min reduction over a
// ------------------------------------------------------------
void s316_run_timed(
    const double *a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        double x;
        for (int nl = 0; nl < 5 * iterations; ++nl) {
            x = a[0];
            for (int i = 1; i < len_1d; ++i) {
                if (a[i] < x) {
                    x = a[i];
                }
            }
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s317: pure scalar product reduction (q *= 0.99)
// ------------------------------------------------------------
void s317_run_timed(
    double *q,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        for (int nl = 0; nl < 5 * iterations; ++nl) {
            q[0] = 1.0;
            for (int i = 0; i < len_1d / 2; ++i) {
                q[0] *= 0.99;
            }
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s318: isamax-style max |a[k]| with increment inc
// ------------------------------------------------------------
void s318_run_timed(
    const double *a,
    int inc,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        int k, index;
        double maxv, chksum;
        for (int nl = 0; nl < iterations / 2; ++nl) {
            k = 0;
            index = 0;
            maxv = std::fabs(a[0]);
            k += inc;
            for (int i = 1; i < len_1d; ++i) {
                double v = std::fabs(a[k]);
                if (v > maxv) {
                    index = i;
                    maxv = v;
                }
                k += inc;
            }
            chksum = maxv + static_cast<double>(index);
            volatile double sink = chksum;
            (void)sink;
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s319: coupled reductions on a and b
// ------------------------------------------------------------
void s319_run_timed(
    double *a,
    double *b,
    const double *c,
    const double *d,
    const double *e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        double sum;
        for (int nl = 0; nl < 2 * iterations; ++nl) {
            sum = 0.0;
            for (int i = 0; i < len_1d; ++i) {
                a[i] = c[i] + d[i];
                sum += a[i];
                b[i] = c[i] + e[i];
                sum += b[i];
            }
            volatile double sink = sum;
            (void)sink;
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s3110: 2D max reduction with indices
// ------------------------------------------------------------
void s3110_run_timed(
    double *aa,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        auto idx = [len_2d](int i, int j) { return i * len_2d + j; };

        int xindex, yindex;
        double maxv, chksum;
        for (int nl = 0; nl < 100 * (iterations / len_2d); ++nl) {
            maxv = aa[idx(0, 0)];
            xindex = 0;
            yindex = 0;
            for (int i = 0; i < len_2d; ++i) {
                for (int j = 0; j < len_2d; ++j) {
                    double v = aa[idx(i, j)];
                    if (v > maxv) {
                        maxv = v;
                        xindex = i;
                        yindex = j;
                    }
                }
            }
            chksum = maxv + static_cast<double>(xindex) + static_cast<double>(yindex);
            volatile double sink = chksum;
            (void)sink;
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s13110: same pattern as s3110 (variant)
// ------------------------------------------------------------
void s13110_run_timed(
    double *aa,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        auto idx = [len_2d](int i, int j) { return i * len_2d + j; };

        int xindex, yindex;
        double maxv, chksum;
        for (int nl = 0; nl < 100 * (iterations / len_2d); ++nl) {
            maxv = aa[idx(0, 0)];
            xindex = 0;
            yindex = 0;
            for (int i = 0; i < len_2d; ++i) {
                for (int j = 0; j < len_2d; ++j) {
                    double v = aa[idx(i, j)];
                    if (v > maxv) {
                        maxv = v;
                        xindex = i;
                        yindex = j;
                    }
                }
            }
            chksum = maxv + static_cast<double>(xindex) + static_cast<double>(yindex);
            volatile double sink = chksum;
            (void)sink;
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ------------------------------------------------------------
// s3111: conditional sum reduction over a>0
// ------------------------------------------------------------
void s3111_run_timed(
    const double *a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;

    auto t1 = clock::now();
    {
        double sum;
        for (int nl = 0; nl < iterations / 2; ++nl) {
            sum = 0.0;
            for (int i = 0; i < len_1d; ++i) {
                if (a[i] > 0.0) {
                    sum += a[i];
                }
            }
            volatile double sink = sum;
            (void)sink;
        }
    }
    auto t2 = clock::now();

    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ------------- Helpers -------------
using clock_highres = std::chrono::high_resolution_clock;

// ======================
// %3.1 – Reductions
// ======================

// s3112: running sum, stored into b
void s3112_run_timed(
    const double *a,
    double *b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    double sum;
    for (int nl = 0; nl < iterations; ++nl) {
        sum = 0.0;
        for (int i = 0; i < len_1d; ++i) {
            sum += a[i];
            b[i] = sum;
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s3113: maximum of absolute value
void s3113_run_timed(
    const double *a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    double maxv;
    for (int nl = 0; nl < iterations * 4; ++nl) {
        maxv = std::fabs(a[0]);
        for (int i = 0; i < len_1d; ++i) {
            double av = std::fabs(a[i]);
            if (av > maxv) {
                maxv = av;
            }
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ======================
// %3.2 – Recurrences
// ======================

// s321: first-order linear recurrence
void s321_run_timed(
    double *a,
    const double *b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 1; i < len_1d; ++i) {
            a[i] += a[i - 1] * b[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s322: second-order linear recurrence
void s322_run_timed(
    double *a,
    const double *b,
    const double *c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < iterations / 2; ++nl) {
        for (int i = 2; i < len_1d; ++i) {
            a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s323: coupled recurrence
void s323_run_timed(
    double *a,
    double *b,
    const double *c,
    const double *d,
    const double *e,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < iterations / 2; ++nl) {
        for (int i = 1; i < len_1d; ++i) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ======================
// %3.3 – Search loops
// ======================

// s331: last index with a[i] < 0
void s331_run_timed(
    const double *a,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    int j = -1;
    for (int nl = 0; nl < iterations; ++nl) {
        j = -1;
        for (int i = 0; i < len_1d; ++i) {
            if (a[i] < 0.0) {
                j = i;
            }
        }
        // chksum = (real_t) j;  // ignored in timed version
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ======================
// %3.4 – Packing
// ======================

// s341: pack positive values from b into a
void s341_run_timed(
    double *a,
    const double *b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    int j;
    for (int nl = 0; nl < iterations; ++nl) {
        j = -1;
        for (int i = 0; i < len_1d; ++i) {
            if (b[i] > 0.0) {
                ++j;
                a[j] = b[i];
            }
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s342: unpacking using a as mask into b
void s342_run_timed(
    double *a,
    const double *b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    int j = 0;
    for (int nl = 0; nl < iterations; ++nl) {
        j = -1;
        for (int i = 0; i < len_1d; ++i) {
            if (a[i] > 0.0) {
                ++j;
                a[i] = b[j];
            }
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s343: pack 2D aa into flat_2d_array based on bb > 0
void s343_run_timed(
    const double *aa,
    const double *bb,
    double *flat_2d_array,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < 10 * (iterations / len_2d); ++nl) {
        int k = -1;
        for (int i = 0; i < len_2d; ++i) {
            for (int j = 0; j < len_2d; ++j) {
                int idx = j * len_2d + i;
                if (bb[idx] > 0.0) {
                    ++k;
                    flat_2d_array[k] = aa[idx];
                }
            }
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ======================
// %3.5 – Loop rerolling
// ======================

// s351: unrolled SAXPY (5-way)
void s351_run_timed(
    double *a,
    const double *b,
    const double *c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    double alpha = c[0];
    for (int nl = 0; nl < 8 * iterations; ++nl) {
        for (int i = 0; i < len_1d; i += 5) {
            a[i]     += alpha * b[i];
            a[i + 1] += alpha * b[i + 1];
            a[i + 2] += alpha * b[i + 2];
            a[i + 3] += alpha * b[i + 3];
            a[i + 4] += alpha * b[i + 4];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s1351: induction pointer recognition – a[i] = b[i] + c[i]
void s1351_run_timed(
    double *a,
    const double *b,
    const double *c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < 8 * iterations; ++nl) {
        const double *B = b;
        const double *C = c;
        double *A = a;
        for (int i = 0; i < len_1d; ++i) {
            *A = *B + *C;
            ++A;
            ++B;
            ++C;
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s352: unrolled dot product (5-way)
void s352_run_timed(
    const double *a,
    const double *b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    double dot;
    for (int nl = 0; nl < 8 * iterations; ++nl) {
        dot = 0.0;
        for (int i = 0; i < len_1d; i += 5) {
            dot += a[i]     * b[i]
                 + a[i + 1] * b[i + 1]
                 + a[i + 2] * b[i + 2]
                 + a[i + 3] * b[i + 3]
                 + a[i + 4] * b[i + 4];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s353: unrolled sparse SAXPY (gather through ip)
void s353_run_timed(
    double *a,
    const double *b,
    const double *c,
    const int *ip,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    double alpha = c[0];
    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; i += 5) {
            a[i]     += alpha * b[ip[i]];
            a[i + 1] += alpha * b[ip[i + 1]];
            a[i + 2] += alpha * b[ip[i + 2]];
            a[i + 3] += alpha * b[ip[i + 3]];
            a[i + 4] += alpha * b[ip[i + 4]];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ===============================
// %4.1 – %4.2 Storage / aliasing
// ===============================

// s421: xx = flat_2d_array; yy = xx;
// xx[i] = yy[i+1] + a[i];
void s421_run_timed(
    const double *a,
    double *flat_2d_array,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d - 1; ++i) {
            flat_2d_array[i] = flat_2d_array[i + 1] + a[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s1421: xx = &b[LEN_1D/2]; b[i] = xx[i] + a[i];
void s1421_run_timed(
    const double *a,
    double *b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    int half = len_1d / 2;
    for (int nl = 0; nl < 8 * iterations; ++nl) {
        for (int i = 0; i < half; ++i) {
            b[i] = b[half + i] + a[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s422: xx = flat_2d_array + 4;
// xx[i] = flat_2d_array[i+8] + a[i];
void s422_run_timed(
    const double *a,
    double *flat_2d_array,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < 8 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            flat_2d_array[4 + i] = flat_2d_array[8 + i] + a[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// s423: xx = flat_2d_array + vl; vl = 64;
// flat_2d_array[i+1] = xx[i] + a[i];
void s423_run_timed(
    const double *a,
    double *flat_2d_array,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    const int vl = 64;
    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d - 1; ++i) {
            flat_2d_array[i + 1] = flat_2d_array[vl + i] + a[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// ============================================================
// vpvtv — vector plus vector times vector
// ============================================================

void vpvtv_run_timed(
    double *a,
    const double *b,
    const double *c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] += b[i] * c[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================
// vpvts — vector plus vector times scalar
// ============================================================

void vpvts_run_timed(
    double *a,
    const double *b,
    int iterations,
    int len_1d,
    double s,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] += b[i] * s;
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================
// vpvpv — vector plus vector plus vector
// ============================================================

void vpvpv_run_timed(
    double *a,
    const double *b,
    const double *c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] += b[i] + c[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================
// vtvtv — vector times vector times vector
// ============================================================

void vtvtv_run_timed(
    double *a,
    const double *b,
    const double *c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] = a[i] * b[i] * c[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// ============================================================
// vsumr — sum reduction
// ============================================================

void vsumr_run_timed(
    const double *a,
    double *sum_out,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    double sum = 0.0;
    for (int nl = 0; nl < iterations * 10; ++nl) {
        sum = 0.0;
        for (int i = 0; i < len_1d; ++i) {
            sum += a[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    *sum_out = sum;
}

// ============================================================
// vdotr — vector dot product
// ============================================================

void vdotr_run_timed(
    const double *a,
    const double *b,
    double *dot_out,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    double dot = 0.0;
    for (int nl = 0; nl < iterations * 10; ++nl) {
        dot = 0.0;
        for (int i = 0; i < len_1d; ++i) {
            dot += a[i] * b[i];
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    *dot_out = dot;
}

// ============================================================
// vbor — 59 flops kernel
// ============================================================

void vbor_run_timed(
    const double *a,
    const double *b,
    const double *c,
    const double *d,
    const double *e,
    double *x,
    int iterations,
    int len_2d,
    std::int64_t* time_ns
){
    auto t1 = clock_highres::now();

    double a1, b1, c1, d1, e1, f1;
    for (int nl = 0; nl < iterations * 10; ++nl) {
        for (int i = 0; i < len_2d; ++i) {
            a1 = a[i];
            b1 = b[i];
            c1 = c[i];
            d1 = d[i];
            e1 = e[i];
            f1 = a[i];

            a1 = a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
                 a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1 +
                 a1 * d1 * f1 + a1 * e1 * f1;

            b1 = b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 +
                 b1 * d1 * e1 + b1 * d1 * f1 + b1 * e1 * f1;

            c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1;

            d1 = d1 * e1 * f1;

            x[i] = a1 * b1 * c1 * d1;
        }
    }

    auto t2 = clock_highres::now();
    time_ns[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}


// -----------------------------------------------------------------------------
// Helpers (pure, small)
// -----------------------------------------------------------------------------
static inline double tsvc_mul(double a, double b) {
    return a * b;
}

// -----------------------------------------------------------------------------
// %4.2  s424
// -----------------------------------------------------------------------------
void s424_run_timed(
    double* a,
    const double* flat,
    double* xx,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    // TSVC uses: vl = 63; xx = flat_2d_array + vl;
    // Here: caller passes xx already pointing to the shifted region.
    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d - 1; ++i) {
            xx[i + 1] = flat[i] + a[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.3  s431
// -----------------------------------------------------------------------------
void s431_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    // k1=1; k2=2; k=2*k1-k2 => k = 0, so a[i] = a[i] + b[i]
    for (int nl = 0; nl < iterations * 10; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] = a[i] + b[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.4  s441
// -----------------------------------------------------------------------------
void s441_run_timed(
    double* a,
    const double* b,
    const double* c,
    const double* d,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (d[i] < 0.0) {
                a[i] += b[i] * c[i];
            } else if (d[i] == 0.0) {
                a[i] += b[i] * b[i];
            } else {
                a[i] += c[i] * c[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.4  s442
// -----------------------------------------------------------------------------
void s442_run_timed(
    double* a,
    const double* b,
    const double* c,
    const double* d,
    const double* e,
    const int* indx,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations / 2; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            switch (indx[i]) {
                case 1:
                    a[i] += b[i] * b[i];
                    break;
                case 2:
                    a[i] += c[i] * c[i];
                    break;
                case 3:
                    a[i] += d[i] * d[i];
                    break;
                case 4:
                    a[i] += e[i] * e[i];
                    break;
                default:
                    break;
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.4  s443
// -----------------------------------------------------------------------------
void s443_run_timed(
    double* a,
    const double* b,
    const double* c,
    const double* d,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < 2 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (d[i] <= 0.0) {
                a[i] += b[i] * c[i];
            } else {
                a[i] += b[i] * b[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.5  s451
// -----------------------------------------------------------------------------
void s451_run_timed(
    double* a,
    const double* b,
    const double* c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations / 4; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] = std::sin(b[i]) + std::cos(c[i]);
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.5  s452
// -----------------------------------------------------------------------------
void s452_run_timed(
    double* a,
    const double* b,
    const double* c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < 4 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] = b[i] + c[i] * static_cast<double>(i + 1);
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.5  s453
// -----------------------------------------------------------------------------
void s453_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    double s = 0.0;
    for (int nl = 0; nl < iterations * 2; ++nl) {
        s = 0.0;
        for (int i = 0; i < len_1d; ++i) {
            s += 2.0;
            a[i] = s * b[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.7  s471  (s471s is a dummy)
// -----------------------------------------------------------------------------
int s471s() {
    return 0;
}

void s471_run_timed(
    double* b,
    const double* c,
    const double* d,
    const double* e,
    double* x,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    int m = len_1d;
    for (int nl = 0; nl < iterations / 2; ++nl) {
        for (int i = 0; i < m; ++i) {
            x[i] = b[i] + d[i] * d[i];
            s471s();
            b[i] = c[i] + d[i] * e[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.8  s481  (exit(0) -> early break)
// -----------------------------------------------------------------------------
void s481_run_timed(
    double* a,
    const double* b,
    const double* c,
    const double* d,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    bool stop = false;
    for (int nl = 0; nl < iterations && !stop; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (d[i] < 0.0) {
                stop = true;
                break;
            }
            a[i] += b[i] * c[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.8  s482
// -----------------------------------------------------------------------------
void s482_run_timed(
    double* a,
    const double* b,
    const double* c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] += b[i] * c[i];
            if (c[i] > b[i]) {
                break;
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.9  s491
// -----------------------------------------------------------------------------
void s491_run_timed(
    double* a,
    const double* b,
    const double* c,
    const double* d,
    const int* ip,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[ip[i]] = b[i] + c[i] * d[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.11  s4112
// -----------------------------------------------------------------------------
void s4112_run_timed(
    double* a,
    const double* b,
    const int* ip,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] += b[ip[i]] * 2.0;
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.11  s4113
// -----------------------------------------------------------------------------
void s4113_run_timed(
    double* a,
    const double* b,
    const double* c,
    const int* ip,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            int idx = ip[i];
            a[idx] = b[idx] + c[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.11  s4114
// -----------------------------------------------------------------------------
void s4114_run_timed(
    double* a,
    const double* b,
    const double* c,
    const double* d,
    const int* ip,
    int iterations,
    int len_1d,
    int n1,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    int k;
    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = n1 - 1; i < len_1d; ++i) {
            k = ip[i];
            a[i] = b[i] + c[len_1d - k - 1] * d[i];
            k += 5; // has no effect on further iterations, kept for fidelity
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.11  s4115
// -----------------------------------------------------------------------------
void s4115_run_timed(
    const double* a,
    const double* b,
    const int* ip,
    double* result_out,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    double sum = 0.0;
    for (int nl = 0; nl < iterations; ++nl) {
        sum = 0.0;
        for (int i = 0; i < len_1d; ++i) {
            sum += a[i] * b[ip[i]];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    result_out[0] = sum;
}

// -----------------------------------------------------------------------------
// %4.11  s4116
// -----------------------------------------------------------------------------
void s4116_run_timed(
    const double* a,
    const double* aa,
    const int* ip,
    double *sum_out,
    int inc,
    int iterations,
    int j,
    int len_1d,
    int len_2d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < 100 * iterations; ++nl) {
        sum_out[0] = 0.0;
        for (int i = 0; i < len_2d - 1; ++i) {
            int off = inc + i;
            sum_out[0] += a[off] * aa[(j - 1) * len_2d + ip[i]];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.11  s4117
// -----------------------------------------------------------------------------
void s4117_run_timed(
    double* a,
    const double* b,
    const double* c,
    const double* d,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] = b[i] + c[i / 2] * d[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %4.12  s4121
// -----------------------------------------------------------------------------
void s4121_run_timed(
    double* a,
    const double* b,
    const double* c,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] += tsvc_mul(b[i], c[i]);
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %5.1  va
// -----------------------------------------------------------------------------
void va_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations * 10; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] = b[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %5.1  vag
// -----------------------------------------------------------------------------
void vag_run_timed(
    double* a,
    const double* b,
    const int* ip,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < 2 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] = b[ip[i]];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %5.1  vas
// -----------------------------------------------------------------------------
void vas_run_timed(
    double* a,
    const double* b,
    const int* ip,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < 2 * iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[ip[i]] = b[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %5.1  vif
// -----------------------------------------------------------------------------
void vif_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            if (b[i] > 0.0) {
                a[i] = b[i];
            }
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %5.1  vpv
// -----------------------------------------------------------------------------
void vpv_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations * 10; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] += b[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

// -----------------------------------------------------------------------------
// %5.1  vtv
// -----------------------------------------------------------------------------
void vtv_run_timed(
    double* a,
    const double* b,
    int iterations,
    int len_1d,
    std::int64_t* time_ns
){
    using clock = std::chrono::high_resolution_clock;
    auto t1 = clock::now();

    for (int nl = 0; nl < iterations * 10; ++nl) {
        for (int i = 0; i < len_1d; ++i) {
            a[i] *= b[i];
        }
    }

    auto t2 = clock::now();
    time_ns[0] =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

} // extern "C"
