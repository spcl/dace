// LEVER 2 -- loop-idiom recognition firing when it should NOT (glibc memcpy per-call overhead).
//
// Copy a narrow block of W columns out of a WIDE matrix (row pitch LDA >> W, so rows are
// discontiguous and cannot coalesce). Two forms of the SAME copy:
//
//   SLOW (idiom): the inner element loop is written canonically. clang's LoopIdiomRecognize
//     turns each row into a memcpy. To force it to a REAL glibc CALL (not inline expansion)
//     the length W*8 is made opaque (runtime, unknown to the backend), so per-call dispatch
//     overhead is paid once per row. This is the shape from CODEGEN_STYLE_PERFORMANCE.md 5.
//   FAST (inline): identical copy done with an explicit fixed-width vector store, which the
//     idiom matcher does not touch -> inlined loads/stores, no call.
//
// Working set kept L2-resident and reused OUTER times, so the wall is per-CALL overhead, not
// memory bandwidth. Result bit-identical (same bytes, same order).
//
// MECHANISM (named): llvm LoopIdiomRecognizePass -> processLoopStoreOfLoopLoad() emitting a
// runtime-length llvm.memcpy that SelectionDAG lowers to `call memcpy@plt`.
#include "bench.h"

// Opaque length: an external-ish barrier so the backend cannot see the constant and must emit a
// real call for the memcpy form.
static volatile int g_bytes = 64;  // = W(8) * 8

// SLOW: memcpy per row, runtime length -> real glibc call per row.
__attribute__((noinline)) void copy_call(const double *src, double *dst, int rows, int lda, size_t nbytes) {
    for (int r = 0; r < rows; ++r)
        std::memcpy(dst + (int64_t)r * lda, src + (int64_t)r * lda, nbytes);
}

// FAST: inlined 8-wide (64-byte) vector copy per row. Same bytes moved.
typedef double v8d __attribute__((vector_size(64)));
__attribute__((noinline)) void copy_inline(const double *src, double *dst, int rows, int lda, size_t) {
    for (int r = 0; r < rows; ++r) {
        v8d v;
        std::memcpy(&v, src + (int64_t)r * lda, 64);
        std::memcpy(dst + (int64_t)r * lda, &v, 64);
    }
}

// TWIN of fast form -> noise/layout floor.
__attribute__((noinline)) void copy_inline_twin(const double *src, double *dst, int rows, int lda, size_t) {
    for (int r = 0; r < rows; ++r) {
        v8d v;
        std::memcpy(&v, src + (int64_t)r * lda, 64);
        std::memcpy(dst + (int64_t)r * lda, &v, 64);
    }
}

int main() {
    const int W = 8;
    const int LDA = 64;
    const int ROWS = 512;       // 2 * 512 * 64 * 8 = 512 KiB -> L2 resident
    const int OUTER = 4000;
    const size_t N = size_t(LDA) * ROWS;
    const size_t NB = size_t(W) * 8;
    g_bytes = (int)NB;

    std::vector<double> src(N), a(N, 0), b(N, 0), c(N, 0);
    for (size_t i = 0; i < N; ++i)
        src[i] = double(i % 1000) * 0.5 + 1.0;

    auto run = [&](auto &&f, std::vector<double> &out) {
        return [&] {
            size_t nb = (size_t)g_bytes;  // opaque
            for (int k = 0; k < OUTER; ++k)
                f(src.data(), out.data(), ROWS, LDA, nb);
        };
    };

    Stat s_inline = bench(run(copy_inline, a));
    Stat s_twin = bench(run(copy_inline_twin, b));
    Stat s_call = bench(run(copy_call, c));

    CHECK_BITSAME(a.data(), c.data(), N * 8, "inline vs call");

    std::printf("LEVER 2: memcpy idiom / glibc per-call overhead  (W=%d, lda=%d, rows=%d, outer=%d, %zu B/row)\n", W, LDA,
                ROWS, OUTER, NB);
    row("copy_inline      [FAST]", s_inline, s_inline.median);
    row("copy_inline_twin [FLOOR]", s_twin, s_inline.median);
    row("copy_call        [SLOW]", s_call, s_inline.median);
    double bytes = double(NB) * ROWS * OUTER;
    std::printf("  effective copy bandwidth: call %.1f GiB/s   inline %.1f GiB/s\n", bytes / s_call.median / (1u << 30),
                bytes / s_inline.median / (1u << 30));
    return 0;
}
