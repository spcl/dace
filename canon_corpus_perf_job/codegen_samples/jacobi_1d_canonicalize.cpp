// DaCe-generated CPU code -- dace-canon-gcc, poly:jacobi_1d @ paper (tsteps=100, N=32000)
// codegen: compiler.cpu.implementation=None
// Kept as a representative of the emitted shape: per-iteration one-element stack
// arrays (double x[1]) alongside plain scalars, inside "#pragma omp parallel for".
// Measured cost of those len-1 arrays: negligible (see memory note); the LLVM arms'
// slowness was the libgomp+libomp co-residency, not this shape.

/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_state_t {

};

static DACE_HDFI constexpr int64_t A_idx(int64_t __d0) { return __d0; }
static DACE_HDFI constexpr int64_t B_idx(int64_t __d0) { return __d0; }
void __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_internal(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_state_t*__state, double * __restrict__ A, double * __restrict__ B, int N, int tsteps)
{
    int64_t _loop_it_0;

    {

        {  // check_assumption_0
            if ((N < 0)) { std::abort(); }
        }
        {  // check_assumption_1
            if ((tsteps < 0)) { std::abort(); }
        }

    }

    for (_loop_it_0 = 1; (_loop_it_0 < tsteps); _loop_it_0 = (_loop_it_0 + 1)) {

        #pragma omp parallel for
        for (auto _loop_it_1 = 0; _loop_it_1 < (N - 2); _loop_it_1 += 1) {
            const double __map_fusion_A_slice_plus_A_slice = (A[A_idx(_loop_it_1)] + A[A_idx((_loop_it_1 + 1))]);  // _Add_
            const double __map_fusion_A_slice_A_slice_plus_A_slice = (__map_fusion_A_slice_plus_A_slice + A[A_idx((_loop_it_1 + 2))]);  // _Add_
            B[B_idx((_loop_it_1 + 1))] = (0.33333 * __map_fusion_A_slice_A_slice_plus_A_slice);  // _Mult_
        }
        #pragma omp parallel for
        for (auto _loop_it_4 = 0; _loop_it_4 < (N - 2); _loop_it_4 += 1) {
            const double __map_fusion_B_slice_plus_B_slice = (B[B_idx(_loop_it_4)] + B[B_idx((_loop_it_4 + 1))]);  // _Add_
            const double __map_fusion_B_slice_B_slice_plus_B_slice = (__map_fusion_B_slice_plus_B_slice + B[B_idx((_loop_it_4 + 2))]);  // _Add_
            A[A_idx((_loop_it_4 + 1))] = (0.33333 * __map_fusion_B_slice_B_slice_plus_B_slice);  // _Mult_
        }

    }

}

DACE_EXPORTED void __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_state_t *__state, double * __restrict__ A, double * __restrict__ B, int N, int tsteps)
{
    __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_internal(__state, A, B, N, tsteps);
}

DACE_EXPORTED jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_state_t *__dace_init_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(int N, int tsteps)
{

    int __result = 0;
    jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_state_t *__state = new jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_state_t;

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc_state_t *__state)
{

    int __err = 0;
    delete __state;
    return __err;
}

#include <dace/dace.h>
typedef void * jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongccHandle_t;
extern "C" jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongccHandle_t __dace_init_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(int N, int tsteps);
extern "C" int __dace_exit_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongccHandle_t handle);
extern "C" void __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongccHandle_t handle, double * __restrict__ A, double * __restrict__ B, int N, int tsteps);

#include <cstdlib>
#include "../include/jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc.h"

int main(int argc, char **argv) {
    jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongccHandle_t handle;
    int err;
    int N = 42;
    int tsteps = 42;
    double * __restrict__ A = (double*) calloc(N, sizeof(double));
    double * __restrict__ B = (double*) calloc(N, sizeof(double));


    handle = __dace_init_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(N, tsteps);
    __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(handle, A, B, N, tsteps);
    err = __dace_exit_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_dacecanongcc(handle);

    free(A);
    free(B);


    return err;
}
