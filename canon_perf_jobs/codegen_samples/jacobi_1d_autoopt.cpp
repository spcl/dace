// DaCe-generated CPU code -- dace-autoopt-gcc, poly:jacobi_1d @ paper (tsteps=100, N=32000)
// codegen: compiler.cpu.implementation=None
// Kept as a representative of the emitted shape: per-iteration one-element stack
// arrays (double x[1]) alongside plain scalars, inside "#pragma omp parallel for".
// Measured cost of those len-1 arrays: negligible (see memory note); the LLVM arms'
// slowness was the libgomp+libomp co-residency, not this shape.

/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_state_t {

};

void __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_internal(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_state_t*__state, double * __restrict__ A, double * __restrict__ B, int N, int tsteps)
{
    int64_t t;


    for (t = 1; (t < tsteps); t = (t + 1)) {
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (N - 2); __i0 += 1) {
                    double __map_fusion_A_slice_plus_A_slice[1];
                    double __map_fusion_A_slice_A_slice_plus_A_slice[1];
                    {
                        double __in1 = A[__i0];
                        double __in2 = A[(__i0 + 1)];
                        double __out;

                        ///////////////////
                        // Tasklet code (_Add_)
                        __out = (__in1 + __in2);
                        ///////////////////

                        __map_fusion_A_slice_plus_A_slice[0] = __out;
                    }
                    {
                        double __in1 = __map_fusion_A_slice_plus_A_slice[0];
                        double __in2 = A[(__i0 + 2)];
                        double __out;

                        ///////////////////
                        // Tasklet code (_Add_)
                        __out = (__in1 + __in2);
                        ///////////////////

                        __map_fusion_A_slice_A_slice_plus_A_slice[0] = __out;
                    }
                    {
                        double __in2 = __map_fusion_A_slice_A_slice_plus_A_slice[0];
                        double __out;

                        ///////////////////
                        // Tasklet code (_Mult_)
                        __out = (0.33333 * __in2);
                        ///////////////////

                        B[(__i0 + 1)] = __out;
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (N - 2); __i0 += 1) {
                    double __map_fusion_B_slice_plus_B_slice[1];
                    double __map_fusion_B_slice_B_slice_plus_B_slice[1];
                    {
                        double __in1 = B[__i0];
                        double __in2 = B[(__i0 + 1)];
                        double __out;

                        ///////////////////
                        // Tasklet code (_Add_)
                        __out = (__in1 + __in2);
                        ///////////////////

                        __map_fusion_B_slice_plus_B_slice[0] = __out;
                    }
                    {
                        double __in1 = __map_fusion_B_slice_plus_B_slice[0];
                        double __in2 = B[(__i0 + 2)];
                        double __out;

                        ///////////////////
                        // Tasklet code (_Add_)
                        __out = (__in1 + __in2);
                        ///////////////////

                        __map_fusion_B_slice_B_slice_plus_B_slice[0] = __out;
                    }
                    {
                        double __in2 = __map_fusion_B_slice_B_slice_plus_B_slice[0];
                        double __out;

                        ///////////////////
                        // Tasklet code (_Mult_)
                        __out = (0.33333 * __in2);
                        ///////////////////

                        A[(__i0 + 1)] = __out;
                    }
                }
            }

        }

    }

}

DACE_EXPORTED void __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_state_t *__state, double * __restrict__ A, double * __restrict__ B, int N, int tsteps)
{
    __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_internal(__state, A, B, N, tsteps);
}

DACE_EXPORTED jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_state_t *__dace_init_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(int N, int tsteps)
{

    int __result = 0;
    jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_state_t *__state = new jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_state_t;

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc_state_t *__state)
{

    int __err = 0;
    delete __state;
    return __err;
}

#include <dace/dace.h>
typedef void * jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgccHandle_t;
extern "C" jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgccHandle_t __dace_init_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(int N, int tsteps);
extern "C" int __dace_exit_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgccHandle_t handle);
extern "C" void __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgccHandle_t handle, double * __restrict__ A, double * __restrict__ B, int N, int tsteps);

#include <cstdlib>
#include "../include/jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc.h"

int main(int argc, char **argv) {
    jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgccHandle_t handle;
    int err;
    int N = 42;
    int tsteps = 42;
    double * __restrict__ A = (double*) calloc(N, sizeof(double));
    double * __restrict__ B = (double*) calloc(N, sizeof(double));


    handle = __dace_init_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(N, tsteps);
    __program_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(handle, A, B, N, tsteps);
    err = __dace_exit_jacobi_1d_tests_corpus_polybench_stencils_jacobi_1d_jacobi1d_daceautooptgcc(handle);

    free(A);
    free(B);


    return err;
}
