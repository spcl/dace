/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct cavity_flow_legacy_state_t {

};

void __program_cavity_flow_legacy_internal(cavity_flow_legacy_state_t*__state, double * __restrict__ p, double * __restrict__ u, double * __restrict__ v, double dt, double dx, double dy, int64_t nit, int64_t nt, double nu, int64_t nx, int64_t ny, double rho)
{
    double *un;
    un = new double DACE_ALIGN(64)[(nx * ny)];
    double *vn;
    vn = new double DACE_ALIGN(64)[(nx * ny)];
    double *b;
    b = new double DACE_ALIGN(64)[(nx * ny)];
    double *pn;
    pn = new double DACE_ALIGN(64)[(nx * ny)];
    int64_t n;
    int64_t q;

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                    {
                        double __out;

                        ///////////////////
                        // Tasklet code (_numpy_full_)
                        __out = 0.0;
                        ///////////////////

                        un[((__i0 * nx) + __i1)] = __out;
                    }
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                    {
                        double __out;

                        ///////////////////
                        // Tasklet code (_numpy_full_)
                        __out = 0.0;
                        ///////////////////

                        vn[((__i0 * nx) + __i1)] = __out;
                    }
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                    {
                        double __out;

                        ///////////////////
                        // Tasklet code (_numpy_full_)
                        __out = 0.0;
                        ///////////////////

                        b[((__i0 * nx) + __i1)] = __out;
                    }
                }
            }
        }

    }
    for (n = 0; (n < nt); n = (n + 1)) {
        {
            double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp0;
            double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp1;
            double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp2;
            double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp4;
            double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp6;
            double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp7;
            double __tmp10;


            dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
            u, un, (nx * ny), 1);

            dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
            v, vn, (nx * ny), 1);
            {
                double __in2 = dt;
                double __out;

                ///////////////////
                // Tasklet code (_Div_)
                __out = (double(1) / __in2);
                ///////////////////

                tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp0 = __out;
            }
            {
                double __in2 = dx;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (double(2) * __in2);
                ///////////////////

                tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp1 = __out;
            }
            {
                double __in2 = dx;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (double(2) * __in2);
                ///////////////////

                tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp4 = __out;
            }
            {
                double __in2 = dx;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (double(2) * __in2);
                ///////////////////

                tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp7 = __out;
            }
            {
                double __in2 = dy;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (double(2) * __in2);
                ///////////////////

                tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp2 = __out;
            }
            {
                double __in2 = dy;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (double(2) * __in2);
                ///////////////////

                tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp6 = __out;
            }
            {
                double __in2 = dy;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (double(2) * __in2);
                ///////////////////

                __tmp10 = __out;
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (ny - 2); __i0 += 1) {
                    for (auto __i1 = 0; __i1 < (nx - 2); __i1 += 1) {
                        double __map_fusion_u_slice_minus_u_slice;
                        double __map_fusion_u_slice_u_slice_div_2_dx;
                        double __map_fusion_v_slice_minus_v_slice;
                        double __map_fusion_v_slice_v_slice_div_2_dy;
                        double __map_fusion_u_slice_u_slice_2_dx_plus_v_slice_v_slice_2_dy;
                        double __map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp3;
                        double __map_fusion_u_slice_minus_u_slice_0;
                        double __map_fusion_u_slice_u_slice_div_2_dx_0;
                        double __map_fusion_u_slice_u_slice_2_dx_pow_2;
                        double __map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp5;
                        double __map_fusion_u_slice_minus_u_slice_1;
                        double __map_fusion_u_slice_u_slice_div_2_dy;
                        double __map_fusion_v_slice_minus_v_slice_0;
                        double __map_fusion_u_slice_u_slice_2_dy_times_v_slice_v_slice;
                        double __map_fusion_u_slice_u_slice_2_dy_v_slice_v_slice_div_2_dx;
                        double __map_fusion___tmp8;
                        double __map_fusion___tmp9;
                        double __map_fusion_v_slice_minus_v_slice_1;
                        double __map_fusion_v_slice_v_slice_div_2_dy_0;
                        double __map_fusion_v_slice_v_slice_2_dy_pow_2;
                        double __map_fusion___tmp11;
                        {
                            double __in1 = v[((__i1 + (nx * (__i0 + 2))) + 1)];
                            double __in2 = v[(((__i0 * nx) + __i1) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_v_slice_minus_v_slice_1 = __out;
                        }
                        {
                            double __in1 = __map_fusion_v_slice_minus_v_slice_1;
                            double __in2 = __tmp10;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_v_slice_v_slice_div_2_dy_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_v_slice_v_slice_div_2_dy_0;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Pow_)
                            __out = (dace::math::ipow(__in1, 2));
                            ///////////////////

                            __map_fusion_v_slice_v_slice_2_dy_pow_2 = __out;
                        }
                        {
                            double __in1 = v[((__i1 + (nx * (__i0 + 1))) + 2)];
                            double __in2 = v[(__i1 + (nx * (__i0 + 1)))];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_v_slice_minus_v_slice_0 = __out;
                        }
                        {
                            double __in1 = v[((__i1 + (nx * (__i0 + 2))) + 1)];
                            double __in2 = v[(((__i0 * nx) + __i1) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_v_slice_minus_v_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_v_slice_minus_v_slice;
                            double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp2;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_v_slice_v_slice_div_2_dy = __out;
                        }
                        {
                            double __in1 = u[((__i1 + (nx * (__i0 + 2))) + 1)];
                            double __in2 = u[(((__i0 * nx) + __i1) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_u_slice_minus_u_slice_1 = __out;
                        }
                        {
                            double __in1 = __map_fusion_u_slice_minus_u_slice_1;
                            double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp6;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_u_slice_u_slice_div_2_dy = __out;
                        }
                        {
                            double __in1 = __map_fusion_u_slice_u_slice_div_2_dy;
                            double __in2 = __map_fusion_v_slice_minus_v_slice_0;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_u_slice_u_slice_2_dy_times_v_slice_v_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_u_slice_u_slice_2_dy_times_v_slice_v_slice;
                            double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp7;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_u_slice_u_slice_2_dy_v_slice_v_slice_div_2_dx = __out;
                        }
                        {
                            double __in2 = __map_fusion_u_slice_u_slice_2_dy_v_slice_v_slice_div_2_dx;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (double(2) * __in2);
                            ///////////////////

                            __map_fusion___tmp8 = __out;
                        }
                        {
                            double __in1 = u[((__i1 + (nx * (__i0 + 1))) + 2)];
                            double __in2 = u[(__i1 + (nx * (__i0 + 1)))];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_u_slice_minus_u_slice_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_u_slice_minus_u_slice_0;
                            double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp4;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_u_slice_u_slice_div_2_dx_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_u_slice_u_slice_div_2_dx_0;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Pow_)
                            __out = (dace::math::ipow(__in1, 2));
                            ///////////////////

                            __map_fusion_u_slice_u_slice_2_dx_pow_2 = __out;
                        }
                        {
                            double __in1 = u[((__i1 + (nx * (__i0 + 1))) + 2)];
                            double __in2 = u[(__i1 + (nx * (__i0 + 1)))];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_u_slice_minus_u_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_u_slice_minus_u_slice;
                            double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp1;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_u_slice_u_slice_div_2_dx = __out;
                        }
                        {
                            double __in1 = __map_fusion_u_slice_u_slice_div_2_dx;
                            double __in2 = __map_fusion_v_slice_v_slice_div_2_dy;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __map_fusion_u_slice_u_slice_2_dx_plus_v_slice_v_slice_2_dy = __out;
                        }
                        {
                            double __in2 = __map_fusion_u_slice_u_slice_2_dx_plus_v_slice_v_slice_2_dy;
                            double __in1 = tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp0;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp3 = __out;
                        }
                        {
                            double __in1 = __map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp3;
                            double __in2 = __map_fusion_u_slice_u_slice_2_dx_pow_2;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp5 = __out;
                        }
                        {
                            double __in1 = __map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp5;
                            double __in2 = __map_fusion___tmp8;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion___tmp9 = __out;
                        }
                        {
                            double __in1 = __map_fusion___tmp9;
                            double __in2 = __map_fusion_v_slice_v_slice_2_dy_pow_2;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion___tmp11 = __out;
                        }
                        {
                            double __in2 = __map_fusion___tmp11;
                            double __in1 = rho;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            b[((__i1 + (nx * (__i0 + 1))) + 1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (_numpy_full_)
                            __out = 0.0;
                            ///////////////////

                            pn[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {


            dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
            p, pn, (nx * ny), 1);

        }
        for (q = 0; (q < nit); q = (q + 1)) {
            {
                double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2;
                double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2;
                double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2_0;
                double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2_0;
                double dx_2_plus_dy_2;
                double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp0;
                double dx_pow_2_1;
                double dy_pow_2_1;
                double dx_2_times_dy_2;
                double dx_pow_2_2;
                double dy_pow_2_2;
                double dx_2_plus_dy_2_0;
                double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp1;
                double dx_2_dy_2_div_2_dx_2_dy_2;


                dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
                p, pn, (nx * ny), 1);
                {
                    double __in1 = dy;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Pow_)
                    __out = (dace::math::ipow(__in1, 2));
                    ///////////////////

                    tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2 = __out;
                }
                {
                    double __in1 = dy;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Pow_)
                    __out = (dace::math::ipow(__in1, 2));
                    ///////////////////

                    tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2_0 = __out;
                }
                {
                    double __in1 = dy;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Pow_)
                    __out = (dace::math::ipow(__in1, 2));
                    ///////////////////

                    dy_pow_2_1 = __out;
                }
                {
                    double __in1 = dy;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Pow_)
                    __out = (dace::math::ipow(__in1, 2));
                    ///////////////////

                    dy_pow_2_2 = __out;
                }
                {
                    double __in1 = dx;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Pow_)
                    __out = (dace::math::ipow(__in1, 2));
                    ///////////////////

                    tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2 = __out;
                }
                {
                    double __in1 = dx;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Pow_)
                    __out = (dace::math::ipow(__in1, 2));
                    ///////////////////

                    tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2_0 = __out;
                }
                {
                    double __in1 = tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2_0;
                    double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2_0;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Add_)
                    __out = (__in1 + __in2);
                    ///////////////////

                    dx_2_plus_dy_2 = __out;
                }
                {
                    double __in2 = dx_2_plus_dy_2;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Mult_)
                    __out = (double(2) * __in2);
                    ///////////////////

                    tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp0 = __out;
                }
                {
                    double __in1 = dx;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Pow_)
                    __out = (dace::math::ipow(__in1, 2));
                    ///////////////////

                    dx_pow_2_1 = __out;
                }
                {
                    double __in2 = dy_pow_2_1;
                    double __in1 = dx_pow_2_1;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Mult_)
                    __out = (__in1 * __in2);
                    ///////////////////

                    dx_2_times_dy_2 = __out;
                }
                {
                    double __in1 = dx;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Pow_)
                    __out = (dace::math::ipow(__in1, 2));
                    ///////////////////

                    dx_pow_2_2 = __out;
                }
                {
                    double __in1 = dx_pow_2_2;
                    double __in2 = dy_pow_2_2;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Add_)
                    __out = (__in1 + __in2);
                    ///////////////////

                    dx_2_plus_dy_2_0 = __out;
                }
                {
                    double __in2 = dx_2_plus_dy_2_0;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Mult_)
                    __out = (double(2) * __in2);
                    ///////////////////

                    tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp1 = __out;
                }
                {
                    double __in1 = dx_2_times_dy_2;
                    double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp1;
                    double __out;

                    ///////////////////
                    // Tasklet code (_Div_)
                    __out = (__in1 / __in2);
                    ///////////////////

                    dx_2_dy_2_div_2_dx_2_dy_2 = __out;
                }
                {
                    #pragma omp parallel for
                    for (auto __i0 = 0; __i0 < (ny - 2); __i0 += 1) {
                        for (auto __i1 = 0; __i1 < (nx - 2); __i1 += 1) {
                            double __map_fusion_pn_slice_plus_pn_slice;
                            double __map_fusion_pn_slice_pn_slice_times_dy_2;
                            double __map_fusion_pn_slice_plus_pn_slice_0;
                            double __map_fusion_pn_slice_pn_slice_times_dx_2;
                            double __map_fusion_pn_slice_pn_slice_dy_2_plus_pn_slice_pn_slice_dx_2;
                            double __map_fusion_pn_slice_pn_slice_dy_2_pn_slice_pn_slice_dx_2_div_2_dx_2_dy_2;
                            double __map_fusion_dx_2_dy_2_2_dx_2_dy_2_times_b_slice;
                            {
                                double __in1 = dx_2_dy_2_div_2_dx_2_dy_2;
                                double __in2 = b[((__i1 + (nx * (__i0 + 1))) + 1)];
                                double __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (__in1 * __in2);
                                ///////////////////

                                __map_fusion_dx_2_dy_2_2_dx_2_dy_2_times_b_slice = __out;
                            }
                            {
                                double __in1 = pn[((__i1 + (nx * (__i0 + 2))) + 1)];
                                double __in2 = pn[(((__i0 * nx) + __i1) + 1)];
                                double __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __map_fusion_pn_slice_plus_pn_slice_0 = __out;
                            }
                            {
                                double __in1 = __map_fusion_pn_slice_plus_pn_slice_0;
                                double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2;
                                double __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (__in1 * __in2);
                                ///////////////////

                                __map_fusion_pn_slice_pn_slice_times_dx_2 = __out;
                            }
                            {
                                double __in1 = pn[((__i1 + (nx * (__i0 + 1))) + 2)];
                                double __in2 = pn[(__i1 + (nx * (__i0 + 1)))];
                                double __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __map_fusion_pn_slice_plus_pn_slice = __out;
                            }
                            {
                                double __in1 = __map_fusion_pn_slice_plus_pn_slice;
                                double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2;
                                double __out;

                                ///////////////////
                                // Tasklet code (_Mult_)
                                __out = (__in1 * __in2);
                                ///////////////////

                                __map_fusion_pn_slice_pn_slice_times_dy_2 = __out;
                            }
                            {
                                double __in1 = __map_fusion_pn_slice_pn_slice_times_dy_2;
                                double __in2 = __map_fusion_pn_slice_pn_slice_times_dx_2;
                                double __out;

                                ///////////////////
                                // Tasklet code (_Add_)
                                __out = (__in1 + __in2);
                                ///////////////////

                                __map_fusion_pn_slice_pn_slice_dy_2_plus_pn_slice_pn_slice_dx_2 = __out;
                            }
                            {
                                double __in1 = __map_fusion_pn_slice_pn_slice_dy_2_plus_pn_slice_pn_slice_dx_2;
                                double __in2 = tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp0;
                                double __out;

                                ///////////////////
                                // Tasklet code (_Div_)
                                __out = (__in1 / __in2);
                                ///////////////////

                                __map_fusion_pn_slice_pn_slice_dy_2_pn_slice_pn_slice_dx_2_div_2_dx_2_dy_2 = __out;
                            }
                            {
                                double __in1 = __map_fusion_pn_slice_pn_slice_dy_2_pn_slice_pn_slice_dx_2_div_2_dx_2_dy_2;
                                double __in2 = __map_fusion_dx_2_dy_2_2_dx_2_dy_2_times_b_slice;
                                double __out;

                                ///////////////////
                                // Tasklet code (_Sub_)
                                __out = (__in1 - __in2);
                                ///////////////////

                                p[((__i1 + (nx * (__i0 + 1))) + 1)] = __out;
                            }
                        }
                    }
                }

                dace::CopyNDDynamic<double, 1, false, 1>::Dynamic::Copy(
                p + (nx - 2), p + (nx - 1), ny, nx, nx);

                dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
                p + nx, p, nx, 1);

                dace::CopyNDDynamic<double, 1, false, 1>::Dynamic::Copy(
                p + 1, p, ny, nx, nx);

            }
            {

                {
                    #pragma omp parallel for
                    for (auto __i0 = (ny - 1); __i0 < ny; __i0 += 1) {
                        for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                            {
                                double __out;

                                ///////////////////
                                // Tasklet code (assign_100_8)
                                __out = 0;
                                ///////////////////

                                p[((__i0 * nx) + __i1)] = __out;
                            }
                        }
                    }
                }

            }

        }
        {
            double __tmp0;
            double __tmp1;
            double dt_div_2_rho_dx;
            double dx_pow_2;
            double dt_div_dx_2;
            double dy_pow_2;
            double dt_div_dy_2;
            double __tmp4;
            double __tmp5;
            double dt_div_2_rho_dy;
            double dx_pow_2_0;
            double dt_div_dx_2_0;
            double dy_pow_2_0;
            double dt_div_dy_2_0;

            {
                double __in1 = dx;
                double __out;

                ///////////////////
                // Tasklet code (_Pow_)
                __out = (dace::math::ipow(__in1, 2));
                ///////////////////

                dx_pow_2_0 = __out;
            }
            {
                double __in2 = dx_pow_2_0;
                double __in1 = dt;
                double __out;

                ///////////////////
                // Tasklet code (_Div_)
                __out = (__in1 / __in2);
                ///////////////////

                dt_div_dx_2_0 = __out;
            }
            {
                double __in1 = dy;
                double __out;

                ///////////////////
                // Tasklet code (_Pow_)
                __out = (dace::math::ipow(__in1, 2));
                ///////////////////

                dy_pow_2_0 = __out;
            }
            {
                double __in2 = dy_pow_2_0;
                double __in1 = dt;
                double __out;

                ///////////////////
                // Tasklet code (_Div_)
                __out = (__in1 / __in2);
                ///////////////////

                dt_div_dy_2_0 = __out;
            }
            {
                double __in2 = rho;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (double(2) * __in2);
                ///////////////////

                __tmp4 = __out;
            }
            {
                double __in1 = __tmp4;
                double __in2 = dy;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (__in1 * __in2);
                ///////////////////

                __tmp5 = __out;
            }
            {
                double __in2 = __tmp5;
                double __in1 = dt;
                double __out;

                ///////////////////
                // Tasklet code (_Div_)
                __out = (__in1 / __in2);
                ///////////////////

                dt_div_2_rho_dy = __out;
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (ny - 2); __i0 += 1) {
                    for (auto __i1 = 0; __i1 < (nx - 2); __i1 += 1) {
                        double __map_fusion_un_slice_times_dt_0;
                        double __map_fusion_un_slice_dt_div_dx_0;
                        double __map_fusion_vn_slice_minus_vn_slice;
                        double __map_fusion_un_slice_dt_dx_times_vn_slice_vn_slice;
                        double __map_fusion_vn_slice_minus_un_slice_dt_dx_vn_slice_vn_slice;
                        double __map_fusion_vn_slice_times_dt_0;
                        double __map_fusion_vn_slice_dt_div_dy_0;
                        double __map_fusion_vn_slice_minus_vn_slice_0;
                        double __map_fusion_vn_slice_dt_dy_times_vn_slice_vn_slice;
                        double __map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_minus_vn_slice_dt_dy_vn_slice_vn_slice;
                        double __map_fusion_p_slice_minus_p_slice_0;
                        double __map_fusion_dt_2_rho_dy_times_p_slice_p_slice;
                        double __map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_vn_slice_dt_dy_vn_slice_vn_slice_minus_dt_2_rho_dy_p_slice_p_slice;
                        double __map_fusion___tmp6;
                        double __map_fusion_vn_slice_minus_2_vn_slice;
                        double __map_fusion_vn_slice_2_vn_slice_plus_vn_slice;
                        double __map_fusion_dt_dx_2_times_vn_slice_2_vn_slice_vn_slice;
                        double __map_fusion___tmp7;
                        double __map_fusion_vn_slice_minus_2_vn_slice_0;
                        double __map_fusion_vn_slice_2_vn_slice_plus_vn_slice_0;
                        double __map_fusion_dt_dy_2_times_vn_slice_2_vn_slice_vn_slice;
                        double __map_fusion_dt_dx_2_vn_slice_2_vn_slice_vn_slice_plus_dt_dy_2_vn_slice_2_vn_slice_vn_slice;
                        double __map_fusion_nu_times_dt_dx_2_vn_slice_2_vn_slice_vn_slice_dt_dy_2_vn_slice_2_vn_slice_vn_slice;
                        {
                            double __in2 = vn[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (double(2) * __in2);
                            ///////////////////

                            __map_fusion___tmp7 = __out;
                        }
                        {
                            double __in2 = __map_fusion___tmp7;
                            double __in1 = vn[((__i1 + (nx * (__i0 + 2))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_vn_slice_minus_2_vn_slice_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_minus_2_vn_slice_0;
                            double __in2 = vn[(((__i0 * nx) + __i1) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __map_fusion_vn_slice_2_vn_slice_plus_vn_slice_0 = __out;
                        }
                        {
                            double __in2 = __map_fusion_vn_slice_2_vn_slice_plus_vn_slice_0;
                            double __in1 = dt_div_dy_2_0;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_dt_dy_2_times_vn_slice_2_vn_slice_vn_slice = __out;
                        }
                        {
                            double __in2 = vn[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (double(2) * __in2);
                            ///////////////////

                            __map_fusion___tmp6 = __out;
                        }
                        {
                            double __in2 = __map_fusion___tmp6;
                            double __in1 = vn[((__i1 + (nx * (__i0 + 1))) + 2)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_vn_slice_minus_2_vn_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_minus_2_vn_slice;
                            double __in2 = vn[(__i1 + (nx * (__i0 + 1)))];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __map_fusion_vn_slice_2_vn_slice_plus_vn_slice = __out;
                        }
                        {
                            double __in2 = __map_fusion_vn_slice_2_vn_slice_plus_vn_slice;
                            double __in1 = dt_div_dx_2_0;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_dt_dx_2_times_vn_slice_2_vn_slice_vn_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_dt_dx_2_times_vn_slice_2_vn_slice_vn_slice;
                            double __in2 = __map_fusion_dt_dy_2_times_vn_slice_2_vn_slice_vn_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __map_fusion_dt_dx_2_vn_slice_2_vn_slice_vn_slice_plus_dt_dy_2_vn_slice_2_vn_slice_vn_slice = __out;
                        }
                        {
                            double __in2 = __map_fusion_dt_dx_2_vn_slice_2_vn_slice_vn_slice_plus_dt_dy_2_vn_slice_2_vn_slice_vn_slice;
                            double __in1 = nu;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_nu_times_dt_dx_2_vn_slice_2_vn_slice_vn_slice_dt_dy_2_vn_slice_2_vn_slice_vn_slice = __out;
                        }
                        {
                            double __in1 = p[((__i1 + (nx * (__i0 + 2))) + 1)];
                            double __in2 = p[(((__i0 * nx) + __i1) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_p_slice_minus_p_slice_0 = __out;
                        }
                        {
                            double __in2 = __map_fusion_p_slice_minus_p_slice_0;
                            double __in1 = dt_div_2_rho_dy;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_dt_2_rho_dy_times_p_slice_p_slice = __out;
                        }
                        {
                            double __in1 = vn[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __in2 = vn[(((__i0 * nx) + __i1) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_vn_slice_minus_vn_slice_0 = __out;
                        }
                        {
                            double __in1 = vn[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __in2 = dt;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_vn_slice_times_dt_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_times_dt_0;
                            double __in2 = dy;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_vn_slice_dt_div_dy_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_dt_div_dy_0;
                            double __in2 = __map_fusion_vn_slice_minus_vn_slice_0;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_vn_slice_dt_dy_times_vn_slice_vn_slice = __out;
                        }
                        {
                            double __in1 = vn[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __in2 = vn[(__i1 + (nx * (__i0 + 1)))];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_vn_slice_minus_vn_slice = __out;
                        }
                        {
                            double __in2 = dt;
                            double __in1 = un[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_un_slice_times_dt_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_times_dt_0;
                            double __in2 = dx;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_un_slice_dt_div_dx_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_dt_div_dx_0;
                            double __in2 = __map_fusion_vn_slice_minus_vn_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_un_slice_dt_dx_times_vn_slice_vn_slice = __out;
                        }
                        {
                            double __in2 = __map_fusion_un_slice_dt_dx_times_vn_slice_vn_slice;
                            double __in1 = vn[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_vn_slice_minus_un_slice_dt_dx_vn_slice_vn_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_minus_un_slice_dt_dx_vn_slice_vn_slice;
                            double __in2 = __map_fusion_vn_slice_dt_dy_times_vn_slice_vn_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_minus_vn_slice_dt_dy_vn_slice_vn_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_minus_vn_slice_dt_dy_vn_slice_vn_slice;
                            double __in2 = __map_fusion_dt_2_rho_dy_times_p_slice_p_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_vn_slice_dt_dy_vn_slice_vn_slice_minus_dt_2_rho_dy_p_slice_p_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_vn_slice_dt_dy_vn_slice_vn_slice_minus_dt_2_rho_dy_p_slice_p_slice;
                            double __in2 = __map_fusion_nu_times_dt_dx_2_vn_slice_2_vn_slice_vn_slice_dt_dy_2_vn_slice_2_vn_slice_vn_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            v[((__i1 + (nx * (__i0 + 1))) + 1)] = __out;
                        }
                    }
                }
            }
            {
                double __in1 = dx;
                double __out;

                ///////////////////
                // Tasklet code (_Pow_)
                __out = (dace::math::ipow(__in1, 2));
                ///////////////////

                dx_pow_2 = __out;
            }
            {
                double __in2 = dx_pow_2;
                double __in1 = dt;
                double __out;

                ///////////////////
                // Tasklet code (_Div_)
                __out = (__in1 / __in2);
                ///////////////////

                dt_div_dx_2 = __out;
            }
            {
                double __in1 = dy;
                double __out;

                ///////////////////
                // Tasklet code (_Pow_)
                __out = (dace::math::ipow(__in1, 2));
                ///////////////////

                dy_pow_2 = __out;
            }
            {
                double __in2 = dy_pow_2;
                double __in1 = dt;
                double __out;

                ///////////////////
                // Tasklet code (_Div_)
                __out = (__in1 / __in2);
                ///////////////////

                dt_div_dy_2 = __out;
            }
            {
                double __in2 = rho;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (double(2) * __in2);
                ///////////////////

                __tmp0 = __out;
            }
            {
                double __in1 = __tmp0;
                double __in2 = dx;
                double __out;

                ///////////////////
                // Tasklet code (_Mult_)
                __out = (__in1 * __in2);
                ///////////////////

                __tmp1 = __out;
            }
            {
                double __in2 = __tmp1;
                double __in1 = dt;
                double __out;

                ///////////////////
                // Tasklet code (_Div_)
                __out = (__in1 / __in2);
                ///////////////////

                dt_div_2_rho_dx = __out;
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (ny - 2); __i0 += 1) {
                    for (auto __i1 = 0; __i1 < (nx - 2); __i1 += 1) {
                        double __map_fusion_un_slice_times_dt;
                        double __map_fusion_un_slice_dt_div_dx;
                        double __map_fusion_un_slice_minus_un_slice;
                        double __map_fusion_un_slice_dt_dx_times_un_slice_un_slice;
                        double __map_fusion_un_slice_minus_un_slice_dt_dx_un_slice_un_slice;
                        double __map_fusion_vn_slice_times_dt;
                        double __map_fusion_vn_slice_dt_div_dy;
                        double __map_fusion_un_slice_minus_un_slice_0;
                        double __map_fusion_vn_slice_dt_dy_times_un_slice_un_slice;
                        double __map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_minus_vn_slice_dt_dy_un_slice_un_slice;
                        double __map_fusion_p_slice_minus_p_slice;
                        double __map_fusion_dt_2_rho_dx_times_p_slice_p_slice;
                        double __map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_vn_slice_dt_dy_un_slice_un_slice_minus_dt_2_rho_dx_p_slice_p_slice;
                        double __map_fusion___tmp2;
                        double __map_fusion_un_slice_minus_2_un_slice;
                        double __map_fusion_un_slice_2_un_slice_plus_un_slice;
                        double __map_fusion_dt_dx_2_times_un_slice_2_un_slice_un_slice;
                        double __map_fusion___tmp3;
                        double __map_fusion_un_slice_minus_2_un_slice_0;
                        double __map_fusion_un_slice_2_un_slice_plus_un_slice_0;
                        double __map_fusion_dt_dy_2_times_un_slice_2_un_slice_un_slice;
                        double __map_fusion_dt_dx_2_un_slice_2_un_slice_un_slice_plus_dt_dy_2_un_slice_2_un_slice_un_slice;
                        double __map_fusion_nu_times_dt_dx_2_un_slice_2_un_slice_un_slice_dt_dy_2_un_slice_2_un_slice_un_slice;
                        {
                            double __in2 = un[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (double(2) * __in2);
                            ///////////////////

                            __map_fusion___tmp3 = __out;
                        }
                        {
                            double __in2 = __map_fusion___tmp3;
                            double __in1 = un[((__i1 + (nx * (__i0 + 2))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_un_slice_minus_2_un_slice_0 = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_minus_2_un_slice_0;
                            double __in2 = un[(((__i0 * nx) + __i1) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __map_fusion_un_slice_2_un_slice_plus_un_slice_0 = __out;
                        }
                        {
                            double __in2 = __map_fusion_un_slice_2_un_slice_plus_un_slice_0;
                            double __in1 = dt_div_dy_2;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_dt_dy_2_times_un_slice_2_un_slice_un_slice = __out;
                        }
                        {
                            double __in2 = un[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (double(2) * __in2);
                            ///////////////////

                            __map_fusion___tmp2 = __out;
                        }
                        {
                            double __in2 = __map_fusion___tmp2;
                            double __in1 = un[((__i1 + (nx * (__i0 + 1))) + 2)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_un_slice_minus_2_un_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_minus_2_un_slice;
                            double __in2 = un[(__i1 + (nx * (__i0 + 1)))];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __map_fusion_un_slice_2_un_slice_plus_un_slice = __out;
                        }
                        {
                            double __in2 = __map_fusion_un_slice_2_un_slice_plus_un_slice;
                            double __in1 = dt_div_dx_2;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_dt_dx_2_times_un_slice_2_un_slice_un_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_dt_dx_2_times_un_slice_2_un_slice_un_slice;
                            double __in2 = __map_fusion_dt_dy_2_times_un_slice_2_un_slice_un_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __map_fusion_dt_dx_2_un_slice_2_un_slice_un_slice_plus_dt_dy_2_un_slice_2_un_slice_un_slice = __out;
                        }
                        {
                            double __in2 = __map_fusion_dt_dx_2_un_slice_2_un_slice_un_slice_plus_dt_dy_2_un_slice_2_un_slice_un_slice;
                            double __in1 = nu;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_nu_times_dt_dx_2_un_slice_2_un_slice_un_slice_dt_dy_2_un_slice_2_un_slice_un_slice = __out;
                        }
                        {
                            double __in1 = p[((__i1 + (nx * (__i0 + 1))) + 2)];
                            double __in2 = p[(__i1 + (nx * (__i0 + 1)))];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_p_slice_minus_p_slice = __out;
                        }
                        {
                            double __in2 = __map_fusion_p_slice_minus_p_slice;
                            double __in1 = dt_div_2_rho_dx;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_dt_2_rho_dx_times_p_slice_p_slice = __out;
                        }
                        {
                            double __in1 = un[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __in2 = un[(((__i0 * nx) + __i1) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_un_slice_minus_un_slice_0 = __out;
                        }
                        {
                            double __in1 = un[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __in2 = un[(__i1 + (nx * (__i0 + 1)))];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_un_slice_minus_un_slice = __out;
                        }
                        {
                            double __in1 = un[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __in2 = dt;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_un_slice_times_dt = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_times_dt;
                            double __in2 = dx;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_un_slice_dt_div_dx = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_dt_div_dx;
                            double __in2 = __map_fusion_un_slice_minus_un_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_un_slice_dt_dx_times_un_slice_un_slice = __out;
                        }
                        {
                            double __in2 = __map_fusion_un_slice_dt_dx_times_un_slice_un_slice;
                            double __in1 = un[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_un_slice_minus_un_slice_dt_dx_un_slice_un_slice = __out;
                        }
                        {
                            double __in1 = vn[((__i1 + (nx * (__i0 + 1))) + 1)];
                            double __in2 = dt;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_vn_slice_times_dt = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_times_dt;
                            double __in2 = dy;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Div_)
                            __out = (__in1 / __in2);
                            ///////////////////

                            __map_fusion_vn_slice_dt_div_dy = __out;
                        }
                        {
                            double __in1 = __map_fusion_vn_slice_dt_div_dy;
                            double __in2 = __map_fusion_un_slice_minus_un_slice_0;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Mult_)
                            __out = (__in1 * __in2);
                            ///////////////////

                            __map_fusion_vn_slice_dt_dy_times_un_slice_un_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_minus_un_slice_dt_dx_un_slice_un_slice;
                            double __in2 = __map_fusion_vn_slice_dt_dy_times_un_slice_un_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_minus_vn_slice_dt_dy_un_slice_un_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_minus_vn_slice_dt_dy_un_slice_un_slice;
                            double __in2 = __map_fusion_dt_2_rho_dx_times_p_slice_p_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Sub_)
                            __out = (__in1 - __in2);
                            ///////////////////

                            __map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_vn_slice_dt_dy_un_slice_un_slice_minus_dt_2_rho_dx_p_slice_p_slice = __out;
                        }
                        {
                            double __in1 = __map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_vn_slice_dt_dy_un_slice_un_slice_minus_dt_2_rho_dx_p_slice_p_slice;
                            double __in2 = __map_fusion_nu_times_dt_dx_2_un_slice_2_un_slice_un_slice_dt_dy_2_un_slice_2_un_slice_un_slice;
                            double __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            u[((__i1 + (nx * (__i0 + 1))) + 1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < 1; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (assign_124_8)
                            __out = 0;
                            ///////////////////

                            u[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < 1; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (assign_125_8)
                            __out = 0;
                            ///////////////////

                            u[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = (nx - 1); __i1 < nx; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (assign_126_8)
                            __out = 0;
                            ///////////////////

                            u[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = (ny - 1); __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (assign_127_8)
                            __out = 1;
                            ///////////////////

                            u[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < 1; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (assign_128_8)
                            __out = 0;
                            ///////////////////

                            v[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = (ny - 1); __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (assign_129_8)
                            __out = 0;
                            ///////////////////

                            v[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < 1; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (assign_130_8)
                            __out = 0;
                            ///////////////////

                            v[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = (nx - 1); __i1 < nx; __i1 += 1) {
                        {
                            double __out;

                            ///////////////////
                            // Tasklet code (assign_131_8)
                            __out = 0;
                            ///////////////////

                            v[((__i0 * nx) + __i1)] = __out;
                        }
                    }
                }
            }

        }

    }
    delete[] un;
    delete[] vn;
    delete[] b;
    delete[] pn;
}

DACE_EXPORTED void __program_cavity_flow_legacy(cavity_flow_legacy_state_t *__state, double * __restrict__ p, double * __restrict__ u, double * __restrict__ v, double dt, double dx, double dy, int64_t nit, int64_t nt, double nu, int64_t nx, int64_t ny, double rho)
{
    __program_cavity_flow_legacy_internal(__state, p, u, v, dt, dx, dy, nit, nt, nu, nx, ny, rho);
}

DACE_EXPORTED cavity_flow_legacy_state_t *__dace_init_cavity_flow_legacy(int64_t nit, int64_t nx, int64_t ny)
{

    int __result = 0;
    cavity_flow_legacy_state_t *__state = new cavity_flow_legacy_state_t;

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_cavity_flow_legacy(cavity_flow_legacy_state_t *__state)
{

    int __err = 0;
    delete __state;
    return __err;
}

#include <cstdlib>
#include "../include/cavity_flow_legacy.h"

int main(int argc, char **argv) {
    cavity_flow_legacyHandle_t handle;
    int err;
    double dt = 42;
    double dx = 42;
    double dy = 42;
    int64_t nit = 42;
    int64_t nt = 42;
    double nu = 42;
    int64_t nx = 42;
    int64_t ny = 42;
    double rho = 42;
    double * __restrict__ p = (double*) calloc((nx * ny), sizeof(double));
    double * __restrict__ u = (double*) calloc((nx * ny), sizeof(double));
    double * __restrict__ v = (double*) calloc((nx * ny), sizeof(double));


    handle = __dace_init_cavity_flow_legacy(nit, nx, ny);
    __program_cavity_flow_legacy(handle, p, u, v, dt, dx, dy, nit, nt, nu, nx, ny, rho);
    err = __dace_exit_cavity_flow_legacy(handle);

    free(p);
    free(u);
    free(v);


    return err;
}
