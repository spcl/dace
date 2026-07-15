/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct cavity_flow_experimental_state_t {

};

static DACE_HDFI constexpr long long un_size(long long nx, long long ny) { return (nx * ny); }
static DACE_HDFI constexpr long long vn_size(long long nx, long long ny) { return (nx * ny); }
static DACE_HDFI constexpr long long b_size(long long nx, long long ny) { return (nx * ny); }
static DACE_HDFI constexpr long long pn_size(long long nx, long long ny) { return (nx * ny); }
static DACE_HDFI constexpr long long un_idx(long long __d0, long long __d1, long long nx) { return ((__d0 * nx) + __d1); }
static DACE_HDFI constexpr long long vn_idx(long long __d0, long long __d1, long long nx) { return ((__d0 * nx) + __d1); }
static DACE_HDFI constexpr long long b_idx(long long __d0, long long __d1, long long nx) { return ((__d0 * nx) + __d1); }
static DACE_HDFI constexpr long long v_idx(long long __d0, long long __d1, long long nx) { return ((__d0 * nx) + __d1); }
static DACE_HDFI constexpr long long u_idx(long long __d0, long long __d1, long long nx) { return ((__d0 * nx) + __d1); }
static DACE_HDFI constexpr long long pn_idx(long long __d0, long long __d1, long long nx) { return ((__d0 * nx) + __d1); }
static DACE_HDFI constexpr long long p_idx(long long __d0, long long __d1, long long nx) { return ((__d0 * nx) + __d1); }
void __program_cavity_flow_experimental_internal(cavity_flow_experimental_state_t*__state, double * __restrict__ p, double * __restrict__ u, double * __restrict__ v, double dt, double dx, double dy, int64_t nit, int64_t nt, double nu, int64_t nx, int64_t ny, double rho)
{
    double *un;
    un = dace::aligned_alloc<double>(un_size(nx, ny), 64);
    double *vn;
    vn = dace::aligned_alloc<double>(vn_size(nx, ny), 64);
    double *b;
    b = dace::aligned_alloc<double>(b_size(nx, ny), 64);
    double *pn;
    pn = dace::aligned_alloc<double>(pn_size(nx, ny), 64);
    int64_t n;
    int64_t q;

    {

        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                    un[un_idx(__i0, __i1, nx)] = 0.0;  // _numpy_full_
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                    vn[vn_idx(__i0, __i1, nx)] = 0.0;  // _numpy_full_
                }
            }
        }
        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                    b[b_idx(__i0, __i1, nx)] = 0.0;  // _numpy_full_
                }
            }
        }

    }
    for (n = 0; (n < nt); n = (n + 1)) {
        {


            dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
            u, un, (nx * ny), 1);

            dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
            v, vn, (nx * ny), 1);
            const double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp0 = (double(1) / dt);  // _Div_
            const double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp1 = (double(2) * dx);  // _Mult_
            const double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp4 = (double(2) * dx);  // _Mult_
            const double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp7 = (double(2) * dx);  // _Mult_
            const double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp2 = (double(2) * dy);  // _Mult_
            const double tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp6 = (double(2) * dy);  // _Mult_
            const double __tmp10 = (double(2) * dy);  // _Mult_
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (ny - 2); __i0 += 1) {
                    for (auto __i1 = 0; __i1 < (nx - 2); __i1 += 1) {
                        const double __map_fusion_v_slice_minus_v_slice_1 = (v[v_idx((__i0 + 2), (__i1 + 1), nx)] - v[v_idx(__i0, (__i1 + 1), nx)]);  // _Sub_
                        const double __map_fusion_v_slice_v_slice_div_2_dy_0 = (__map_fusion_v_slice_minus_v_slice_1 / __tmp10);  // _Div_
                        const double __map_fusion_v_slice_v_slice_2_dy_pow_2 = (dace::math::ipow(__map_fusion_v_slice_v_slice_div_2_dy_0, 2));  // _Pow_
                        const double __map_fusion_v_slice_minus_v_slice_0 = (v[v_idx((__i0 + 1), (__i1 + 2), nx)] - v[v_idx((__i0 + 1), __i1, nx)]);  // _Sub_
                        const double __map_fusion_v_slice_minus_v_slice = (v[v_idx((__i0 + 2), (__i1 + 1), nx)] - v[v_idx(__i0, (__i1 + 1), nx)]);  // _Sub_
                        const double __map_fusion_v_slice_v_slice_div_2_dy = (__map_fusion_v_slice_minus_v_slice / tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp2);  // _Div_
                        const double __map_fusion_u_slice_minus_u_slice_1 = (u[u_idx((__i0 + 2), (__i1 + 1), nx)] - u[u_idx(__i0, (__i1 + 1), nx)]);  // _Sub_
                        const double __map_fusion_u_slice_u_slice_div_2_dy = (__map_fusion_u_slice_minus_u_slice_1 / tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp6);  // _Div_
                        const double __map_fusion_u_slice_u_slice_2_dy_times_v_slice_v_slice = (__map_fusion_u_slice_u_slice_div_2_dy * __map_fusion_v_slice_minus_v_slice_0);  // _Mult_
                        const double __map_fusion_u_slice_u_slice_2_dy_v_slice_v_slice_div_2_dx = (__map_fusion_u_slice_u_slice_2_dy_times_v_slice_v_slice / tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp7);  // _Div_
                        const double __map_fusion___tmp8 = (double(2) * __map_fusion_u_slice_u_slice_2_dy_v_slice_v_slice_div_2_dx);  // _Mult_
                        const double __map_fusion_u_slice_minus_u_slice_0 = (u[u_idx((__i0 + 1), (__i1 + 2), nx)] - u[u_idx((__i0 + 1), __i1, nx)]);  // _Sub_
                        const double __map_fusion_u_slice_u_slice_div_2_dx_0 = (__map_fusion_u_slice_minus_u_slice_0 / tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp4);  // _Div_
                        const double __map_fusion_u_slice_u_slice_2_dx_pow_2 = (dace::math::ipow(__map_fusion_u_slice_u_slice_div_2_dx_0, 2));  // _Pow_
                        const double __map_fusion_u_slice_minus_u_slice = (u[u_idx((__i0 + 1), (__i1 + 2), nx)] - u[u_idx((__i0 + 1), __i1, nx)]);  // _Sub_
                        const double __map_fusion_u_slice_u_slice_div_2_dx = (__map_fusion_u_slice_minus_u_slice / tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp1);  // _Div_
                        const double __map_fusion_u_slice_u_slice_2_dx_plus_v_slice_v_slice_2_dy = (__map_fusion_u_slice_u_slice_div_2_dx + __map_fusion_v_slice_v_slice_div_2_dy);  // _Add_
                        const double __map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp3 = (tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp0 * __map_fusion_u_slice_u_slice_2_dx_plus_v_slice_v_slice_2_dy);  // _Mult_
                        const double __map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp5 = (__map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp3 - __map_fusion_u_slice_u_slice_2_dx_pow_2);  // _Sub_
                        const double __map_fusion___tmp9 = (__map_fusion_tests_corpus_npbench_structured_grids_cavity_flow_build_up_b___tmp5 - __map_fusion___tmp8);  // _Sub_
                        const double __map_fusion___tmp11 = (__map_fusion___tmp9 - __map_fusion_v_slice_v_slice_2_dy_pow_2);  // _Sub_
                        b[b_idx((__i0 + 1), (__i1 + 1), nx)] = (rho * __map_fusion___tmp11);  // _Mult_
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        pn[pn_idx(__i0, __i1, nx)] = 0.0;  // _numpy_full_
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


                dace::CopyNDDynamic<double, 1, false, 1>::template ConstDst<1>::Copy(
                p, pn, (nx * ny), 1);
                const double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2 = (dace::math::ipow(dy, 2));  // _Pow_
                const double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2_0 = (dace::math::ipow(dy, 2));  // _Pow_
                const double dy_pow_2_1 = (dace::math::ipow(dy, 2));  // _Pow_
                const double dy_pow_2_2 = (dace::math::ipow(dy, 2));  // _Pow_
                const double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2 = (dace::math::ipow(dx, 2));  // _Pow_
                const double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2_0 = (dace::math::ipow(dx, 2));  // _Pow_
                const double dx_2_plus_dy_2 = (tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2_0 + tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2_0);  // _Add_
                const double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp0 = (double(2) * dx_2_plus_dy_2);  // _Mult_
                const double dx_pow_2_1 = (dace::math::ipow(dx, 2));  // _Pow_
                const double dx_2_times_dy_2 = (dx_pow_2_1 * dy_pow_2_1);  // _Mult_
                const double dx_pow_2_2 = (dace::math::ipow(dx, 2));  // _Pow_
                const double dx_2_plus_dy_2_0 = (dx_pow_2_2 + dy_pow_2_2);  // _Add_
                const double tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp1 = (double(2) * dx_2_plus_dy_2_0);  // _Mult_
                const double dx_2_dy_2_div_2_dx_2_dy_2 = (dx_2_times_dy_2 / tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp1);  // _Div_
                {
                    #pragma omp parallel for
                    for (auto __i0 = 0; __i0 < (ny - 2); __i0 += 1) {
                        for (auto __i1 = 0; __i1 < (nx - 2); __i1 += 1) {
                            const double __map_fusion_dx_2_dy_2_2_dx_2_dy_2_times_b_slice = (dx_2_dy_2_div_2_dx_2_dy_2 * b[b_idx((__i0 + 1), (__i1 + 1), nx)]);  // _Mult_
                            const double __map_fusion_pn_slice_plus_pn_slice_0 = (pn[pn_idx((__i0 + 2), (__i1 + 1), nx)] + pn[pn_idx(__i0, (__i1 + 1), nx)]);  // _Add_
                            const double __map_fusion_pn_slice_pn_slice_times_dx_2 = (__map_fusion_pn_slice_plus_pn_slice_0 * tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dx_pow_2);  // _Mult_
                            const double __map_fusion_pn_slice_plus_pn_slice = (pn[pn_idx((__i0 + 1), (__i1 + 2), nx)] + pn[pn_idx((__i0 + 1), __i1, nx)]);  // _Add_
                            const double __map_fusion_pn_slice_pn_slice_times_dy_2 = (__map_fusion_pn_slice_plus_pn_slice * tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson_dy_pow_2);  // _Mult_
                            const double __map_fusion_pn_slice_pn_slice_dy_2_plus_pn_slice_pn_slice_dx_2 = (__map_fusion_pn_slice_pn_slice_times_dy_2 + __map_fusion_pn_slice_pn_slice_times_dx_2);  // _Add_
                            const double __map_fusion_pn_slice_pn_slice_dy_2_pn_slice_pn_slice_dx_2_div_2_dx_2_dy_2 = (__map_fusion_pn_slice_pn_slice_dy_2_plus_pn_slice_pn_slice_dx_2 / tests_corpus_npbench_structured_grids_cavity_flow_pressure_poisson___tmp0);  // _Div_
                            p[p_idx((__i0 + 1), (__i1 + 1), nx)] = (__map_fusion_pn_slice_pn_slice_dy_2_pn_slice_pn_slice_dx_2_div_2_dx_2_dy_2 - __map_fusion_dx_2_dy_2_2_dx_2_dy_2_times_b_slice);  // _Sub_
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
                            p[p_idx(__i0, __i1, nx)] = 0;  // assign_100_8
                        }
                    }
                }

            }

        }
        {

            const double dx_pow_2_0 = (dace::math::ipow(dx, 2));  // _Pow_
            const double dt_div_dx_2_0 = (dt / dx_pow_2_0);  // _Div_
            const double dy_pow_2_0 = (dace::math::ipow(dy, 2));  // _Pow_
            const double dt_div_dy_2_0 = (dt / dy_pow_2_0);  // _Div_
            const double __tmp4 = (double(2) * rho);  // _Mult_
            const double __tmp5 = (__tmp4 * dy);  // _Mult_
            const double dt_div_2_rho_dy = (dt / __tmp5);  // _Div_
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (ny - 2); __i0 += 1) {
                    for (auto __i1 = 0; __i1 < (nx - 2); __i1 += 1) {
                        const double __map_fusion___tmp7 = (double(2) * vn[vn_idx((__i0 + 1), (__i1 + 1), nx)]);  // _Mult_
                        const double __map_fusion_vn_slice_minus_2_vn_slice_0 = (vn[vn_idx((__i0 + 2), (__i1 + 1), nx)] - __map_fusion___tmp7);  // _Sub_
                        const double __map_fusion_vn_slice_2_vn_slice_plus_vn_slice_0 = (__map_fusion_vn_slice_minus_2_vn_slice_0 + vn[vn_idx(__i0, (__i1 + 1), nx)]);  // _Add_
                        const double __map_fusion_dt_dy_2_times_vn_slice_2_vn_slice_vn_slice = (dt_div_dy_2_0 * __map_fusion_vn_slice_2_vn_slice_plus_vn_slice_0);  // _Mult_
                        const double __map_fusion___tmp6 = (double(2) * vn[vn_idx((__i0 + 1), (__i1 + 1), nx)]);  // _Mult_
                        const double __map_fusion_vn_slice_minus_2_vn_slice = (vn[vn_idx((__i0 + 1), (__i1 + 2), nx)] - __map_fusion___tmp6);  // _Sub_
                        const double __map_fusion_vn_slice_2_vn_slice_plus_vn_slice = (__map_fusion_vn_slice_minus_2_vn_slice + vn[vn_idx((__i0 + 1), __i1, nx)]);  // _Add_
                        const double __map_fusion_dt_dx_2_times_vn_slice_2_vn_slice_vn_slice = (dt_div_dx_2_0 * __map_fusion_vn_slice_2_vn_slice_plus_vn_slice);  // _Mult_
                        const double __map_fusion_dt_dx_2_vn_slice_2_vn_slice_vn_slice_plus_dt_dy_2_vn_slice_2_vn_slice_vn_slice = (__map_fusion_dt_dx_2_times_vn_slice_2_vn_slice_vn_slice + __map_fusion_dt_dy_2_times_vn_slice_2_vn_slice_vn_slice);  // _Add_
                        const double __map_fusion_nu_times_dt_dx_2_vn_slice_2_vn_slice_vn_slice_dt_dy_2_vn_slice_2_vn_slice_vn_slice = (nu * __map_fusion_dt_dx_2_vn_slice_2_vn_slice_vn_slice_plus_dt_dy_2_vn_slice_2_vn_slice_vn_slice);  // _Mult_
                        const double __map_fusion_p_slice_minus_p_slice_0 = (p[p_idx((__i0 + 2), (__i1 + 1), nx)] - p[p_idx(__i0, (__i1 + 1), nx)]);  // _Sub_
                        const double __map_fusion_dt_2_rho_dy_times_p_slice_p_slice = (dt_div_2_rho_dy * __map_fusion_p_slice_minus_p_slice_0);  // _Mult_
                        const double __map_fusion_vn_slice_minus_vn_slice_0 = (vn[vn_idx((__i0 + 1), (__i1 + 1), nx)] - vn[vn_idx(__i0, (__i1 + 1), nx)]);  // _Sub_
                        const double __map_fusion_vn_slice_times_dt_0 = (vn[vn_idx((__i0 + 1), (__i1 + 1), nx)] * dt);  // _Mult_
                        const double __map_fusion_vn_slice_dt_div_dy_0 = (__map_fusion_vn_slice_times_dt_0 / dy);  // _Div_
                        const double __map_fusion_vn_slice_dt_dy_times_vn_slice_vn_slice = (__map_fusion_vn_slice_dt_div_dy_0 * __map_fusion_vn_slice_minus_vn_slice_0);  // _Mult_
                        const double __map_fusion_vn_slice_minus_vn_slice = (vn[vn_idx((__i0 + 1), (__i1 + 1), nx)] - vn[vn_idx((__i0 + 1), __i1, nx)]);  // _Sub_
                        const double __map_fusion_un_slice_times_dt_0 = (un[un_idx((__i0 + 1), (__i1 + 1), nx)] * dt);  // _Mult_
                        const double __map_fusion_un_slice_dt_div_dx_0 = (__map_fusion_un_slice_times_dt_0 / dx);  // _Div_
                        const double __map_fusion_un_slice_dt_dx_times_vn_slice_vn_slice = (__map_fusion_un_slice_dt_div_dx_0 * __map_fusion_vn_slice_minus_vn_slice);  // _Mult_
                        const double __map_fusion_vn_slice_minus_un_slice_dt_dx_vn_slice_vn_slice = (vn[vn_idx((__i0 + 1), (__i1 + 1), nx)] - __map_fusion_un_slice_dt_dx_times_vn_slice_vn_slice);  // _Sub_
                        const double __map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_minus_vn_slice_dt_dy_vn_slice_vn_slice = (__map_fusion_vn_slice_minus_un_slice_dt_dx_vn_slice_vn_slice - __map_fusion_vn_slice_dt_dy_times_vn_slice_vn_slice);  // _Sub_
                        const double __map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_vn_slice_dt_dy_vn_slice_vn_slice_minus_dt_2_rho_dy_p_slice_p_slice = (__map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_minus_vn_slice_dt_dy_vn_slice_vn_slice - __map_fusion_dt_2_rho_dy_times_p_slice_p_slice);  // _Sub_
                        v[v_idx((__i0 + 1), (__i1 + 1), nx)] = (__map_fusion_vn_slice_un_slice_dt_dx_vn_slice_vn_slice_vn_slice_dt_dy_vn_slice_vn_slice_minus_dt_2_rho_dy_p_slice_p_slice + __map_fusion_nu_times_dt_dx_2_vn_slice_2_vn_slice_vn_slice_dt_dy_2_vn_slice_2_vn_slice_vn_slice);  // _Add_
                    }
                }
            }
            const double dx_pow_2 = (dace::math::ipow(dx, 2));  // _Pow_
            const double dt_div_dx_2 = (dt / dx_pow_2);  // _Div_
            const double dy_pow_2 = (dace::math::ipow(dy, 2));  // _Pow_
            const double dt_div_dy_2 = (dt / dy_pow_2);  // _Div_
            const double __tmp0 = (double(2) * rho);  // _Mult_
            const double __tmp1 = (__tmp0 * dx);  // _Mult_
            const double dt_div_2_rho_dx = (dt / __tmp1);  // _Div_
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < (ny - 2); __i0 += 1) {
                    for (auto __i1 = 0; __i1 < (nx - 2); __i1 += 1) {
                        const double __map_fusion___tmp3 = (double(2) * un[un_idx((__i0 + 1), (__i1 + 1), nx)]);  // _Mult_
                        const double __map_fusion_un_slice_minus_2_un_slice_0 = (un[un_idx((__i0 + 2), (__i1 + 1), nx)] - __map_fusion___tmp3);  // _Sub_
                        const double __map_fusion_un_slice_2_un_slice_plus_un_slice_0 = (__map_fusion_un_slice_minus_2_un_slice_0 + un[un_idx(__i0, (__i1 + 1), nx)]);  // _Add_
                        const double __map_fusion_dt_dy_2_times_un_slice_2_un_slice_un_slice = (dt_div_dy_2 * __map_fusion_un_slice_2_un_slice_plus_un_slice_0);  // _Mult_
                        const double __map_fusion___tmp2 = (double(2) * un[un_idx((__i0 + 1), (__i1 + 1), nx)]);  // _Mult_
                        const double __map_fusion_un_slice_minus_2_un_slice = (un[un_idx((__i0 + 1), (__i1 + 2), nx)] - __map_fusion___tmp2);  // _Sub_
                        const double __map_fusion_un_slice_2_un_slice_plus_un_slice = (__map_fusion_un_slice_minus_2_un_slice + un[un_idx((__i0 + 1), __i1, nx)]);  // _Add_
                        const double __map_fusion_dt_dx_2_times_un_slice_2_un_slice_un_slice = (dt_div_dx_2 * __map_fusion_un_slice_2_un_slice_plus_un_slice);  // _Mult_
                        const double __map_fusion_dt_dx_2_un_slice_2_un_slice_un_slice_plus_dt_dy_2_un_slice_2_un_slice_un_slice = (__map_fusion_dt_dx_2_times_un_slice_2_un_slice_un_slice + __map_fusion_dt_dy_2_times_un_slice_2_un_slice_un_slice);  // _Add_
                        const double __map_fusion_nu_times_dt_dx_2_un_slice_2_un_slice_un_slice_dt_dy_2_un_slice_2_un_slice_un_slice = (nu * __map_fusion_dt_dx_2_un_slice_2_un_slice_un_slice_plus_dt_dy_2_un_slice_2_un_slice_un_slice);  // _Mult_
                        const double __map_fusion_p_slice_minus_p_slice = (p[p_idx((__i0 + 1), (__i1 + 2), nx)] - p[p_idx((__i0 + 1), __i1, nx)]);  // _Sub_
                        const double __map_fusion_dt_2_rho_dx_times_p_slice_p_slice = (dt_div_2_rho_dx * __map_fusion_p_slice_minus_p_slice);  // _Mult_
                        const double __map_fusion_un_slice_minus_un_slice_0 = (un[un_idx((__i0 + 1), (__i1 + 1), nx)] - un[un_idx(__i0, (__i1 + 1), nx)]);  // _Sub_
                        const double __map_fusion_un_slice_minus_un_slice = (un[un_idx((__i0 + 1), (__i1 + 1), nx)] - un[un_idx((__i0 + 1), __i1, nx)]);  // _Sub_
                        const double __map_fusion_un_slice_times_dt = (un[un_idx((__i0 + 1), (__i1 + 1), nx)] * dt);  // _Mult_
                        const double __map_fusion_un_slice_dt_div_dx = (__map_fusion_un_slice_times_dt / dx);  // _Div_
                        const double __map_fusion_un_slice_dt_dx_times_un_slice_un_slice = (__map_fusion_un_slice_dt_div_dx * __map_fusion_un_slice_minus_un_slice);  // _Mult_
                        const double __map_fusion_un_slice_minus_un_slice_dt_dx_un_slice_un_slice = (un[un_idx((__i0 + 1), (__i1 + 1), nx)] - __map_fusion_un_slice_dt_dx_times_un_slice_un_slice);  // _Sub_
                        const double __map_fusion_vn_slice_times_dt = (vn[vn_idx((__i0 + 1), (__i1 + 1), nx)] * dt);  // _Mult_
                        const double __map_fusion_vn_slice_dt_div_dy = (__map_fusion_vn_slice_times_dt / dy);  // _Div_
                        const double __map_fusion_vn_slice_dt_dy_times_un_slice_un_slice = (__map_fusion_vn_slice_dt_div_dy * __map_fusion_un_slice_minus_un_slice_0);  // _Mult_
                        const double __map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_minus_vn_slice_dt_dy_un_slice_un_slice = (__map_fusion_un_slice_minus_un_slice_dt_dx_un_slice_un_slice - __map_fusion_vn_slice_dt_dy_times_un_slice_un_slice);  // _Sub_
                        const double __map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_vn_slice_dt_dy_un_slice_un_slice_minus_dt_2_rho_dx_p_slice_p_slice = (__map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_minus_vn_slice_dt_dy_un_slice_un_slice - __map_fusion_dt_2_rho_dx_times_p_slice_p_slice);  // _Sub_
                        u[u_idx((__i0 + 1), (__i1 + 1), nx)] = (__map_fusion_un_slice_un_slice_dt_dx_un_slice_un_slice_vn_slice_dt_dy_un_slice_un_slice_minus_dt_2_rho_dx_p_slice_p_slice + __map_fusion_nu_times_dt_dx_2_un_slice_2_un_slice_un_slice_dt_dy_2_un_slice_2_un_slice_un_slice);  // _Add_
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < 1; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        u[u_idx(__i0, __i1, nx)] = 0;  // assign_124_8
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < 1; __i1 += 1) {
                        u[u_idx(__i0, __i1, nx)] = 0;  // assign_125_8
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = (nx - 1); __i1 < nx; __i1 += 1) {
                        u[u_idx(__i0, __i1, nx)] = 0;  // assign_126_8
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = (ny - 1); __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        u[u_idx(__i0, __i1, nx)] = 1;  // assign_127_8
                    }
                }
            }
            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < 1; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        v[v_idx(__i0, __i1, nx)] = 0;  // assign_128_8
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = (ny - 1); __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < nx; __i1 += 1) {
                        v[v_idx(__i0, __i1, nx)] = 0;  // assign_129_8
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = 0; __i1 < 1; __i1 += 1) {
                        v[v_idx(__i0, __i1, nx)] = 0;  // assign_130_8
                    }
                }
            }

        }
        {

            {
                #pragma omp parallel for
                for (auto __i0 = 0; __i0 < ny; __i0 += 1) {
                    for (auto __i1 = (nx - 1); __i1 < nx; __i1 += 1) {
                        v[v_idx(__i0, __i1, nx)] = 0;  // assign_131_8
                    }
                }
            }

        }

    }
    dace::free(un);
    dace::free(vn);
    dace::free(b);
    dace::free(pn);
}

DACE_EXPORTED void __program_cavity_flow_experimental(cavity_flow_experimental_state_t *__state, double * __restrict__ p, double * __restrict__ u, double * __restrict__ v, double dt, double dx, double dy, int64_t nit, int64_t nt, double nu, int64_t nx, int64_t ny, double rho)
{
    __program_cavity_flow_experimental_internal(__state, p, u, v, dt, dx, dy, nit, nt, nu, nx, ny, rho);
}

DACE_EXPORTED cavity_flow_experimental_state_t *__dace_init_cavity_flow_experimental(int64_t nit, int64_t nx, int64_t ny)
{

    int __result = 0;
    cavity_flow_experimental_state_t *__state = new cavity_flow_experimental_state_t;

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_cavity_flow_experimental(cavity_flow_experimental_state_t *__state)
{

    int __err = 0;
    delete __state;
    return __err;
}

#include <cstdlib>
#include "../include/cavity_flow_experimental.h"

int main(int argc, char **argv) {
    cavity_flow_experimentalHandle_t handle;
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
    double * __restrict__ p = dace::calloc<double>((nx * ny));
    double * __restrict__ u = dace::calloc<double>((nx * ny));
    double * __restrict__ v = dace::calloc<double>((nx * ny));


    handle = __dace_init_cavity_flow_experimental(nit, nx, ny);
    __program_cavity_flow_experimental(handle, p, u, v, dt, dx, dy, nit, nt, nu, nx, ny, rho);
    err = __dace_exit_cavity_flow_experimental(handle);

    dace::free(p);
    dace::free(u);
    dace::free(v);


    return err;
}
