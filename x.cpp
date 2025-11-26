#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <omp.h>

// === Your first version ===
struct fuse_branches_test_pattern_from_cloudsc_one_state_t {};

inline void fuse_branches_test_pattern_from_cloudsc_one_1526_4_1527_8_0_0_9(
    fuse_branches_test_pattern_from_cloudsc_one_state_t *__state,
    const double&  __tmp_1528_21_r, const double&  __tmp_1528_34_r, const double&  __tmp_1529_34_r,
    double&  __tmp_1528_12_w, double&  __tmp_1531_16_w, double&  __tmp_1532_16_w)
{
    bool _if_cond_5;
    {
        double B_slice;
        {
            double __in1 = __tmp_1528_21_r;
            double __in2 = __tmp_1528_34_r;
            double __out;
            __out = (__in1 + __in2);
            B_slice = __out;
        }
        {
            double __inp = B_slice;
            double __out;
            __out = __inp;
            __tmp_1528_12_w = __out;
        }
    }
    _if_cond_5 = (__tmp_1528_12_w > __tmp_1529_34_r);
    if (_if_cond_5) {
        {
            double D_slice;
            double E_slice;
            {
                double __in1 = __tmp_1528_12_w;
                double __in2 = __tmp_1528_34_r;
                double __out;
                __out = (__in1 / __in2);
                D_slice = __out;
            }
            {
                double __inp = D_slice;
                double __out;
                __out = __inp;
                __tmp_1531_16_w = __out;
            }
            {
                double __in2 = __tmp_1531_16_w;
                double __out;
                __out = (1.0 - __in2);
                E_slice = __out;
            }
            {
                double __inp = E_slice;
                double __out;
                __out = __inp;
                __tmp_1532_16_w = __out;
            }
        }
    } else {
        {
            {
                double __out;
                __out = 0.0;
                __tmp_1531_16_w = __out;
            }
            {
                double __out;
                __out = 0.0;
                __tmp_1532_16_w = __out;
            }
        }
    }
}

void __program_fuse_branches_test_pattern_from_cloudsc_one_internal(
    fuse_branches_test_pattern_from_cloudsc_one_state_t*__state,
    double * __restrict__ A, double * __restrict__ B, double * __restrict__ D,
    double * __restrict__ E, double c)
{
    #pragma omp parallel for
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            fuse_branches_test_pattern_from_cloudsc_one_1526_4_1527_8_0_0_9(
                __state,
                A[((32 * i) + j) + 1024],
                A[(32 * i) + j],
                c,
                B[(32 * i) + j],
                D[(32 * i) + j],
                E[(32 * i) + j]);
        }
    }
}

// === Your transformed version ===
struct fuse_branches_test_pattern_from_cloudsc_one_transformed_state_t {};

inline void fuse_branches_test_pattern_from_cloudsc_one_1526_4_1527_8_0_0_9(
    fuse_branches_test_pattern_from_cloudsc_one_transformed_state_t *__state,
    const double&  __tmp_1528_21_r, const double&  __tmp_1528_34_r, const double&  __tmp_1529_34_r,
    double&  __tmp_1528_12_w, double&  __tmp_1531_16_w, double&  __tmp_1532_16_w)
{
    bool _if_cond_5;
    {
        double B_slice;
        {
            double __in1 = __tmp_1528_21_r;
            double __in2 = __tmp_1528_34_r;
            double __out;
            __out = (__in1 + __in2);
            B_slice = __out;
        }
        {
            double __inp = B_slice;
            double __out;
            __out = __inp;
            __tmp_1528_12_w = __out;
        }
    }
    _if_cond_5 = (__tmp_1528_12_w > __tmp_1529_34_r);
    {
        double D_slice;
        double E_slice;
        double float__if_cond_5;
        double if_body_tmp_0;
        double else_body_tmp_0;
        double if_body_tmp_0_0;
        double else_body_tmp_0_0;
        double tmp_ieassign;
        {
            double __in1 = __tmp_1528_12_w;
            double __in2 = __tmp_1528_34_r;
            double __out;
            __out = (__in1 / (__in2 + 2.220446049250313e-16));
            D_slice = __out;
        }
        {
            double __inp = D_slice;
            double __out;
            __out = __inp;
            if_body_tmp_0 = __out;
        }
        {
            double __out;
            __out = 0.0;
            else_body_tmp_0 = __out;
        }
        {
            double __out;
            __out = 0.0;
            else_body_tmp_0_0 = __out;
        }
        {
            double _in___tmp_1528_12_w_0 = __tmp_1528_12_w;
            double _in___tmp_1529_34_r_1 = __tmp_1529_34_r;
            double _out_float__if_cond_5;
            _out_float__if_cond_5 = (_in___tmp_1528_12_w_0 > _in___tmp_1529_34_r_1);
            tmp_ieassign = _out_float__if_cond_5;
        }
        {
            double _in = tmp_ieassign;
            double _out;
            _out = _in;
            float__if_cond_5 = _out;
        }
        {
            double _in_left = if_body_tmp_0;
            double _in_right = else_body_tmp_0;
            double _in_factor = float__if_cond_5;
            double _out;
            _out = ((_in_factor * _in_left) + ((1.0 - _in_factor) * _in_right));
            __tmp_1531_16_w = _out;
        }
        {
            double __in2 = __tmp_1531_16_w;
            double __out;
            __out = (1.0 - __in2);
            E_slice = __out;
        }
        {
            double __inp = E_slice;
            double __out;
            __out = __inp;
            if_body_tmp_0_0 = __out;
        }
        {
            double _in = tmp_ieassign;
            double _out;
            _out = _in;
            float__if_cond_5 = _out;
        }
        {
            double _in_left = if_body_tmp_0_0;
            double _in_right = else_body_tmp_0_0;
            double _in_factor = float__if_cond_5;
            double _out;
            _out = ((_in_factor * _in_left) + ((1.0 - _in_factor) * _in_right));
            __tmp_1532_16_w = _out;
        }
    }
}

void __program_fuse_branches_test_pattern_from_cloudsc_one_transformed_internal(
    fuse_branches_test_pattern_from_cloudsc_one_transformed_state_t*__state,
    double * __restrict__ A, double * __restrict__ B, double * __restrict__ D,
    double * __restrict__ E, double c)
{
    #pragma omp parallel for
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            fuse_branches_test_pattern_from_cloudsc_one_1526_4_1527_8_0_0_9(
                __state,
                A[((32 * i) + j) + 1024],
                A[(32 * i) + j],
                c,
                B[(32 * i) + j],
                D[(32 * i) + j],
                E[(32 * i) + j]);
        }
    }
}

// === Test harness ===
int main() {
    const int size = 32 * 32 + 1024;
    std::vector<double> A(size), B1(32 * 32), D1(32 * 32), E1(32 * 32);
    std::vector<double> B2(32 * 32), D2(32 * 32), E2(32 * 32);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 0.0);
    for (auto &v : A) v = dist(rng);

    fuse_branches_test_pattern_from_cloudsc_one_state_t s1;
    fuse_branches_test_pattern_from_cloudsc_one_transformed_state_t s2;

    double c = dist(rng);
    __program_fuse_branches_test_pattern_from_cloudsc_one_internal(&s1, A.data(), B1.data(), D1.data(), E1.data(), c);
    __program_fuse_branches_test_pattern_from_cloudsc_one_transformed_internal(&s2, A.data(), B2.data(), D2.data(), E2.data(), c);

    int mismatches = 0;
    for (int i = 0; i < 32 * 32; i++) {
        auto close = [](double x, double y) {
            if (std::isnan(x) && std::isnan(y)) return true;
            if (std::isinf(x) && std::isinf(y)) return true;
            return std::fabs(x - y) < 1e-12;
        };
        if (!close(B1[i], B2[i]) || !close(D1[i], D2[i]) || !close(E1[i], E2[i])) {
            if (mismatches < 10) {
                std::cerr << "Mismatch at " << i << ":\n";
                std::cerr << "  B: " << B1[i] << " vs " << B2[i] << "\n";
                std::cerr << "  D: " << D1[i] << " vs " << D2[i] << "\n";
                std::cerr << "  E: " << E1[i] << " vs " << E2[i] << "\n\n";
            }
            mismatches++;
        }
    }
    std::cout << "Total mismatches: " << mismatches << std::endl;
    return 0;
}
