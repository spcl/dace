/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"
constexpr long long P = 16;
constexpr long long M = 32;
constexpr long long N = 32;
constexpr long long K = 32;

struct gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t {
    dace::fpga::Context *fpga_context;
};



DACE_EXPORTED void __dace_runstate_0_gemm_1(gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state, hlslib::ocl::Buffer<float, hlslib::ocl::Access::readWrite> &A_device, hlslib::ocl::Buffer<dace::vec<float, 8>, hlslib::ocl::Access::readWrite> &B_device, hlslib::ocl::Buffer<dace::vec<float, 8>, hlslib::ocl::Access::readWrite> &C_device);

void __program_gemm_fpga_systolic_vectorized_d16_w8_32x32x32_internal(gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state, float * __restrict__ A, dace::vec<float, 8> * __restrict__ B, dace::vec<float, 8> * __restrict__ C)
{
    hlslib::ocl::Buffer <float, hlslib::ocl::Access::readWrite> A_device;
    A_device = __state->fpga_context->Get().MakeBuffer<float, hlslib::ocl::Access::readWrite>(hlslib::ocl::StorageType::DDR, -1, (K * N));
    hlslib::ocl::Buffer <dace::vec<float, 8>, hlslib::ocl::Access::readWrite> B_device;
    B_device = __state->fpga_context->Get().MakeBuffer<dace::vec<float, 8>, hlslib::ocl::Access::readWrite>(hlslib::ocl::StorageType::DDR, -1, ((K * M) / 8));
    hlslib::ocl::Buffer <dace::vec<float, 8>, hlslib::ocl::Access::readWrite> C_device;
    C_device = __state->fpga_context->Get().MakeBuffer<dace::vec<float, 8>, hlslib::ocl::Access::readWrite>(hlslib::ocl::StorageType::DDR, -1, ((M * N) / 8));

    {

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                A_device.CopyFromHost(0, N * K, A);
            } // End omp section
            #pragma omp section
            {
                B_device.CopyFromHost(0, K * (M / 8), B);
            } // End omp section
            #pragma omp section
            {
                C_device.CopyFromHost(0, N * (M / 8), C);
            } // End omp section
        } // End omp sections

    }
    {
        __dace_runstate_0_gemm_1(__state, A_device, B_device, C_device);

    }
    {

        C_device.CopyToHost(0, N * (M / 8), C);

    }
}

DACE_EXPORTED void __program_gemm_fpga_systolic_vectorized_d16_w8_32x32x32(gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state, float * __restrict__ A, dace::vec<float, 8> * __restrict__ B, dace::vec<float, 8> * __restrict__ C)
{
    __program_gemm_fpga_systolic_vectorized_d16_w8_32x32x32_internal(__state, A, B, C);
}
DACE_EXPORTED int __dace_init_xilinx(gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state);

DACE_EXPORTED gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__dace_init_gemm_fpga_systolic_vectorized_d16_w8_32x32x32()
{
    int __result = 0;
    gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state = new gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t;


    __result |= __dace_init_xilinx(__state);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_gemm_fpga_systolic_vectorized_d16_w8_32x32x32(gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state)
{
    delete __state;
}

