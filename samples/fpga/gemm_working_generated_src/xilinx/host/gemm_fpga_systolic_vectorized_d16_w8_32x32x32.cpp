#include "dace/xilinx/host.h"
#include "dace/dace.h"


constexpr long long P = 16;
constexpr long long M = 32;
constexpr long long N = 32;
constexpr long long K = 32;

struct gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t {
    dace::fpga::Context *fpga_context;
};


DACE_EXPORTED int __dace_init_xilinx(gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state) {
    dace::set_environment_variable("XCL_EMULATION_MODE", "hw_emu");
    dace::set_environment_variable("XILINX_SDX", DACE_VITIS_DIR);
    dace::set_environment_variable("EMCONFIG_PATH", DACE_BINARY_DIR);
    
    
    __state->fpga_context = new dace::fpga::Context();
    __state->fpga_context->Get().MakeProgram(DACE_BINARY_DIR "/gemm_fpga_systolic_vectorized_d16_w8_32x32x32_hw_emu.xclbin");
    return 0;
}

DACE_EXPORTED void __dace_exit_xilinx(gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state) {
    delete __state->fpga_context;
}

///////////////////////////////////////////////////////////////////////////////
// Kernel: gemm_0_0
///////////////////////////////////////////////////////////////////////////////

// Signature of kernel function (with raw pointers) for argument matching
DACE_EXPORTED void gemm_0_0(float * __restrict__ __A_device_in_0, dace::vec<float, 8> * __restrict__ __B_device_in_0, dace::vec<float, 8> * __restrict__ __C_device_in_0, dace::vec<float, 8> * __restrict__ __C_device_out_0);

DACE_EXPORTED void __dace_runstate_0_gemm_1(gemm_fpga_systolic_vectorized_d16_w8_32x32x32_t *__state, hlslib::ocl::Buffer<float, hlslib::ocl::Access::readWrite> &A_device, hlslib::ocl::Buffer<dace::vec<float, 8>, hlslib::ocl::Access::readWrite> &B_device, hlslib::ocl::Buffer<dace::vec<float, 8>, hlslib::ocl::Access::readWrite> &C_device) {
    hlslib::ocl::Program program = __state->fpga_context->Get().CurrentlyLoadedProgram();
    std::vector<cl::Event> all_events;
    all_events.push_back(program.MakeKernel("dp_kernel_top").ExecuteTaskFork());
    auto gemm_0_0_kernel = program.MakeKernel(gemm_0_0, "gemm_0_0", A_device, B_device, C_device, C_device);
    cl::Event gemm_0_0_event = gemm_0_0_kernel.ExecuteTaskFork();
    all_events.push_back(gemm_0_0_event);
    cl::Event::waitForEvents(all_events);
}


