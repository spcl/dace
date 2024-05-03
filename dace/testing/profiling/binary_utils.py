import ctypes

# Requires the .so be in LD_LIBRARY_PATH.
import os

import torch
from dace.codegen import targets, compiler
from dace.codegen.codeobject import CodeObject


def _compile_and_get_utils():
    cpp_code = """
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Forward declaration.
void gpu_wait(double length, cudaStream_t stream);

extern "C" void do_gpu_wait(double length) {
  gpu_wait(length, nullptr);
}
extern "C" void profiler_start() {
    cudaProfilerStart(); 
}
extern "C" void profiler_stop() {
    cudaProfilerStop(); 
}
    """
    program = CodeObject("cpp_code", cpp_code, "cpp", targets.cpu.CPUCodeGen,
                         "cpp_code")
    kernel_code = """
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void wait_kernel(long long int cycles) {
  const long long int start = clock64();
  long long int cur;
  do {
    cur = clock64();
  } while (cur - start < cycles);
}

}  // anonymous namespace

/**
 * Launch a kernel on stream that waits for length seconds.
 */
void gpu_wait(double length, cudaStream_t stream) {
  // Estimate GPU frequency to convert seconds to cycles.
  static long long int freq_hz = 0;  // Cache.
  if (freq_hz == 0) {
    int device;
    cudaGetDevice(&device);
    int freq_khz;
    cudaDeviceGetAttribute(&freq_khz, cudaDevAttrClockRate, device);
    freq_hz = (long long int) freq_khz * 1000;   // Convert from KHz.
  }
  double cycles = length * freq_hz;
  wait_kernel<<<1, 1, 0, stream>>>((long long int) cycles);
}
    """

    kernel = CodeObject("gpu_wait_kernel", kernel_code, "cu",
                        targets.cuda.CUDACodeGen, "gpu_wait_kernel")

    BUILD_PATH = os.path.join('.dacecache', "daceml_profiling_utils")
    compiler.generate_program_folder(None, [program, kernel], BUILD_PATH)

    compiler.configure_and_compile(BUILD_PATH)
    return ctypes.CDLL(
        os.path.join(BUILD_PATH, "build", "libdaceml_profiling_utils.so"))


handle = _compile_and_get_utils()
gpu_wait = lambda t: handle.do_gpu_wait(ctypes.c_double(t))
start_cuda_profiling = handle.profiler_start
stop_cuda_profiling = handle.profiler_stop
__all__ = ["gpu_wait", "start_cuda_profiling", "stop_cuda_profiling"]
