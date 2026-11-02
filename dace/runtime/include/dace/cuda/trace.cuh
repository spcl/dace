// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Events that thread blocks record inside GPU kernels: the GPU_Events instrumentation of elements in device code.
// Every thread block that runs an instrumented element records its start and end on a global timer (one time axis
// for all multiprocessors) and the cycles of its multiprocessor in between.
#ifndef __DACE_CUDA_TRACE_CUH
#define __DACE_CUDA_TRACE_CUH

#include <cstdint>

namespace dace {
namespace perf {

struct GPUTraceEvent {
    uint64_t start;   // Global timer ticks (see gpu_trace_ticks)
    uint64_t end;
    uint64_t cycles;  // Multiprocessor clock cycles from start to end
    uint32_t element; // Index of the instrumented element in the program's table
    uint32_t block;   // Linear thread-block index
    uint32_t sm;      // Multiprocessor (compute unit) that ran the block
    uint32_t padding;
};

// The events of a program run, in GPU global memory. Events beyond the capacity are counted, not stored.
struct GPUTraceBuffer {
    GPUTraceEvent *events;
    unsigned long long capacity;
    unsigned long long count;
};

#if defined(__CUDACC__) || defined(__HIPCC__)

// The global timer: nanoseconds on CUDA, the constant-rate wall clock on HIP (see the host-side conversion)
__device__ __forceinline__ uint64_t gpu_trace_ticks() {
#if defined(__HIPCC__)
    return wall_clock64();
#else
    uint64_t ticks;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks));
    return ticks;
#endif
}

__device__ __forceinline__ uint32_t gpu_trace_sm() {
#if defined(__HIPCC__)
    return __smid();
#else
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    return sm;
#endif
}

__device__ __forceinline__ bool gpu_trace_is_first_thread() {
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}

__device__ __forceinline__ void gpu_trace_record(GPUTraceBuffer &buffer, uint32_t element, uint64_t start,
                                                 uint64_t start_cycles) {
    uint64_t end = gpu_trace_ticks();
    uint64_t cycles = clock64() - start_cycles;
    unsigned long long slot = atomicAdd(&buffer.count, 1ull);
    if (slot < buffer.capacity) {
        uint32_t block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        buffer.events[slot] = GPUTraceEvent{start, end, cycles, element, block, gpu_trace_sm(), 0};
    }
}

// Writes the global timer once (to relate it to the host's clock)
static __global__ void gpu_trace_read_ticks(uint64_t *ticks) { *ticks = gpu_trace_ticks(); }

#endif  // defined(__CUDACC__) || defined(__HIPCC__)

}  // namespace perf
}  // namespace dace

#endif  // __DACE_CUDA_TRACE_CUH
