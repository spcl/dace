// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Memory engine (TMA) helpers for the ``MemoryEngineBulk`` expansion of
// ``CopyLibraryNode``. Included through the ``MemoryEngine`` environment, by
// both the host (frame) and the device code.

#ifndef __DACE_MEMORYENGINE_CUH
#define __DACE_MEMORYENGINE_CUH

#if defined(__HIPCC__) || defined(WITH_HIP)
#error "Memory engine copies are not implemented for AMD GPUs"
#endif

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#include <cuda/ptx>
#endif

#include "cudacommon.cuh"

namespace dace {
namespace cuda {

/// Resolve ``cuTensorMapEncodeTiled`` from the driver through the runtime, so
/// no libcuda link is needed. The lookup happens once; later calls return the
/// cached pointer. ``__state`` receives GPU errors.
template <typename State>
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled(State* __state) {
  static const PFN_cuTensorMapEncodeTiled_v12000 encode =
      [__state]() -> PFN_cuTensorMapEncodeTiled_v12000 {
    static constexpr int MIN_DRIVER_VERSION = 12000;
    static constexpr auto FUNCTION_NAME = "cuTensorMapEncodeTiled";
    int version;
    DACE_GPU_CHECK(cudaDriverGetVersion(&version));
    if (version < MIN_DRIVER_VERSION) {
      fprintf(stderr,
              "Driver version (%d) too old for TMA, requires at least %d\n",
              version, MIN_DRIVER_VERSION);
      return nullptr;
    }

    cudaDriverEntryPointQueryResult driver_status;
    void* ptr = nullptr;
    DACE_GPU_CHECK(cudaGetDriverEntryPointByVersion(
        FUNCTION_NAME, &ptr, MIN_DRIVER_VERSION, cudaEnableDefault,
        &driver_status));
    switch (driver_status) {
      case cudaDriverEntryPointSuccess:
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
      case cudaDriverEntryPointSymbolNotFound:
        fprintf(stderr, "%s not found in driver\n", FUNCTION_NAME);
        return nullptr;
      case cudaDriverEntryPointVersionNotSufficent:
        fprintf(stderr,
                "Driver version insufficient for %s (got %d, need %d)\n",
                FUNCTION_NAME, version, MIN_DRIVER_VERSION);
        return nullptr;
      default:
        fprintf(stderr, "Unknown error: %d\n", static_cast<int>(driver_status));
        return nullptr;
    }
  }();
  if (encode == nullptr) {
    fprintf(stderr,
            "cuTensorMapEncodeTiled is unavailable, cannot use memory engine "
            "copies\n");
    std::abort();
  }
  return encode;
}

/// Encode a tiled tensor map (TMA descriptor). All arrays are in TMA order:
/// dimension 0 is the contiguous one. ``strides_bytes`` holds ``rank - 1``
/// strides (in bytes) of dimensions 1..rank-1.
template <typename State>
CUtensorMap make_tiled_tensor_map(State* __state, CUtensorMapDataType dtype,
                                  void* global_address, cuuint32_t rank,
                                  const cuuint64_t* dims,
                                  const cuuint64_t* strides_bytes,
                                  const cuuint32_t* box) {
  CUtensorMap tensor_map;
  cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
  CUresult result = get_cuTensorMapEncodeTiled(__state)(
      &tensor_map, dtype, rank, global_address, dims, strides_bytes, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    fprintf(
        stderr,
        "cuTensorMapEncodeTiled failed (error %d) for address %p, rank %u:\n",
        static_cast<int>(result), global_address, rank);
    for (cuuint32_t i = 0; i < rank; ++i) {
      fprintf(
          stderr, "  dim %u: size %llu, stride %llu B, box %u\n", i,
          static_cast<unsigned long long>(dims[i]),
          static_cast<unsigned long long>(i == 0 ? 0 : strides_bytes[i - 1]),
          box[i]);
    }
    fprintf(stderr,
            "Requirements: 16 B aligned address, strides multiple of 16 B, box "
            "<= 256, box[0] * element size multiple of 16 B\n");
    std::abort();
  }
  return tensor_map;
}

#ifdef __CUDACC__

/// Linear thread index within the block.
__device__ __forceinline__ unsigned int linear_thread_index() {
  return threadIdx.x + (threadIdx.y + threadIdx.z * blockDim.y) * blockDim.x;
}

/// The single thread that issues memory engine operations.
__device__ __forceinline__ bool is_leader_thread() {
  return linear_thread_index() == 0;
}

/// Initialize ``bar`` so that every thread of the block arrives once per phase.
/// Called by all threads.
__device__ __forceinline__ void tma_barrier_init(uint64_t* bar) {
  if (is_leader_thread()) {
    ::cuda::ptx::mbarrier_init(bar, blockDim.x * blockDim.y * blockDim.z);
    // Make the initialized barrier visible to the memory engine (async proxy)
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
  }
  __syncthreads();
}

/// Block-collective global -> shared tensor copy. Called by all threads;
/// returns once ``bytes`` arrived. The wait is token based, so the barrier can
/// be reused (e.g. in a loop) without tracking phase parity. The SDFG models
/// the barrier as a read-only input (its state changes through the barrier
/// operations only), so a nested SDFG may hand it over as ``const``.
template <int Rank>
__device__ __forceinline__ void tma_load(void* smem, const CUtensorMap* desc,
                                         const int32_t (&coords)[Rank],
                                         const uint64_t* barrier,
                                         uint32_t bytes) {
  uint64_t* bar = const_cast<uint64_t*>(barrier);
  uint64_t token;
  if (is_leader_thread()) {
    ::cuda::ptx::cp_async_bulk_tensor(::cuda::ptx::space_cluster,
                                      ::cuda::ptx::space_global, smem, desc,
                                      coords, bar);
    token = ::cuda::ptx::mbarrier_arrive_expect_tx(
        ::cuda::ptx::sem_release, ::cuda::ptx::scope_cta,
        ::cuda::ptx::space_shared, bar, bytes);
  } else {
    token = ::cuda::ptx::mbarrier_arrive(bar);
  }
  while (!::cuda::ptx::mbarrier_try_wait(bar, token)) {
  }
}

/// Block-collective shared -> global tensor copy. Called by all threads;
/// returns once the copy completed.
template <int Rank>
__device__ __forceinline__ void tma_store(const CUtensorMap* desc,
                                          const int32_t (&coords)[Rank],
                                          const void* smem) {
  // Make the generic-proxy shared memory writes of all threads visible to the
  // memory engine
  ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
  __syncthreads();
  if (is_leader_thread()) {
    ::cuda::ptx::cp_async_bulk_tensor(::cuda::ptx::space_global,
                                      ::cuda::ptx::space_shared, desc, coords,
                                      smem);
    ::cuda::ptx::cp_async_bulk_commit_group();
    ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>());
  }
  // The shared buffer must not be overwritten before the memory engine reads it
  __syncthreads();
}

#endif  // __CUDACC__

}  // namespace cuda
}  // namespace dace

#endif  // __DACE_MEMORYENGINE_CUH
