// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_STREAM_CUH
#define __DACE_STREAM_CUH

#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <new>  // Used for the in-memory ctor call in the move assignment operator below
#include <vector>

#ifdef __HIPCC__
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#define gpuLaunchKernel hipLaunchKernel
#define gpuMalloc hipMalloc
#define gpuMemset hipMemset
#define gpuFree hipFree
#else
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#define gpuLaunchKernel cudaLaunchKernel
#define gpuMalloc cudaMalloc
#define gpuMemset cudaMemset
#define gpuFree cudaFree
#endif

#include "cudacommon.cuh"

namespace dace {
// Adapted from
// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
#ifndef __HIPCC__
__inline__ __device__ uint32_t atomicAggInc(uint32_t *ctr) {
  auto g = cooperative_groups::coalesced_threads();
  uint32_t warp_res;
  int rank = g.thread_rank();
  if (rank == 0) warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + rank;
}

__inline__ __device__ uint32_t atomicAggDec(uint32_t *ctr) {
  auto g = cooperative_groups::coalesced_threads();
  uint32_t warp_res;
  int rank = g.thread_rank();
  if (rank == 0) warp_res = atomicAdd(ctr, -g.size());
  return g.shfl(warp_res, 0) - g.size() + rank;
}
#else
// Version without cooperative groups to support HIP
__inline__ __device__ uint32_t atomicAggInc(uint32_t *ctr) {
  unsigned int lane = threadIdx.x % warpSize;
  unsigned long long int mask =
      __ballot(1);  // 64-bit because AMD warp size is 64
  unsigned int thread_offset = __popcll(mask & ((1ULL << lane) - 1));
  unsigned int leader = __ffsll(mask) - 1;
  unsigned int count = __popcll(mask);
  uint32_t warp_res;
  if (lane == leader) warp_res = atomicAdd(ctr, count);

  warp_res = __shfl(warp_res, leader);
  return warp_res + thread_offset;
}

__inline__ __device__ uint32_t atomicAggDec(uint32_t *ctr) {
  unsigned int lane = threadIdx.x % warpSize;
  unsigned long long int mask =
      __ballot(1);  // 64-bit because AMD warp size is 64
  unsigned int thread_offset = __popcll(mask & ((1ULL << lane) - 1));
  unsigned int leader = __ffsll(mask) - 1;
  unsigned int count = __popcll(mask);
  uint32_t warp_res;
  if (lane == leader) warp_res = atomicAdd(ctr, -count);

  warp_res = __shfl(warp_res, leader);
  return warp_res - count + thread_offset;
}
#endif  // __HIPCC__

//
// Queue classes (device):
//

/*
 * @brief A device-level MPMC Queue
 */
template <typename T, bool IS_POWEROFTWO = false>
class GPUStream {
 public:
  T *m_data;
  uint32_t *m_start, *m_end, *m_pending;
  uint32_t m_capacity_mask;

  __host__ GPUStream()
      : m_data(nullptr),
        m_start(nullptr),
        m_end(nullptr),
        m_pending(nullptr),
        m_capacity_mask(0) {}
  __host__ __device__ GPUStream(T *data, uint32_t capacity, uint32_t *start,
                                uint32_t *end, uint32_t *pending)
      : m_data(data),
        m_start(start),
        m_end(end),
        m_pending(pending),
        m_capacity_mask(IS_POWEROFTWO ? (capacity - 1) : capacity) {
    if (IS_POWEROFTWO) {
      assert((capacity - 1 & capacity) ==
             0);  // Must be a power of two for handling circular overflow
                  // correctly
    }
  }

  __device__ __forceinline__ void reset() const {
    *m_start = 0;
    *m_end = 0;
    *m_pending = 0;
  }

  __device__ __forceinline__ T pop() {
    uint32_t allocation = atomicAggInc(m_start);
    return m_data[get_addr(allocation)];
  }

  __device__ __forceinline__ T *leader_pop(uint32_t count) {
    uint32_t current = *m_start;
    T *result = m_data + get_addr(current);
    *m_start += count;
    return result;
  }

  __device__ __forceinline__ uint32_t get_addr(const uint32_t &i) const {
    if (IS_POWEROFTWO)
      return i & m_capacity_mask;
    else
      return i % m_capacity_mask;
  }

  __device__ __forceinline__ void push(const T &item) {
    uint32_t allocation = atomicAggInc(m_pending);
    m_data[get_addr(allocation)] = item;
  }

  /*
  __device__ __forceinline__ void push(T *items, int count)
  {
      // Perform a warp-wide scan to get thread offsets
      typedef cub::WarpScan<int> WarpScan;
      __shared__ typename WarpScan::TempStorage temp_storage[4];
      int offset;
      int warp_id = threadIdx.x / 32;
      WarpScan(temp_storage[warp_id]).ExclusiveSum(count, offset);

      // Atomic-add the total count once per warp
      uint32_t addr;
      if (threadIdx.x & 31 == 31) // Last thread
          addr = atomicAdd(m_pending, offset + count);
      // Broadcast starting address
      addr = cub::ShuffleIndex(addr, 31, 0xffffffff);

      // Copy data from each thread
      for(int i = 0; i < count; ++i)
          m_data[get_addr(addr + offset + i)] = items[i];
  }
  */

  __device__ __forceinline__ void prepend(const T &item) {
    uint32_t allocation = atomicAggDec(m_start) - 1;
    m_data[get_addr(allocation)] = item;
  }

  __device__ __forceinline__ T read(uint32_t i) const {
    return m_data[get_addr(*m_start + i)];
  }

  __device__ __forceinline__ uint32_t count() const {
    return *m_end - *m_start;
  }

  // Returns the 'count' of pending items and commits
  __device__ __forceinline__ uint32_t commit_pending() const {
    uint32_t count = *m_pending - *m_end;

    // Sync end with pending, this makes the pushed items visible to the
    // consumer
    *m_end = *m_pending;
    return count;
  }

  __device__ __forceinline__ uint32_t get_start() const { return *m_start; }

  __device__ __forceinline__ uint32_t
  get_start_delta(uint32_t prev_start) const {
    return prev_start - *m_start;
  }
};

////////////////////////////////////////////////////////////
// Host controllers for GPU streams

template <typename T, bool IS_POW2>
__global__ void ResetGPUStream_kernel(GPUStream<T, IS_POW2> stream) {
  stream.reset();
}

template <typename T, bool IS_POW2>
void ResetGPUStream(GPUStream<T, IS_POW2> &stream) {
  void *args_reset[1] = {&stream};
  gpuLaunchKernel((void *)&ResetGPUStream_kernel<T, IS_POW2>, dim3(1, 1, 1),
                  dim3(1, 1, 1), args_reset, 0, (gpuStream_t)0);
}

template <typename T, bool IS_POW2>
__global__ void PushToGPUStream_kernel(GPUStream<T, IS_POW2> stream, T item) {
  stream.push(item);
  stream.commit_pending();
}

template <typename T, bool IS_POW2>
void PushToGPUStream(GPUStream<T, IS_POW2> &stream, const T &item) {
  void *args_push[2] = {&stream, &item};
  gpuLaunchKernel((void *)&PushToGPUStream_kernel<T, IS_POW2>, dim3(1, 1, 1),
                  dim3(1, 1, 1), args_push, 0, (gpuStream_t)0);
}

////////////////////////////////////////////////////////////
// Host memory management for GPU streams

template <typename T, bool IS_POW2>
GPUStream<T, IS_POW2> AllocGPUArrayStreamView(T *ptr, uint32_t capacity) {
  uint32_t *gStart, *gEnd, *gPending;
  gpuMalloc(&gStart, sizeof(uint32_t));
  gpuMalloc(&gEnd, sizeof(uint32_t));
  gpuMalloc(&gPending, sizeof(uint32_t));
  gpuMemset(gStart, 0, sizeof(uint32_t));
  gpuMemset(gEnd, 0, sizeof(uint32_t));
  gpuMemset(gPending, 0, sizeof(uint32_t));
  return GPUStream<T, IS_POW2>(ptr, capacity, gStart, gEnd, gPending);
}

template <typename T, bool IS_POW2>
GPUStream<T, IS_POW2> AllocGPUStream(uint32_t capacity) {
  T *gData;
  gpuMalloc(&gData, capacity * sizeof(T));
  return AllocGPUArrayStreamView<T, IS_POW2>(gData, capacity);
}

template <typename T, bool IS_POW2>
void FreeGPUArrayStreamView(GPUStream<T, IS_POW2> &stream) {
  gpuFree(stream.m_start);
  gpuFree(stream.m_end);
  gpuFree(stream.m_pending);
}

template <typename T, bool IS_POW2>
void FreeGPUStream(GPUStream<T, IS_POW2> &stream) {
  FreeGPUArrayStreamView(stream);
  gpuFree(stream.m_data);
}

}  // namespace dace

#endif  // __DACE_STREAM_CUH
