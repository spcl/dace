#ifndef __DACE_STREAM_CUH
#define __DACE_STREAM_CUH

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <new> // Used for the in-memory ctor call in the move assignment operator below  

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../../../../external/cub/cub/util_ptx.cuh"
#include "../../../../external/cub/cub/warp/warp_reduce.cuh"
#include "../../../../external/cub/cub/warp/warp_scan.cuh"

#include "cudacommon.cuh"

namespace dace {
    // Adapted from https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
    __inline__ __device__ uint32_t atomicAggInc(uint32_t *ctr) {
        auto g = cooperative_groups::coalesced_threads();
        uint32_t warp_res;
        int rank = g.thread_rank();
        if (rank == 0)
            warp_res = atomicAdd(ctr, g.size());
        return g.shfl(warp_res, 0) + rank;
    }

    __inline__ __device__ uint32_t atomicAggDec(uint32_t *ctr) {
        auto g = cooperative_groups::coalesced_threads();
        uint32_t warp_res;
        int rank = g.thread_rank();
        if (rank == 0)
            warp_res = atomicAdd(ctr, -g.size());
        return g.shfl(warp_res, 0) + rank;
    }

    /*
    __inline__ __device__ uint32_t warpReduceSum(uint32_t val) {
        for (int offset = CUB_PTX_WARP_THREADS / 2; offset > 0; offset /= 2)
            val += __shfl_down(val, offset);
        return val;
    }
    */

    //
    // Queue classes (device):
    //

    /*
    * @brief A device-level MPMC Queue
    */
    template<typename T, bool IS_POWEROFTWO = false>
    class GPUStream
    {
    public:
        T* m_data;
        uint32_t *m_start, *m_end, *m_pending;
        uint32_t m_capacity_mask;

        __host__ GPUStream() : m_data(nullptr), m_start(nullptr), m_end(nullptr),
            m_pending(nullptr), m_capacity_mask(0) {}
        __host__ __device__ GPUStream(T* data, uint32_t capacity,
                                      uint32_t *start, uint32_t *end, 
                                      uint32_t *pending) :
            m_data(data), m_start(start), m_end(end), m_pending(pending),
            m_capacity_mask(IS_POWEROFTWO ? (capacity - 1) : capacity)
        {
            if (IS_POWEROFTWO) {
                assert((capacity - 1 & capacity) == 0); // Must be a power of two for handling circular overflow correctly  
            }
        }

        __device__ __forceinline__ void reset() const
        {
            *m_start = 0; 
            *m_end = 0;
            *m_pending = 0;
        }

        __device__ __forceinline__ T pop()
        {
            uint32_t allocation = atomicAggInc(m_start);
            return m_data[get_addr(allocation)];
        }

        __device__ __forceinline__ T *leader_pop(uint32_t count) {
            uint32_t current = *m_start;
            T *result = m_data + get_addr(current);
            *m_start += count;
            return result;
        }


        __device__ __forceinline__ uint32_t get_addr(const uint32_t& i) const {
            if (IS_POWEROFTWO)
                return i & m_capacity_mask;
            else
                return i % m_capacity_mask;
        }

        __device__ __forceinline__ void push(const T& item)
        {
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

        __device__ __forceinline__ void prepend(const T& item)
        {
            uint32_t allocation = atomicAggDec(m_start) - 1;
            m_data[get_addr(allocation)] = item;
        }

        __device__ __forceinline__ T read(uint32_t i) const
        {
            return m_data[get_addr(*m_start + i)];
        }
                        
        __device__ __forceinline__ uint32_t count() const
        {
            return *m_end - *m_start;
        }

        // Returns the 'count' of pending items and commits
        __device__ __forceinline__ uint32_t commit_pending() const
        {
            uint32_t count = *m_pending - *m_end;
                
            // Sync end with pending, this makes the pushed items visible to the consumer
            *m_end = *m_pending;
            return count;
        }

        __device__ __forceinline__ uint32_t get_start() const
        {
            return *m_start;
        }

        __device__ __forceinline__ uint32_t get_start_delta(uint32_t prev_start) const
        {
            return prev_start - *m_start;
        }
    };

    ////////////////////////////////////////////////////////////
    // Host controllers for GPU streams

    template<typename T, bool IS_POW2>
    __global__ void ResetGPUStream_kernel(GPUStream<T, IS_POW2> stream)
    {
        stream.reset();
    }

    template<typename T, bool IS_POW2>
    void ResetGPUStream(GPUStream<T, IS_POW2>& stream)
    {
        void *args_reset[1] = { &stream };
        DACE_CUDA_CHECK(cudaLaunchKernel((void *)&ResetGPUStream_kernel<T, IS_POW2>,
                                         dim3(1, 1, 1), dim3(1, 1, 1), 
                                         args_reset, 0, (cudaStream_t)0));
    }

    template<typename T, bool IS_POW2>
    __global__ void PushToGPUStream_kernel(GPUStream<T, IS_POW2> stream, T item)
    {
        stream.push(item);
        stream.commit_pending();
    }

    template<typename T, bool IS_POW2>
    void PushToGPUStream(GPUStream<T, IS_POW2>& stream, const T& item)
    {
        void *args_push[2] = { &stream, &item };
        DACE_CUDA_CHECK(cudaLaunchKernel((void *)&PushToGPUStream_kernel<T, IS_POW2>,
                                         dim3(1, 1, 1), dim3(1, 1, 1), 
                                         args_push, 0, (cudaStream_t)0));
    }

    ////////////////////////////////////////////////////////////
    // Host memory management for GPU streams


    template<typename T, bool IS_POW2>
    GPUStream<T, IS_POW2> AllocGPUArrayStreamView(T *ptr, uint32_t capacity)
    {
        uint32_t *gStart, *gEnd, *gPending;
        DACE_CUDA_CHECK(cudaMalloc(&gStart, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMalloc(&gEnd, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMalloc(&gPending, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMemsetAsync(gStart, 0, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMemsetAsync(gEnd, 0, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMemsetAsync(gPending, 0, sizeof(uint32_t)));
        return GPUStream<T, IS_POW2>(ptr, capacity, gStart, gEnd, gPending);
    }

    template<typename T, bool IS_POW2>
    GPUStream<T, IS_POW2> AllocGPUStream(uint32_t capacity)
    {
        T *gData;
        DACE_CUDA_CHECK(cudaMalloc(&gData, capacity * sizeof(T)));
        return AllocGPUArrayStreamView<T, IS_POW2>(gData, capacity);
    }

    template<typename T, bool IS_POW2>
    void FreeGPUArrayStreamView(GPUStream<T, IS_POW2>& stream)
    {
        DACE_CUDA_CHECK(cudaFree(stream.m_start));
        DACE_CUDA_CHECK(cudaFree(stream.m_end));
        DACE_CUDA_CHECK(cudaFree(stream.m_pending));
    }

    template<typename T, bool IS_POW2>
    void FreeGPUStream(GPUStream<T, IS_POW2>& stream)
    {
        FreeGPUArrayStreamView(stream);
        DACE_CUDA_CHECK(cudaFree(stream.m_data));
    }

}  // namespace dace
#endif // __DACE_STREAM_CUH