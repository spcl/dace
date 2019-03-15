// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Adapted from "Groute: An Asynchronous Multi-GPU Programming Framework"
// http://www.github.com/groute/groute


#ifndef __DACE_DYNMAP_CUH
#define __DACE_DYNMAP_CUH

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>
#include <mutex>

#include "../../../../external/cub/cub/util_ptx.cuh"

#define __FULL_MASK 0xffffffff

namespace dace {   
    /**
     * A map (usually dynamically sized) that can be rescheduled across a 
     * threadblock
     **/
    template<int BLOCK_SIZE, typename index_type = int32_t, bool WARP_INTRINSICS = true>
    struct DynamicMap
    {
        template<const int WARPS_PER_TB> struct warp_np {
            volatile index_type owner[WARPS_PER_TB];
            volatile index_type start[WARPS_PER_TB];
            volatile index_type size[WARPS_PER_TB];
            volatile index_type src[WARPS_PER_TB];
        };

        struct tb_np {
            index_type owner;
            index_type start;
            index_type size;
            index_type src;
        };

        struct empty_np {
        };


        template <typename ts_type, typename TTB, typename TWP>
        union np_shared {
            // for scans
            ts_type temp_storage;

            // for tb-level np
            TTB tb;

            // for warp-level np
            TWP warp;

            // fine-grained schedule (unused)
            //TFG fg;
        };

        /*
        * @brief A structure representing a scheduled chunk of work
        */
        struct np_local
        {
            index_type size; // work size
            index_type start; // work start
            index_type src; // work source thread / metadata
        };


        template <typename Functor>
        __device__ __forceinline__ static void schedule(index_type local_start, index_type local_end, index_type local_src, Functor&& work)
        {
            const int WP_SIZE = CUB_PTX_WARP_THREADS;
            const int TB_SIZE = BLOCK_SIZE;

            const int NP_WP_CROSSOVER = CUB_PTX_WARP_THREADS;
            const int NP_TB_CROSSOVER = blockDim.x;

            typedef union std::conditional<WARP_INTRINSICS,
                np_shared<empty_np, tb_np, empty_np>,
                np_shared<empty_np, tb_np, warp_np<BLOCK_SIZE / CUB_PTX_WARP_THREADS>>>::type np_shared_type;

            __shared__ np_shared_type np_shared;

            index_type local_size = local_end - local_start;

            if (threadIdx.x == 0)
            {
                np_shared.tb.owner = TB_SIZE + 1;
            }

            __syncthreads();

            //
            // First scheduler: processing high-degree work items using the entire block
            //
            while (true)
            {
                if (local_size >= NP_TB_CROSSOVER)
                {
                    // 'Elect' one owner for the entire thread block 
                    np_shared.tb.owner = threadIdx.x;
                }

                __syncthreads();

                if (np_shared.tb.owner == TB_SIZE + 1)
                {
                    // No owner was elected, i.e. no high-degree work items remain 

                    // No need to sync threads before moving on to WP scheduler  
                    // because it does not use shared memory
                    if (!WARP_INTRINSICS)
                        __syncthreads(); // Necessary do to the shared memory union used by both TB and WP schedulers
                    break;
                }

                if (np_shared.tb.owner == threadIdx.x)
                {
                    // This thread is the owner
                    np_shared.tb.start = local_start;
                    np_shared.tb.size = local_size;
                    np_shared.tb.src = local_src;

                    // Mark this work-item as processed for future schedulers 
                    local_start = 0;
                    local_size = 0;
                }

                __syncthreads();

                index_type start = np_shared.tb.start;
                index_type size = np_shared.tb.size;
                index_type src = np_shared.tb.src;

                if (np_shared.tb.owner == threadIdx.x)
                {
                    np_shared.tb.owner = TB_SIZE + 1;
                }

                // Use all threads in thread block to execute individual work  
                for (int ii = threadIdx.x; ii < size; ii += TB_SIZE)
                {
                    work(start + ii, src);
                }

                __syncthreads();
            }

            //
            // Second scheduler: tackle medium-degree work items using the warp 
            //
            const int warp_id = cub::WarpId();
            const int lane_id = cub::LaneId();

            while (__any_sync(__FULL_MASK, local_size >= NP_WP_CROSSOVER))
            {
                index_type start, size, src;
                if (WARP_INTRINSICS)
                {
                    // Compete for work scheduling  
                    unsigned int mask = __ballot_sync(__FULL_MASK, local_size >= NP_WP_CROSSOVER ? 1 : 0);
                    // Select a deterministic winner  
                    int leader = __ffs(mask) - 1;

                    // Broadcast data from the leader  
                    start = cub::ShuffleIndex<WP_SIZE>(local_start, leader, mask);
                    size = cub::ShuffleIndex<WP_SIZE>(local_size, leader, mask);
                    src = cub::ShuffleIndex<WP_SIZE>(local_src, leader, mask);

                    if (leader == lane_id)
                    {
                        // Mark this work-item as processed   
                        local_start = 0;
                        local_size = 0;
                    }
                }
                else
                {
                    // In order for this to compile, it should be refactored to another function
                    /*
                    if (local_size >= NP_WP_CROSSOVER)
                    {
                        // Again, race to select an owner for warp 
                        np_shared.warp.owner[warp_id] = lane_id;
                    }
                    if (np_shared.warp.owner[warp_id] == lane_id)
                    {
                        // This thread is owner 
                        np_shared.warp.start[warp_id] = local_start;
                        np_shared.warp.size[warp_id] = local_size;

                        // Mark this work-item as processed   
                        local_start = 0;
                        local_size = 0;
                    }
                    start = np_shared.warp.start[warp_id];
                    size = np_shared.warp.size[warp_id];
                    */
                }

                for (int ii = lane_id; ii < size; ii += WP_SIZE)
                {
                    work(start + ii, src);
                }
            }

            __syncthreads();

            //
            // Third scheduler: tackle all work-items with size < 32 serially   
            //
            // It is possible to disable this scheduler by setting NP_WP_CROSSOVER to 0 

            for (int ii = 0; ii < local_size; ii++)
            {
                work(local_start + ii, local_src);
            }
        }
    };

}  // namespace dace

#endif // __DACE_DYNMAP_CUH
