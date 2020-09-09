// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

#ifndef __DACE_DYNMAP_CUH
#define __DACE_DYNMAP_CUH

#ifdef __CUDACC__
#include <cooperative_groups.h>

// HIP does not yet support features in cooperative groups used here.
/*#elif defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>
#endif*/

namespace dace {

    namespace cg = cooperative_groups;

    template<bool FINE_GRAINED, int BLOCK_SIZE, int WARP_SIZE = 32, typename index_type = int32_t>
    struct DynamicMap {

        // empty field so the compiler doesn't get confused when empty_type is used instead of fg_type
        struct empty_type {
            index_type src[1];  // outer map index
            index_type data[1]; // inner map index
        };

        struct tb_type {
            index_type owner;
            index_type src;
            index_type start;
            index_type size;
        };

        // fine grained needs space to accommodate all thread in the warp having the most possible jobs
        struct fg_type {
            index_type src[(BLOCK_SIZE / WARP_SIZE) * WARP_SIZE * WARP_SIZE];  // outer map index
            index_type data[(BLOCK_SIZE / WARP_SIZE) * WARP_SIZE * WARP_SIZE]; // inner map index
        };

        // depending on configuration of scheduler generate smaller union to save space
        template <typename tb_type, typename fg_type>
        union shared_type_temp {
            tb_type tb; // for tb-level np
            fg_type fg; // fine-grained schedule
        };

        typedef union std::conditional<FINE_GRAINED,
            shared_type_temp<tb_type, fg_type>,
            shared_type_temp<tb_type, empty_type>
        >::type shared_type;


        template<typename Functor>
        __device__ static void schedule(shared_type& s, index_type localStart, index_type localEnd, index_type localSrc, Functor&& work) {

            // defining other local variables
            index_type localSize = localEnd - localStart;

            cg::thread_block block = cg::this_thread_block();
            unsigned int threadRank = block.thread_rank();
            int blockSize = BLOCK_SIZE;

            // highest thread id is size - 1 thus no thread has this id
            if (threadRank == 0) {
                s.tb.owner = blockSize;
            }

            block.sync();

            // Thread block level scheduler
            while (true) {

                // every thread that has enough workload will compete for the memory, one thread will win the race condition
                if (localSize >= blockSize) {
                    s.tb.owner = threadIdx.x;
                }

                block.sync();

                // if no thread has the enough work, go to next granularity (e.g. warp or next lower group size)
                if (s.tb.owner == blockSize) {
                    break;
                }

                // The winner writes their name into the struct
                if (s.tb.owner == threadRank) {
                    s.tb.src = localSrc;
                    s.tb.start = localStart;
                    s.tb.size = localSize;
                    localSize = 0; // mark as processed
                }

                block.sync();

                // get winning values and reset owner for next round
                index_type src = s.tb.src;
                index_type start = s.tb.start;
                index_type size = s.tb.size;

                if (s.tb.owner == threadRank) {
                    s.tb.owner = blockSize;
                }

                block.sync();

                // do work
                for (unsigned int j = threadRank; j < size; j += blockSize) {
                    work(src, start + j);
                }

                block.sync();

            }

            cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
            unsigned int warpId = threadRank / WARP_SIZE;
            unsigned int warpThreadRank = warp.thread_rank();
            unsigned int warpSize = warp.size();

            // warp level scheduler, as long as there is any thread with >= warpSize tasks
            while (warp.any(localSize >= warpSize)) {

                // find a thread that has >= 32 jobs
                unsigned int mask = warp.ballot(localSize >= warpSize);
                int owner = __ffs(mask) - 1;

                // share value across warp
                index_type src = warp.shfl(localSrc, owner);
                index_type start = warp.shfl(localStart, owner);
                index_type size = warp.shfl(localSize, owner);

                // mark as processed
                if (owner == warpThreadRank) {
                    localSize = 0;
                }

                // do work
                for (unsigned int j = warpThreadRank; j < size; j += warpSize) {
                    work(src, start + j);
                }

            }

            // fine grained warp level load balancing
            if (FINE_GRAINED) {

                // prefix sum (after execution the prefix sum will include the value of the current thread)
                index_type prefix = localSize;
                for (int delta = 1; delta < warpSize; delta *= 2) {
                    index_type tmp = warp.shfl_up(prefix, delta);
                    prefix += (warpThreadRank >= delta) ? tmp : 0;
                }

                // total number and local start index (need to subtract localSize, because it is including the
                // local element)
                index_type total = warp.shfl(prefix, warpSize - 1);
                index_type warp_offset = warpId * WARP_SIZE*WARP_SIZE;
                index_type thread_offset = prefix - localSize;
                index_type offset = warp_offset + thread_offset;

                // write element range and owner to shared memory
                for (int i = 0; i < localSize; i++) {
                    s.fg.src[offset + i] = localSrc;
                    s.fg.data[offset + i] = localStart + i;
                }

                warp.sync(); // necessary because warps may diverge more since volta

                // do work
                for (unsigned int j = warp_offset + warpThreadRank; j < warp_offset + total; j += warpSize) {
                    work(s.fg.src[j], s.fg.data[j]);
                }

            } else {  // if (FINE_GRAINED)

                // do work
                if (localSize > 0) {
                    for (int j = localStart; j < localEnd; j++) {
                        work(localSrc, j);
                    }
                }

            }  // if (FINE_GRAINED)
        }  // __device__ static void schedule()
    };  // struct DynamicMap
}  // namespace dace

#endif // __CUDACC__

#endif // __DACE_DYNMAP_CUH
