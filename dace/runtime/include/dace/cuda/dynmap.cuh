// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

#ifndef __DACE_DYNMAP_CUH
#define __DACE_DYNMAP_CUH

#if defined(__CUDACC__) || defined(__HIPCC__)

#include <cstdint>
#include <type_traits>

namespace dace {

    // Warp primitives over a full warp of WARP_SIZE lanes
    namespace dynmap_warp {
#if defined(__HIPCC__)
        __device__ __forceinline__ unsigned long long ballot(bool predicate) { return __ballot(predicate); }
        template <typename T> __device__ __forceinline__ T shfl(T value, int lane) { return __shfl(value, lane); }
        template <typename T> __device__ __forceinline__ T shfl_up(T value, unsigned int delta) {
            return __shfl_up(value, delta);
        }
        __device__ __forceinline__ int first_lane(unsigned long long mask) { return __ffsll(mask) - 1; }
        // A value the whole wavefront agrees on, in a scalar register: atomics to an address derived from it combine
        template <typename T> __device__ __forceinline__ T uniform(T value) {
            if constexpr (sizeof(T) == 8) {
                const unsigned long long bits = static_cast<unsigned long long>(value);
                const unsigned int lo = __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(bits));
                const unsigned int hi = __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(bits >> 32));
                return static_cast<T>((static_cast<unsigned long long>(hi) << 32) | lo);
            } else {
                return static_cast<T>(__builtin_amdgcn_readfirstlane(static_cast<int>(value)));
            }
        }
#else
        __device__ __forceinline__ unsigned int ballot(bool predicate) { return __ballot_sync(0xffffffffu, predicate); }
        template <typename T> __device__ __forceinline__ T shfl(T value, int lane) {
            return __shfl_sync(0xffffffffu, value, lane);
        }
        template <typename T> __device__ __forceinline__ T shfl_up(T value, unsigned int delta) {
            return __shfl_up_sync(0xffffffffu, value, delta);
        }
        __device__ __forceinline__ int first_lane(unsigned int mask) { return __ffs(mask) - 1; }
        template <typename T> __device__ __forceinline__ T uniform(T value) { return value; }
#endif
    }  // namespace dynmap_warp

    /**
     * Load-balances the iterations of a dynamic thread-block map over a one-dimensional thread block. Every thread
     * owns a range ``[start, end)`` of iterations of the item ``src``; the block executes the iterations of all ranges
     * in rounds, in each of which a thread receives a strided batch of the iterations of one item. No step
     * synchronizes: the caller places the barriers (``block`` for ``__syncthreads``, ``warp`` for ``__syncwarp``):
     *
     *     cursor = DynamicMap::open(start, end, src); DynamicMap::reset(state); block;
     *     do {
     *         DynamicMap::block_elect(state, cursor); block;
     *         more = DynamicMap::block_claim(state, cursor);
     *         if (more) { block; DynamicMap::block_take(state, cursor, src, first, count, stride); block;
     *                     batch(src, first, count, stride); }
     *     } while (more);
     *     if (LISTED) block;
     *     do { more = DynamicMap::warp_round(state, cursor, src, first, count, stride); batch(...); } while (more);
     *     if (LISTED) warp;
     *     do { more = DynamicMap::thread_round(state, cursor, src, first, count); batch(src, first, count, 1); }
     *     while (more);
     *     if (LISTED) block;
     *
     * where ``batch`` runs the body for ``first + k * stride``, ``k < count``. Ranges of at least ``BLOCK_SIZE``
     * iterations are shared by the whole block, then ranges of at least a warp by their warp; the rest are either
     * spread over the warp one iteration per thread (``FINE_GRAINED``) or run by their thread. In the block and warp
     * rounds, the item is the same for the whole warp. The shared state is the range of the current block round and,
     * fine-grained on CUDA (``LISTED``), the list of the items each warp spreads; HIP's 64-wide warps find each item's
     * owner by shuffles instead, as their lists would not fit in shared memory. Every thread of the block takes every
     * step, out-of-range threads with an empty range. A body may not synchronize.
     */
    template<bool FINE_GRAINED, int BLOCK_SIZE, int WARP_SIZE = 32, typename INDEX_TYPE = int32_t>
    struct DynamicMap {
        static_assert(BLOCK_SIZE % WARP_SIZE == 0, "The block size of a dynamic map must be a multiple of the warp size");

#if defined(__HIPCC__)
        static constexpr bool LISTED = false;
#else
        static constexpr bool LISTED = FINE_GRAINED;
#endif

        // The range the block shares, and the thread that owns it
        struct block_range {
            INDEX_TYPE owner;
            INDEX_TYPE src;
            INDEX_TYPE start;
            INDEX_TYPE size;
        };

        // Fine-grained: per warp, the remaining items of its threads (fewer than a warp each)
        struct item_lists {
            INDEX_TYPE src[(BLOCK_SIZE / WARP_SIZE) * WARP_SIZE * WARP_SIZE];
            INDEX_TYPE data[(BLOCK_SIZE / WARP_SIZE) * WARP_SIZE * WARP_SIZE];
        };
        struct no_lists {};

        union shared_type {
            block_range tb;
            std::conditional_t<LISTED, item_lists, no_lists> fg;
        };

        // The scheduling state of one thread
        struct cursor_type {
            INDEX_TYPE local_start, local_size, local_src;  // what remains of the thread's own range
            INDEX_TYPE prefix, size, next;                  // fine-grained: the warp's items up to this thread's,
                                                            // all of them, and the first of the next round
            INDEX_TYPE start;                               // fine-grained on CUDA: the warp's part of the lists
            bool won;                                       // whether the thread owns the range of the block round
        };

        /// The cursor of a thread over its range ``[start, end)`` of the item ``src``.
        __device__ static cursor_type open(INDEX_TYPE start, INDEX_TYPE end, INDEX_TYPE src) {
            return cursor_type{start, end - start, src, 0, 0, 0, 0, false};
        }

        /// Starts the block rounds with no owner (the highest thread id is ``BLOCK_SIZE - 1``); a block barrier follows.
        __device__ static void reset(shared_type& s) {
            if (threadIdx.x == 0) {
                s.tb.owner = BLOCK_SIZE;
            }
        }

        /// Every thread with at least ``BLOCK_SIZE`` iterations left competes for the block round; a block barrier
        /// follows.
        __device__ static void block_elect(shared_type& s, const cursor_type& c) {
            if (c.local_size >= BLOCK_SIZE) {
                s.tb.owner = threadIdx.x;
            }
        }

        /// Whether a thread won the block round (the same for the whole block); the winner publishes its range, which
        /// ``block_take`` reads after a block barrier.
        __device__ static bool block_claim(shared_type& s, cursor_type& c) {
            const INDEX_TYPE owner = s.tb.owner;
            c.won = owner == static_cast<INDEX_TYPE>(threadIdx.x);
            if (c.won) {
                s.tb.src = c.local_src;
                s.tb.start = c.local_start;
                s.tb.size = c.local_size;
                c.local_size = 0;
            }
            return owner != BLOCK_SIZE;
        }

        /// The batch of this thread of the published range; the winner resets the owner, which every thread read
        /// before the barrier. A block barrier follows, before the next election.
        __device__ static void block_take(shared_type& s, cursor_type& c, INDEX_TYPE& src, INDEX_TYPE& first,
                                          INDEX_TYPE& count, INDEX_TYPE& stride) {
            src = dynmap_warp::uniform(s.tb.src);
            const INDEX_TYPE start = dynmap_warp::uniform(s.tb.start), size = dynmap_warp::uniform(s.tb.size);
            if (c.won) {
                s.tb.owner = BLOCK_SIZE;
            }
            batch(start, size, threadIdx.x, BLOCK_SIZE, first, count, stride);
        }

        /// The batch of this thread of the next range of its warp of at least a warp of iterations, or false (for the
        /// whole warp) and an empty batch if none is left; then, fine-grained, the warp lists its remaining items,
        /// which a warp barrier publishes on CUDA.
        __device__ static bool warp_round(shared_type& s, cursor_type& c, INDEX_TYPE& src, INDEX_TYPE& first,
                                          INDEX_TYPE& count, INDEX_TYPE& stride) {
            const INDEX_TYPE lane = threadIdx.x % WARP_SIZE;
            // find a thread that has at least a warp of jobs
            const auto mask = dynmap_warp::ballot(c.local_size >= WARP_SIZE);
            if (mask == 0) {
                if (FINE_GRAINED) {
                    fine_grained_ranges(s, c, lane);
                }
                count = 0;
                return false;
            }
            const int owner = dynmap_warp::uniform(dynmap_warp::first_lane(mask));
            src = dynmap_warp::uniform(dynmap_warp::shfl(c.local_src, owner));
            const INDEX_TYPE start = dynmap_warp::uniform(dynmap_warp::shfl(c.local_start, owner));
            const INDEX_TYPE size = dynmap_warp::uniform(dynmap_warp::shfl(c.local_size, owner));
            if (owner == lane) {
                c.local_size = 0;
            }
            batch(start, size, lane, WARP_SIZE, first, count, stride);
            return true;
        }

        /// The remaining iterations: fine-grained, one round of one iteration per thread of the warp, true while
        /// more follow (for the whole warp); otherwise the thread's own range, at once.
        __device__ static bool thread_round(shared_type& s, cursor_type& c, INDEX_TYPE& src, INDEX_TYPE& first,
                                            INDEX_TYPE& count) {
            if constexpr (LISTED) {
                // The warp's remaining items, listed in shared memory by ``fine_grained_ranges``
                const INDEX_TYPE item = c.next + threadIdx.x % WARP_SIZE;
                count = item < c.size ? 1 : 0;
                if (count) {
                    src = s.fg.src[c.start + item];
                    first = s.fg.data[c.start + item];
                }
                c.next += WARP_SIZE;
                return c.next < c.size;
            } else if (FINE_GRAINED) {
                // The warp's remaining items, in the order of the threads that own them: the owner of the item is the
                // first thread whose prefix exceeds it
                const INDEX_TYPE lane = threadIdx.x % WARP_SIZE;
                const INDEX_TYPE item = c.next + lane;
                int owner = 0;
                for (int width = WARP_SIZE / 2; width > 0; width /= 2) {
                    if (dynmap_warp::shfl(c.prefix, owner + width - 1) <= item) {
                        owner += width;
                    }
                }
                owner = owner < WARP_SIZE ? owner : WARP_SIZE - 1;
                const INDEX_TYPE owner_start = dynmap_warp::shfl(c.local_start, owner);
                const INDEX_TYPE owner_src = dynmap_warp::shfl(c.local_src, owner);
                const INDEX_TYPE owner_first = dynmap_warp::shfl(c.prefix - c.local_size, owner);
                src = owner_src;
                first = owner_start + (item - owner_first);
                count = item < c.size ? 1 : 0;
                c.next += WARP_SIZE;
                return c.next < c.size;
            }
            src = c.local_src;
            first = c.local_start;
            count = c.local_size;
            return false;
        }

    private:
        // The iterations ``offset, offset + width, ...`` of a range of ``size`` iterations from ``start``
        __device__ static void batch(INDEX_TYPE start, INDEX_TYPE size, INDEX_TYPE offset, INDEX_TYPE width,
                                     INDEX_TYPE& first, INDEX_TYPE& count, INDEX_TYPE& stride) {
            first = start + offset;
            count = offset < size ? (size - offset + width - 1) / width : 0;
            stride = width;
        }

        // The prefix sum of the warp's remaining items, up to and including this thread's, and on CUDA their lists
        __device__ static void fine_grained_ranges(shared_type& s, cursor_type& c, INDEX_TYPE lane) {
            INDEX_TYPE prefix = c.local_size;
            for (int delta = 1; delta < WARP_SIZE; delta *= 2) {
                const INDEX_TYPE tmp = dynmap_warp::shfl_up(prefix, delta);
                prefix += (lane >= delta) ? tmp : 0;
            }
            c.prefix = prefix;
            c.size = dynmap_warp::shfl(prefix, WARP_SIZE - 1);
            c.next = 0;
            if constexpr (LISTED) {
                const INDEX_TYPE warp_offset = (threadIdx.x / WARP_SIZE) * WARP_SIZE * WARP_SIZE;
                const INDEX_TYPE offset = warp_offset + prefix - c.local_size;
                for (INDEX_TYPE i = 0; i < c.local_size; i++) {
                    s.fg.src[offset + i] = c.local_src;
                    s.fg.data[offset + i] = c.local_start + i;
                }
                c.start = warp_offset;
            }
        }
    };  // struct DynamicMap
}  // namespace dace

#endif // __CUDACC__ || __HIPCC__

#endif // __DACE_DYNMAP_CUH
