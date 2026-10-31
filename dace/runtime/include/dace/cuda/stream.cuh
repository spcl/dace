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

/*
 * @brief A device-level MPMC work queue for streams that are popped on the
 * device while tasks push into them (lowered consume scopes).
 *
 * Unlike GPUStream, every push is visible on its own. The queue is a bounded
 * ring whose cells carry sequence numbers: the cell of position ``pos`` is
 * free for that push when its sequence is ``pos``, and full when it is
 * ``pos + 1``. A push reserves a position, fills the cell, and then counts the
 * element as queued. A pop is admitted by that count first (a subtraction,
 * the excess given back), so every position it then takes from the head holds
 * an element or is about to; there is no compare-and-swap to retry. Pops never
 * block on an empty queue.
 *
 * ``pending`` counts elements pushed and not yet retired: a push counts before
 * it publishes, and a pop with ``noupdate`` leaves its elements counted until
 * ``task_done``. It reaches zero only once nothing is queued and no task is
 * running, and then stays zero (only running tasks push).
 *
 * With ``BLOCK``, the queue lives in shared memory and serves the threads of
 * one block (see ``init_block``); its fences are block-scoped.
 *
 * The capacity is a power of two. A push into a full queue waits for a pop,
 * so the capacity must hold every element that can be queued at once.
 * Waiting for a cell relies on independent thread scheduling (sm_70 and
 * newer) when the thread it waits for is in the same warp. Compile with
 * ``DACE_GPUQUEUE_DEBUG`` to trap with the queue state on a wait that does not
 * end (e.g., a full queue).
 */
#ifdef DACE_GPUQUEUE_DEBUG
#define DACE_GPUQUEUE_WAIT(cond, what, pos, seq)                                  \
  do {                                                                           \
    unsigned long long __spins = 0;                                              \
    while (cond) {                                                               \
      if (++__spins == (1ull << 24)) {                                           \
        printf("GPUQueue %s stuck: pos %u seq %u head %u tail %u queued %d "      \
               "pending %u\n", what, pos, seq, load(head()), load(tail()),        \
               (int)load(queued()), load(pending_counter()));              \
        __trap();                                                                \
      }                                                                          \
    }                                                                            \
  } while (0)
#else
// The thread waited for may be in the same warp: sleeping lets the warp
// scheduler run it (a tight spin can starve it)
#define DACE_GPUQUEUE_WAIT(cond, what, pos, seq) \
  do {                                          \
    while (cond) {                              \
      GPUQueue_yield();                         \
    }                                           \
  } while (0)
#endif

__device__ __forceinline__ void GPUQueue_yield() {
#if defined(__HIPCC__)
  __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __nanosleep(32);
#endif
}

template <typename T, bool BLOCK = false>
class GPUQueue {
 public:
  T *m_data;
  uint32_t *m_seq;
  // Head, tail, queued count and pending count
  uint32_t *m_counters;
  uint32_t m_mask;

  // Global counters 128 bytes apart, so that their atomics do not share a line
  static constexpr uint32_t COUNTER_STRIDE = BLOCK ? 1 : 32;
  static constexpr uint32_t NUM_COUNTERS = 4;

  __host__ __device__ GPUQueue()
      : m_data(nullptr), m_seq(nullptr), m_counters(nullptr), m_mask(0) {}

  // Sets up a queue on shared memory; every thread of the block calls it
  __device__ __forceinline__ void init_block(T *data, uint32_t *seq,
                                             uint32_t *counters,
                                             uint32_t capacity) {
    m_data = data;
    m_seq = seq;
    m_counters = counters;
    m_mask = capacity - 1;
    const uint32_t nthreads = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t rank =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    for (uint32_t i = rank; i < capacity; i += nthreads) seq[i] = i;
    for (uint32_t i = rank; i < NUM_COUNTERS * COUNTER_STRIDE; i += nthreads)
      counters[i] = 0;
    __syncthreads();
  }

  static __device__ __forceinline__ void fence() {
    if (BLOCK)
      __threadfence_block();
    else
      __threadfence();
  }

  __device__ __forceinline__ uint32_t *head() const { return m_counters; }
  __device__ __forceinline__ uint32_t *tail() const {
    return m_counters + COUNTER_STRIDE;
  }
  __device__ __forceinline__ uint32_t *queued() const {
    return m_counters + 2 * COUNTER_STRIDE;
  }
  __device__ __forceinline__ uint32_t *pending_counter() const {
    return m_counters + 3 * COUNTER_STRIDE;
  }

  // Memory ordering: counters are polled relaxed; a cell's sequence number is
  // read with acquire and written with release, which orders the element
  // itself. Scope is the block for shared-memory queues, the device otherwise.
#if !defined(__HIPCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  static __device__ __forceinline__ uint32_t load(const uint32_t *ptr) {
    uint32_t value;
    if (BLOCK)
      asm volatile("ld.relaxed.cta.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    else
      asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
  }
  static __device__ __forceinline__ uint32_t load_acquire(const uint32_t *ptr) {
    uint32_t value;
    if (BLOCK)
      asm volatile("ld.acquire.cta.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    else
      asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
  }
  static __device__ __forceinline__ void store_release(uint32_t *ptr, uint32_t value) {
    if (BLOCK)
      asm volatile("st.release.cta.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
    else
      asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
  }
  static __device__ __forceinline__ void store(uint32_t *ptr, uint32_t value) {
    if (BLOCK)
      asm volatile("st.relaxed.cta.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
    else
      asm volatile("st.relaxed.gpu.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
  }
  // With relaxed accesses around it, acquires what was read before and
  // releases what is written after: one fence for a whole chunk of cells
  static __device__ __forceinline__ void fence_acq_rel() {
    if (BLOCK)
      asm volatile("fence.acq_rel.cta;" : : : "memory");
    else
      asm volatile("fence.acq_rel.gpu;" : : : "memory");
  }
#else
  static __device__ __forceinline__ uint32_t load(const uint32_t *ptr) {
    return *(const volatile uint32_t *)ptr;
  }
  static __device__ __forceinline__ uint32_t load_acquire(const uint32_t *ptr) {
    const uint32_t value = load(ptr);
    fence();
    return value;
  }
  static __device__ __forceinline__ void store_release(uint32_t *ptr, uint32_t value) {
    fence();
    *(volatile uint32_t *)ptr = value;
  }
  static __device__ __forceinline__ void store(uint32_t *ptr, uint32_t value) {
    *(volatile uint32_t *)ptr = value;
  }
  static __device__ __forceinline__ void fence_acq_rel() { fence(); }
#endif

  // Counter atomics, relaxed: a cell's sequence number orders the element.
  // Every thread issues its own; ptxas aggregates atomics on one address
  // across the warp (one atomic for a whole warp), which also keeps the code
  // free of warp primitives on run-time lane masks (calls into the device
  // runtime under relocatable device code).
#if !defined(__HIPCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  static __device__ __forceinline__ uint32_t fetch_add(uint32_t *ptr, uint32_t value) {
    uint32_t old;
    if (BLOCK)
      asm volatile("atom.relaxed.cta.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(value) : "memory");
    else
      asm volatile("atom.relaxed.gpu.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(value) : "memory");
    return old;
  }
  static __device__ __forceinline__ void add(uint32_t *ptr, uint32_t value) {
    if (BLOCK)
      asm volatile("red.relaxed.cta.add.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
    else
      asm volatile("red.relaxed.gpu.add.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
  }
#else
  static __device__ __forceinline__ uint32_t fetch_add(uint32_t *ptr, uint32_t value) {
    return atomicAdd(ptr, value);
  }
  static __device__ __forceinline__ void add(uint32_t *ptr, uint32_t value) { atomicAdd(ptr, value); }
#endif

  // Consume protocol: the number of elements queued or still being processed
  __device__ __forceinline__ uint32_t pending() const {
    return load(pending_counter());
  }

  // Consume protocol: retires ``count`` elements popped with ``noupdate``
  __device__ __forceinline__ void task_done(uint32_t count) const {
    if (count > 0) add(pending_counter(), 0u - count);
  }

  __device__ __forceinline__ void push(const T &item) const {
    add(pending_counter(), 1u);
    const uint32_t pos = fetch_add(tail(), 1u);
    uint32_t *seq = m_seq + (pos & m_mask);
    // Full: wait for the pop of the previous lap
    DACE_GPUQUEUE_WAIT(load_acquire(seq) != pos, "push", pos, load(seq));
    m_data[pos & m_mask] = item;
    store_release(seq, pos + 1);
    add(queued(), 1u);
  }

  __device__ __forceinline__ void push(const T *items, int count) const {
    for (int i = 0; i < count; ++i) push(items[i]);
  }

  // Pops up to ``max_count`` elements without blocking; returns how many
  __device__ __forceinline__ uint32_t pop_try(T *out, uint32_t max_count,
                                              bool noupdate = false) const {
    if ((int)load(queued()) <= 0) return 0;  // No atomic when empty
#ifndef __HIPCC__
    // Admitted to up to ``max_count`` of the queued elements, the excess given
    // back; every position then claimed holds an element or is about to
    const int before = (int)fetch_add(queued(), 0u - max_count);
    const uint32_t mine = before >= (int)max_count ? max_count : (before > 0 ? before : 0);
    if (mine < max_count) add(queued(), max_count - mine);
    if (mine == 0) return 0;
    const uint32_t first = fetch_add(head(), mine);
    if (max_count <= 4) {
      // A few cells (a constant at the call site): acquire and release each
      for (uint32_t i = 0; i < mine; ++i) {
        uint32_t *seq = m_seq + ((first + i) & m_mask);
        // Admitted: the element is in or on its way
        DACE_GPUQUEUE_WAIT(load_acquire(seq) != first + i + 1, "pop", first + i, load(seq));
        out[i] = m_data[(first + i) & m_mask];
      }
      for (uint32_t i = 0; i < mine; ++i)  // Free the cells for the next lap
        store_release(m_seq + ((first + i) & m_mask), first + i + m_mask + 1);
      if (!noupdate) add(pending_counter(), 0u - mine);
      return mine;
    }
    // A chunk: poll every cell, then one fence acquires them all
    for (uint32_t i = 0; i < mine; ++i) {
      const uint32_t pos = first + i;
      DACE_GPUQUEUE_WAIT(load(m_seq + (pos & m_mask)) != pos + 1, "pop", pos,
                         load(m_seq + (pos & m_mask)));
    }
    fence_acq_rel();
    for (uint32_t i = 0; i < mine; ++i) out[i] = m_data[(first + i) & m_mask];
#else
    // Without independent thread scheduling a pop must not wait for a push of
    // its own wavefront: claim only cells that are full already
    uint32_t first, mine;
    while (true) {
      first = load(head());
      mine = 0;
      while (mine < max_count &&
             load_acquire(m_seq + ((first + mine) & m_mask)) == first + mine + 1)
        ++mine;
      if (mine == 0) return 0;
      if (atomicCAS(head(), first, first + mine) == first) break;
    }
    add(queued(), 0u - mine);
    for (uint32_t i = 0; i < mine; ++i) out[i] = m_data[(first + i) & m_mask];
#endif
    fence_acq_rel();
    for (uint32_t i = 0; i < mine; ++i)  // Free the cells for the next lap
      store(m_seq + ((first + i) & m_mask), first + i + m_mask + 1);
    if (!noupdate && mine > 0) add(pending_counter(), 0u - mine);
    return mine;
  }

  __device__ __forceinline__ bool pop_try(T &out, bool noupdate = false) const {
    return pop_try(&out, 1, noupdate) != 0;
  }

  // Block-collective variants, for a thread block that acts as one consumer: one thread
  // operates on the queue and every thread of the block sees the same result.
  static __device__ __forceinline__ bool block_leader() {
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
  }

  // Pops up to ``max_count`` elements into block-shared ``out`` and returns how many
  __device__ __forceinline__ uint32_t block_pop_try(T *out, uint32_t max_count,
                                                    bool noupdate = false) const {
    __shared__ uint32_t popped;
    __syncthreads();  // The previous chunk has been read by every thread
    if (block_leader()) popped = pop_try(out, max_count, noupdate);
    __syncthreads();
    return popped;
  }

  __device__ __forceinline__ uint32_t block_pending() const {
    __shared__ uint32_t result;
    __syncthreads();
    if (block_leader()) result = pending();
    __syncthreads();
    return result;
  }

  __device__ __forceinline__ void block_task_done(uint32_t count) const {
    __syncthreads();  // Every thread finished the task (and its pushes)
    if (block_leader() && count > 0) add(pending_counter(), 0u - count);
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

////////////////////////////////////////////////////////////
// Host memory management for GPU work queues

template <typename T>
__global__ void InitGPUQueue_kernel(GPUQueue<T> queue) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= queue.m_mask) queue.m_seq[i] = i;
  if (i < GPUQueue<T>::NUM_COUNTERS * GPUQueue<T>::COUNTER_STRIDE) queue.m_counters[i] = 0;
}

template <typename T>
GPUQueue<T> AllocGPUQueue(uint32_t capacity) {
  uint32_t size = 32;
  while (size < capacity) size <<= 1;
  GPUQueue<T> queue;
  queue.m_mask = size - 1;
  gpuMalloc(&queue.m_data, size * sizeof(T));
  gpuMalloc(&queue.m_seq, size * sizeof(uint32_t));
  gpuMalloc(&queue.m_counters, GPUQueue<T>::NUM_COUNTERS * GPUQueue<T>::COUNTER_STRIDE * sizeof(uint32_t));
  void *args[1] = {&queue};
  gpuLaunchKernel((void *)&InitGPUQueue_kernel<T>, dim3((size + 255) / 256),
                  dim3(256), args, 0, (gpuStream_t)0);
  // Kernels run on non-blocking streams, which do not wait for the default
  // stream: the queue must be set up before any of them touches it
  gpuStreamSynchronize((gpuStream_t)0);
  return queue;
}

template <typename T>
void FreeGPUStream(GPUQueue<T> &queue) {
  gpuFree(queue.m_data);
  gpuFree(queue.m_seq);
  gpuFree(queue.m_counters);
}

}  // namespace dace

#endif  // __DACE_STREAM_CUH
