// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_STREAM_H
#define __DACE_STREAM_H

#include "../../../external/moodycamel/blockingconcurrentqueue.h"

// Consume
#include <thread>
#include <atomic>
#include <vector>

#include "vector.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "cuda/stream.cuh"
#else
#include "cudainterop.h"

namespace dace {
    // Forward/mirror declaration of GPU classes
    template<typename T, bool IS_POWEROFTWO = false>
    class GPUStream
    {
    public:
        T* m_data;
        uint32_t *m_start, *m_end, *m_pending;
        uint32_t m_capacity_mask;

        GPUStream() : m_data(nullptr), m_start(nullptr), m_end(nullptr),
            m_pending(nullptr), m_capacity_mask(0) {}
        GPUStream(T* data, uint32_t capacity,
                  uint32_t *start, uint32_t *end,
                  uint32_t *pending) :
            m_data(data), m_start(start), m_end(end), m_pending(pending),
            m_capacity_mask(IS_POWEROFTWO ? (capacity - 1) : capacity)
        { }
    };

    template<typename T, bool IS_POW2>
    void FreeGPUArrayStreamView(GPUStream<T, IS_POW2>& stream)
    {
        DACE_GPU_CHECK(gpuFree(stream.m_start));
        DACE_GPU_CHECK(gpuFree(stream.m_end));
        DACE_GPU_CHECK(gpuFree(stream.m_pending));
    }

    template<typename T, bool IS_POW2>
    void FreeGPUStream(GPUStream<T, IS_POW2>& stream)
    {
        FreeGPUArrayStreamView(stream);
        DACE_GPU_CHECK(gpuFree(stream.m_data));
    }
}  // namespace dace
#endif

namespace dace {

    using moodycamel::BlockingConcurrentQueue;
    using moodycamel::ConcurrentQueueDefaultTraits;


    // Stream implementation with a direct array connection
    template <typename T, bool ALIGNED = false>
    class ArrayStreamView;
    template <typename T, bool ALIGNED = false>
    class ArrayStreamViewThreadlocal;


    template <typename T, bool ALIGNED = false>
    class Stream;

    // Performance can be increased by removing qsize, but this is necessary for
    // consume to work for now.
    template <typename T, bool ALIGNED>
    class Stream {
    protected:
        BlockingConcurrentQueue<T> m_queue;
    public:
        std::atomic<unsigned int> m_elements;

        Stream(size_t capacity = 6 * ConcurrentQueueDefaultTraits::BLOCK_SIZE) :
            m_queue(BlockingConcurrentQueue<T>(capacity)), m_elements(0) {}

        inline void pop(T& item, bool noupdate = false) {
            m_queue.wait_dequeue(item);
            if (!noupdate)
                m_elements--;
        }
        inline T pop(bool noupdate = false) {
            T item;
            pop(item, noupdate);
            return item;
        }
        inline size_t pop(T *valarr, int max_size, bool noupdate = false) {
            size_t result = m_queue.wait_dequeue_bulk(valarr, max_size);
            if (result > 0 && !noupdate)
                m_elements -= result;
            return result;
        }
        inline bool pop_try(T& output, bool noupdate = false) {
            bool result = m_queue.try_dequeue(output);
            if (result && !noupdate)
                m_elements--;
            return result;
        }
        inline size_t pop_try(T *valarr, int max_size, bool noupdate = false) {
            size_t result = m_queue.try_dequeue_bulk(valarr, max_size);
            if (result > 0 && !noupdate)
                m_elements -= result;
            return result;
        }

        inline void push(T const& val) {
            m_queue.enqueue(val);
            m_elements++;
        }
        inline void push(T&& val) {
            m_queue.enqueue(val);
            m_elements++;
        }
        inline void push(const T* valarr, int size) {
            m_queue.enqueue_bulk(valarr, size);
            m_elements += size;
        }

        template <bool A>
        void push(const ArrayStreamView<T, A>& s) {
            m_queue.enqueue_bulk(s.m_array, s.m_elements);
            m_elements += s.m_elements;
        }

        template <bool A>
        void push(const ArrayStreamViewThreadlocal<T, A>& s) {
            m_queue.enqueue_bulk(s.m_array, s.m_elements);
            m_elements += s.m_elements;
        }
    
        inline bool push_try(T const& val) {
            bool result = m_queue.try_enqueue(val);
            if (result)
                m_elements++;
            return result;
        }
        inline bool push_try(T&& val) {
            bool result = m_queue.try_enqueue(val);
            if (result)
                m_elements++;
            return result;
        }
        inline bool push_try(const T* valarr, int size) {
            bool result = m_queue.try_enqueue_bulk(valarr, size);
            if (result)
                m_elements += size;
            return result;
        }

    };

    // Stream implementation with a direct array connection
    template <typename T, bool ALIGNED>
    class ArrayStreamView {
    protected:
        T* m_array;
        std::atomic<unsigned int> m_elements;

        friend class ArrayStreamView<T, !ALIGNED>;
        friend class ArrayStreamViewThreadlocal<T, ALIGNED>;
        friend class ArrayStreamViewThreadlocal<T, !ALIGNED>;
        friend class Stream<T, ALIGNED>;
        friend class Stream<T, !ALIGNED>;

    public:
        static constexpr bool aligned = ALIGNED;

        explicit ArrayStreamView(T* sink) : m_array(sink), m_elements(0) {}

        void push(const T& element) {
            const unsigned int offset = m_elements.fetch_add(1);
            *(m_array + offset) = element;
        }

        template <int VECTOR_LEN>
        void push(const dace::vec<T, VECTOR_LEN>& element) {
            // The internal pointer type relies on the alignment of the original array
            typedef typename std::conditional<ALIGNED, vec<T, VECTOR_LEN>,
                vecu<T, VECTOR_LEN>>::type vec_t;

            const unsigned int offset = m_elements.fetch_add(VECTOR_LEN);
            *((vec_t*)(m_array + offset)) = element;
        }

        void push_if(const T& element, bool condition) {
            if (condition) {
                const unsigned int offset = m_elements.fetch_add(1);
                *(m_array + offset) = element;
            }
        }

        template <int VECTOR_LEN>
        void push_if(const dace::vec<T, VECTOR_LEN>& element,
                     const dace::vec<int, VECTOR_LEN>& mask) {
            int ppcnt = 0;
            for (int v = 0; v < VECTOR_LEN; ++v) ppcnt += (mask[v] ? 1 : 0);
            const unsigned int off = m_elements.fetch_add(ppcnt);
            T* ptr = m_array + off;
            for (int v = 0; v < VECTOR_LEN; ++v) {
                if (mask[v]) {
                    *ptr++ = element[v];
                }
            }
        }

        void push(const T* elements, unsigned int num_elements) {
            const unsigned int offset = m_elements.fetch_add(num_elements);
            std::copy(elements, elements + num_elements, m_array + offset);
        }

        template <bool A>
        void push(const ArrayStreamView<T, A>& s) {
            const unsigned int offset = m_elements.fetch_add(s.m_elements);
            std::copy(s.m_array, s.m_array + s.m_elements, m_array + offset);
        }

        template <bool A>
        void push(const ArrayStreamViewThreadlocal<T, A>& s) {
            const unsigned int offset = m_elements.fetch_add(s.m_elements);
            std::copy(s.m_array, s.m_array + s.m_elements, m_array + offset);
        }

        template <int VECTOR_LEN>
        void push(const dace::vec<T, VECTOR_LEN>* elements,
                  unsigned int num_elements) {
            // The internal pointer type relies on the alignment of the original array
            typedef typename std::conditional<ALIGNED, vec<T, VECTOR_LEN>,
                vecu<T, VECTOR_LEN>>::type vec_t;

            const unsigned int offset = m_elements.fetch_add(num_elements * VECTOR_LEN);
            std::copy(elements, elements + num_elements, (vec_t*)(m_array + offset));
        }
    };


    // Stream implementation with a direct array connection - thread-local
    template <typename T, bool ALIGNED>
    class ArrayStreamViewThreadlocal {
    protected:
        T* m_array;
        unsigned int m_elements;

        friend class ArrayStreamViewThreadlocal<T, !ALIGNED>;
        friend class ArrayStreamView<T, ALIGNED>;
        friend class ArrayStreamView<T, !ALIGNED>;
        friend class Stream<T, ALIGNED>;
        friend class Stream<T, !ALIGNED>;

    public:
        static constexpr bool aligned = ALIGNED;

        explicit ArrayStreamViewThreadlocal(T* sink) : m_array(sink), m_elements(0) {}

        void push(const T& element) {
            m_array[m_elements++] = element;
        }

        template <int VECTOR_LEN>
        void push(const dace::vec<T, VECTOR_LEN>& element) {
            // The internal pointer type relies on the alignment of the original array
            typedef typename std::conditional<ALIGNED, vec<T, VECTOR_LEN>,
                vecu<T, VECTOR_LEN>>::type vec_t;

            *((vec_t*)(m_array + m_elements)) = element;
            m_elements += VECTOR_LEN;
        }

        void push_if(const T& element, bool condition) {
            if (condition) {
                m_array[m_elements++] = element;
            }
        }

        template <int VECTOR_LEN>
        void push_if(const dace::vec<T, VECTOR_LEN>& element,
                     const dace::vec<int, VECTOR_LEN>& mask) {
            for (int v = 0; v < VECTOR_LEN; ++v) {
                if (mask[v]) {
                    m_array[m_elements++] = element[v];
                }
            }
        }
/*
        void push_if(const dace::vec<T, 4>& element, const dace::vec<int, 4>& mask) {
            for (int v = 0; v < 4; ++v) {
                if (mask[v]) {
                    m_array[m_elements++] = element[v];
                }
            }
        }

        void push_if(const dace::vec<T, 8>& element, const dace::vec<int, 8>& mask) {
            for (int v = 0; v < 8; ++v) {
                if (mask[v]) {
                    m_array[m_elements++] = element[v];
                }
            }
        }
*/
        void push(const T* elements, unsigned int num_elements) {
            std::copy(elements, elements + num_elements, m_array + m_elements);
            m_elements += num_elements;
        }

        template <bool A>
        void push(const ArrayStreamView<T, A>& s) {
            std::copy(s.m_array, s.m_array + s.m_elements, m_array + m_elements);
            m_elements += s.m_elements;
        }

        template <bool A>
        void push(const ArrayStreamViewThreadlocal<T, A>& s) {
            std::copy(s.m_array, s.m_array + s.m_elements, m_array + m_elements);
            m_elements += s.m_elements;
        }

        template <int VECTOR_LEN>
        void push(const dace::vec<T, VECTOR_LEN>* elements,
                  unsigned int num_elements) {
            // The internal pointer type relies on the alignment of the original array
            typedef typename std::conditional<ALIGNED, vec<T, VECTOR_LEN>,
                vecu<T, VECTOR_LEN>>::type vec_t;

            std::copy(elements, elements + num_elements, (vec_t*)(m_array + m_elements));
            m_elements += num_elements * VECTOR_LEN;
        }
    };

    template <int CHUNKSIZE = 1>
    struct Consume;

    template <int CHUNKSIZE>
    struct Consume {
        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED,
                  typename Functor>
        static void consume(StreamT<T, ALIGNED>& stream, unsigned num_threads,
                            Functor&& contents) {
            std::vector<std::thread> threads;
            auto thread_contents = [&](int pe) {
                T consumed_elements[CHUNKSIZE];
                while (stream.m_elements > 0) {
                    size_t elems = stream.pop_try(consumed_elements, CHUNKSIZE, true);
                    if (elems > 0) {
                        contents(pe, consumed_elements, elems);
                        stream.m_elements -= elems;
                    }
                }
            };
            for (unsigned i = 0; i < num_threads; ++i)
                threads.emplace_back(std::thread(thread_contents, i));

            for (auto& t : threads) t.join();
        }

        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED,
                  typename CondFunctor, typename Functor>
        static void consume_cond(StreamT<T, ALIGNED>& stream, unsigned num_threads,
                                 CondFunctor&& quiescence, Functor&& contents) {
            std::vector<std::thread> threads;
            auto thread_contents = [&](int pe) {
                T consumed_elements[CHUNKSIZE];
                while (!quiescence()) {
                    size_t elems = stream.pop_try(consumed_elements, CHUNKSIZE, true);
                    if (elems > 0) {
                        contents(pe, consumed_elements, elems);
                        stream.m_elements -= elems;
                    }
                }
            };
            for (unsigned i = 0; i < num_threads; ++i)
                threads.emplace_back(std::thread(thread_contents, i));

            for (auto& t : threads) t.join();
        }
    };

    // Specialization for consumption of 1 element
    template<>
    struct Consume<1> {
        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED,
                  typename Functor>
        static void consume(StreamT<T, ALIGNED>& stream, unsigned num_threads,
                            Functor&& contents) {
            std::vector<std::thread> threads;
            auto thread_contents = [&](int pe) {
                T consumed_element;
                while (stream.m_elements > 0) {
                    if (stream.pop_try(consumed_element, true)) {
                        contents(pe, consumed_element);
                        stream.m_elements--;
                    }
                }
            };
            for (unsigned i = 0; i < num_threads; ++i)
                threads.emplace_back(std::thread(thread_contents, i));

            for (auto& t : threads) t.join();
        }

        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED,
                  typename CondFunctor, typename Functor>
        static void consume_cond(StreamT<T, ALIGNED>& stream, unsigned num_threads,
                                 CondFunctor&& quiescence, Functor&& contents) {
            std::vector<std::thread> threads;
            auto thread_contents = [&](int pe) {
                T consumed_element;
                while (!quiescence()) {
                    if (stream.pop_try(consumed_element, true)) {
                        contents(pe, consumed_element);
                        stream.m_elements--;
                    }
                }
            };
            for (unsigned i = 0; i < num_threads; ++i)
                threads.emplace_back(std::thread(thread_contents, i));

            for (auto& t : threads) t.join();
        }
    };

}  // namespace dace

#endif  // __DACE_STREAM_H
