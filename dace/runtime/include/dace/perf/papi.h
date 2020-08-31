// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_PERF_PAPI_H
#define __DACE_PERF_PAPI_H

#include <string>
#include <future>
#include <mutex>
#include <omp.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <iostream>
#include <memory>
#include <chrono>
#include <array>

#include <papi.h>
#ifdef __WIN32__
    #include <processthreadsapi.h>
#else
    #include <sched.h>
#endif

#ifdef __x86_64__ // We don't support i386 (macro: __i386__)
  // Implemented in gcc and clang
  #ifdef __GNUC__
    #include <x86intrin.h>
    #define DACE_PERF_mfence _mm_mfence()
  #else
    #define DACE_PERF_mfence /* Default: NO FENCE AVAILABLE*/
  #endif
#elif defined(_WIN64)
  #include <windows.h>
  #define DACE_PERF_mfence MemoryBarrier()
#else
  #define DACE_PERF_mfence /* Default: NO FENCE AVAILABLE*/
#endif

#include "reporting.h"


#define LOG_ERRORS
#define TEST_ALIGNMENT
#define CHECK_BOUNDS
// Disable runtime byte movement recording. Defining this can reduce cache line ping pong.
#define NO_RUNTIME_BYTEMOVEMENT_ACCUMULATION

#ifndef OVERHEAD_REPETITIONS
#define OVERHEAD_REPETITIONS 100
#endif

#ifndef DACE_INSTRUMENTATION_SUPERSECTION_FLUSH_THRESHOLD
// Define a threshold for flushing (= don't flush at every supersection, but only if the buffer is filled to a certain
// percentage.)
#define DACE_INSTRUMENTATION_SUPERSECTION_FLUSH_THRESHOLD 0.5f
#endif

namespace dace {
namespace perf {

using byte_counter_size_t = uint64_t;
constexpr uint32_t invalid_node_id = std::numeric_limits<uint32_t>::max();
constexpr size_t CACHE_LINE_SIZE = 64;

void LogError(const char *format, ...)
{
#ifdef LOG_ERRORS
    FILE* fp = fopen("errors.log", "a");
    if(fp) {
        va_list args;
        va_start(args, format);
        vfprintf(fp, format, args);
        va_end(args);
        fclose(fp);
    }
#endif
}

template<int... events> class PAPIPerf;
template<int... events> class PAPIValueSet;
template<int... events> class PAPIValueStore;

template<typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedElement
{
public:

    static constexpr size_t alignment_padding = (sizeof(T) == Alignment) ? (0) : (Alignment - (sizeof(T) & (Alignment - 1)));

    AlignedElement() {}
    AlignedElement(const T& x) : m_elem(x) {}
    AlignedElement(T&& x) : m_elem(x) {}

    operator T&() {
        static_assert(sizeof(*this) % Alignment == 0);
        return m_elem;
    }

    ~AlignedElement() {}

private:
    T m_elem;
    uint8_t m_padding[alignment_padding];
};

template<typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedContainer
{
public:
    static constexpr auto realsize = (sizeof(T) == Alignment) ? (sizeof(T)) : ((sizeof(T) + Alignment) & ~(Alignment - 1));
    static_assert(realsize >= sizeof(T), "Realsize is less than an object!");
    static_assert(realsize <= sizeof(T) + Alignment, "Realsize larger than necessary");
    static_assert(realsize == sizeof(AlignedElement<T, Alignment>), "realsize should be identical to aligned element size");

    AlignedContainer()
        : m_rawdat(nullptr), align_offset(std::numeric_limits<size_t>::max()), m_size(0), m_alloc_size(0)
    {
        
    }
    
    ~AlignedContainer()
    {
        clear();
    }

    void resize(size_t n)
    {
        LogError("Buffer resized\n");
        clear();
        if(n == m_size)
        {
            initialize_elements(); // Reset the elements
            return;
        }
        m_alloc_size = (n + 1) * realsize * sizeof(*m_rawdat.get());
        m_rawdat.reset(new uint8_t[m_alloc_size]);
        if(m_rawdat == nullptr)
        {
            LogError("Failed to allocate buffer\n");
        }

        align_offset = Alignment - (reinterpret_cast<uintptr_t>(m_rawdat.get()) & (Alignment - 1));

        m_size = n;
        initialize_elements();
    }

    void initialize_elements()
    {
        auto* data = elementArray();
        assert(data != nullptr);
        for(size_t i = 0; i < m_size; ++i)
        {
            // The following 2 lines should have the same effect (line 1 is copy-assigned, line 2 is using the placement operator new).
            // The 2nd line should be more efficient
            //data[i] = AlignedElement<T, Alignment>();
            new (data + i) AlignedElement<T, Alignment>();

            #ifdef CHECK_BOUNDS
            assert((uint8_t*)&data[i] <= m_rawdat.get() + m_alloc_size);
            #endif
        }
    }

    void clear()
    {
        if(align_offset == std::numeric_limits<size_t>::max())
            return;
        for(size_t i = 0; i < m_size; ++i)
        {
            (*this)[i].~T();
        }
    }

    size_t size() const {
        return m_size;
    }

    AlignedElement<T, Alignment>* elementArray() const
    {
        #ifdef CHECK_BOUNDS
        assert(align_offset != std::numeric_limits<size_t>::max());
        assert(m_rawdat.get() != nullptr);
        #endif
        auto* ptr = reinterpret_cast<AlignedElement<T, Alignment>*>(m_rawdat.get() + align_offset);
        #ifdef CHECK_BOUNDS
        assert((uint8_t*)ptr > m_rawdat.get() && (uint8_t*)ptr < m_rawdat.get() + m_alloc_size && "out of bounds");
        #endif
        return ptr;
    }

    T& operator[](size_t index)
    {
        auto* ptr = &(elementArray()[index]);
        #ifdef CHECK_BOUNDS
        assert((uint8_t*)ptr > m_rawdat.get() && (uint8_t*)ptr + 1 < m_rawdat.get() + m_alloc_size && "out of bounds");
        #endif
        return *ptr;
    }

private:
    std::unique_ptr<uint8_t> m_rawdat;
    size_t align_offset;
    size_t m_size;
    size_t m_alloc_size;
};

class thread_lock_context_t
{
public:
    thread_lock_context_t()
        : notified(true), iteration(0)
    {
        
    }
    std::mutex mutex;
    std::condition_variable cond_var;
    std::atomic_bool notified;
    std::atomic<size_t> iteration;

};


int64_t getThreadID()
{
#ifdef __linux__
    const auto thread_id = sched_getcpu();
#elif defined(__WIN32__)
    // More than 64 CPUs are put into processor groups, so the normal GetCurrentProcessorNumber() does not work
    // with > 64 Threads.
    PROCESSOR_NUMBER pn;
    GetCurrentProcessorNumberEx(&pn);
    const auto thread_id = static_cast<size_t>(pn.Group) * 64 + static_cast<size_t>(pn.Number) ;
#else
    #error Unsupported platform, provide code to get hardware thread number here.
#endif
    return thread_id;
}


class ThreadLockReleaser
{
public:
    ThreadLockReleaser(thread_lock_context_t& ctx)
        : m_ctx(ctx), m_iteration(ctx.iteration)
    {}

    ~ThreadLockReleaser()
    {
        m_ctx.notified = true;
        m_ctx.cond_var.notify_one();
    }

    size_t getIteration() const {
        return m_iteration;
    }

    thread_lock_context_t& m_ctx;
    size_t m_iteration;
};

// Lock a task to a thread
void lockThreadID(unsigned long core)
{
#ifdef __linux__
    auto threadid = pthread_self();
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(core, &cpu_set);
    auto done = pthread_setaffinity_np(threadid, sizeof(cpu_set_t), &cpu_set);
    if(done != 0)
    {
        std::cout << "Failed pthread_setaffinity_np with code " << done << std::endl;
        exit(-1);
    }
#else
    #error Implement lockThreadID function to lock a task to a given thread.
#endif
}

// This class provides an interface to lock threads to prevent multiple concurrent measurements on the same core
class ThreadLockProvider
{
public:
    ThreadLockProvider()
    {
        size_t thread_count = std::thread::hardware_concurrency();
        m_contexts.resize(thread_count);

        m_global_iteration = 0;
    }

    size_t getAndIncreaseCounter() 
    {
        size_t old = m_global_iteration;

        while(!m_global_iteration.compare_exchange_weak(old, old + 1));

        return old;
    }

    ThreadLockReleaser enqueue()
    {
        // Parameter is implicit through std::this_thread (or equivalent)

        const auto thread_id = dace::perf::getThreadID();

        // Lock it
        dace::perf::lockThreadID(thread_id);

        auto& ctx = m_contexts[thread_id];
        std::unique_lock<std::mutex> lock(ctx.mutex);

        bool old = true;

        while(!ctx.notified.compare_exchange_weak(old, false))
        {
            old = true;
            ctx.cond_var.wait(lock);
        }
        ++(ctx.iteration);
        
        // This instance is now blocking the thread. When the return value is destroyed, it will no longer block
        return ThreadLockReleaser(ctx);
    }
private:
    AlignedContainer<thread_lock_context_t> m_contexts;
    std::atomic<size_t> m_global_iteration;
};

class PAPI
{
public:

    static void init()
    {
        init_library();
        init_threads();
    }

    static void init_library()
    {
        const auto r_init = PAPI_library_init(PAPI_VER_CURRENT);
#ifndef SKIP_RETVAL_CHECKS
        if(r_init != PAPI_VER_CURRENT && r_init != PAPI_OK)
        {
            LogError("init_library error: %d\n", r_init);
        }
#endif
    }

    static void init_threads()
    {
        const auto r_init = ::PAPI_thread_init((long unsigned int (*)())omp_get_thread_num);
#ifndef SKIP_RETVAL_CHECKS
        if(r_init != PAPI_VER_CURRENT && r_init != PAPI_OK)
        {
            LogError("init_threads error: %d\n", r_init);
        }
#endif
    }

    static void init_multiplexing()
    {
        const auto r_init = ::PAPI_multiplex_init();
        #ifndef SKIP_RETVAL_CHECKS
        if(r_init != PAPI_VER_CURRENT && r_init != PAPI_OK)
        {
            LogError("init_multiplexing error: %d\n", r_init);
        }
        #endif
    }
};


enum class ValueSetType : uint32_t
{
    Default = 0,
    Raw,
    OMP_marker_parfor_start,
    OMP_marker_parfor_end,

    marker_section_start,
    marker_supersection_start,

    Copy,
    CounterOverride,
    OverheadComp,
};

template<int... events>
class PAPIValueSet
{
public:

    PAPIValueSet() : m_flags(ValueSetType::Default) { }

    PAPIValueSet(uint32_t nodeid, uint32_t coreid, uint32_t iteration, ValueSetType flags = ValueSetType::Default)
        : m_nodeid(nodeid), m_coreid(coreid), m_iteration(iteration), m_flags(flags) { }

    ~PAPIValueSet() { }

    void report(Report& rep, int* event_override = nullptr) const
    {
        int event_tags[] = {events...};
        if(event_override)
        {
            std::copy(event_override, event_override + sizeof...(events), event_tags);
        }
        std::string entry_name = "papi_entry";

        if(m_flags == ValueSetType::Default || m_flags == ValueSetType::Copy)
            entry_name += " (" + std::to_string(m_nodeid) + ", " + std::to_string(m_coreid) + ", " +
                          std::to_string(m_iteration) + ", " + std::to_string((int)m_flags) + ") ";
        else if(m_flags == ValueSetType::OverheadComp)
            entry_name = "papi_overhead";
        else if(m_flags == ValueSetType::marker_section_start)
        {
            entry_name = "papi_section_start (node " + std::to_string(m_nodeid) + ", core " + std::to_string(m_coreid) + ") ";
            rep.add((entry_name + "bytes").c_str(), static_cast<double>(m_values[0]));
            if(m_values[1] != 0)
                rep.add((entry_name + "input_bytes").c_str(), static_cast<double>(m_values[1]));
            return;
        }
        else if(m_flags == ValueSetType::OMP_marker_parfor_start)
        {
            // Markers have no values attached to them, so they are not included in the report
            // entry_name = "papi_loop_start";
            return;
        }
        else if(m_flags == ValueSetType::marker_supersection_start)
        {
            // entry_name = "papi_supersection_start (node " + std::to_string(m_nodeid) + ")";
            return;
        }

        size_t i = 0;

        for(const auto& e : event_tags)
        {
            if(e == 0) continue; // Skip unnecessary/invalid entries
            rep.add((entry_name + std::to_string(e)).c_str(), static_cast<double>(m_values[i]));
            ++i;
        }
    }

    // Return a reference to the value-array
    long long (&store())[sizeof...(events)]
    {
        return m_values;
    }

    const long long (&cstore() const)[sizeof...(events)]
    {
        return m_values;
    }

    std::array<int, sizeof...(events)> event_array() const
    {
        return {events...};
    }

    const ValueSetType& flags() const { return m_flags; }

private:
    // Note: If we keep adding variables we will at some point run into issues.
    // We aim for 64 bytes, so we have enough room.
    alignas(CACHE_LINE_SIZE) long long m_values[sizeof...(events)];
    uint32_t m_nodeid;      // The node in the graph
    uint32_t m_coreid;      // The ID of the core.
    uint32_t m_iteration;   // The iteration (in a given loop)
    ValueSetType m_flags;   // The flags (mode) for this value set.
};

// Class to store the value sets during execution and writing it to disk after execution
template<int... events>
class PAPIValueStore
{
protected:
    std::recursive_mutex m_vec_mutex;
    //aligned_vector<PAPIValueSet<events...>> m_values;
    //std::vector<PAPIValueSet<events...>> m_values;
    AlignedContainer<PAPIValueSet<events...>> m_values;

    std::atomic<size_t> m_insertion_position;
    std::atomic<uint64_t> m_contention_value;
    std::atomic<byte_counter_size_t> m_moved_bytes;

    Report& m_report;
public:
    static constexpr size_t store_reserve_size = 4096 * 1024;
    PAPIValueStore(Report& report) : m_report(report)
    {
        assert(m_moved_bytes.is_lock_free() && "Moved byte counter is not lockfree!");
        // Skip first few growth operations
        //m_values.reserve(store_reserve_size);
        m_values.resize(store_reserve_size);

        m_insertion_position = 0;
        m_moved_bytes = 0;
        m_contention_value = 0;

        LogError("Value store created\n");
    }

    ~PAPIValueStore()
    {
        if (this->m_insertion_position > 0)
            flush();
    }

    // Does a couple of empty runs (starting and immediately stopping counters) to get a rough estimate of overheads
    void getMeasuredOverhead()
    {
        auto _min = PAPIValueSet<events...>(-1, 0, 0, ValueSetType::OverheadComp);
        auto _max = PAPIValueSet<events...>();

        // flush values
        for(auto& x : _min.store()) x = std::numeric_limits<long long>::max();
        for(auto& x : _max.store()) x = std::numeric_limits<long long>::min();

        auto set_min_max = [&](const auto& vs) {
            for(size_t i = 0; i < sizeof(vs.cstore()) / sizeof(vs.cstore()[0]); ++i)
            {
                _min.store()[i] = std::min(_min.store()[i], vs.cstore()[i]);
                _max.store()[i] = std::max(_max.store()[i], vs.cstore()[i]);
            }
        };

        for(auto i = 0; i < OVERHEAD_REPETITIONS; ++i) {
            PAPIPerf<events...> perfctr;
            PAPIValueSet<events...> vs;
            perfctr.enterCritical();
            perfctr.leaveCritical(vs);

            set_min_max(vs);   
        }

        addEntry(_min);
        
    }

    // Provides a thread-safe implementation to increase a counter representing the bytes moved
    inline void addBytesMoved(byte_counter_size_t size)
    {
        #ifndef NO_RUNTIME_BYTEMOVEMENT_ACCUMULATION
        byte_counter_size_t oldval = m_moved_bytes;
        byte_counter_size_t newval;
        do {
            newval = oldval + size;
        } while(!m_moved_bytes.compare_exchange_strong(oldval, newval));
        #endif
    }

    inline byte_counter_size_t collectBytesMoved()
    {
        byte_counter_size_t oldval = m_moved_bytes;
        byte_counter_size_t newval = 0;
        do {
            // Nothing
        } while(!m_moved_bytes.compare_exchange_strong(oldval, newval));
        return oldval;
    }

    void flush()
    {
        LogError("Flushing with pos %ld\n", size_t(this->m_insertion_position));

        // We want to store information about what we actually measured
        int event_override[sizeof...(events)];
        bool override_events = false;
        for(size_t i = 0; i < m_insertion_position; ++i)
        {
            const auto& e = m_values[i];
            if(e.flags() == ValueSetType::CounterOverride)
            {
                // We have to override the counter type
                for(size_t i = 0; i < sizeof...(events); ++i)
                {
                    event_override[i] = e.cstore()[i];
                }
                override_events = true;
            }
            else
            {
                if(override_events)
                {
                    e.report(this->m_report, event_override);
                    override_events = false;
                }
                else
                {
                    e.report(this->m_report);
                }
            }
        }
        if(m_insertion_position > 0)
        {
#ifndef NO_RUNTIME_BYTEMOVEMENT_ACCUMULATION
            byte_counter_size_t bm = collectBytesMoved();
            this->m_report.add("papi_moved_bytes", static_cast<double>(bm));
#endif
            // Also store contention
            uint64_t cont = 0;
            cont = m_contention_value.exchange(cont);
            
            if(cont != 0)
                this->m_report.add("papi_contention", static_cast<double>(cont));
        }
        m_values.clear();
        m_values.resize(store_reserve_size);
        m_insertion_position = 0;
    }

    // This is to provide a default sync point. Its effect on the output can be disregarded
    void markSuperSectionStart(uint32_t nodeid, ValueSetType flags = ValueSetType::marker_supersection_start)
    {
        if(this->m_insertion_position >= static_cast<size_t>(store_reserve_size * DACE_INSTRUMENTATION_SUPERSECTION_FLUSH_THRESHOLD))
            flush();
        
        PAPIValueSet<events...> set(nodeid, 0, 0, flags);
        addEntry(set);
    }

    // This marks sections in a threadsafe way. In principle, instead of being a "barrier" syncing threads, it now just guarantees serial properties per thread
    void markSectionStart(uint32_t nodeid, long long SizeInBytes, long long InputSize, uint32_t threadid, uint32_t iteration = 0, ValueSetType flags = ValueSetType::marker_section_start)
    {
        // Difference SizeInBytes and InputSize: InputSize is just the amount of bytes moved INTO the section, while sizeInBytes is the amount of bytes MOVED inside a section (without reuses of the same data)
        PAPIValueSet<events...> set(nodeid, threadid, iteration, flags);
        set.store()[0] = SizeInBytes; // Use the first slot for memory information
        set.store()[1] = InputSize; // Use the second slot for memory input information
        static_assert(sizeof...(events) >= 2, "Must have at least 2 counters specified");
        addEntry(set);
    }

    template<int slots = 1>
    size_t getNewSlotPosition()
    {
        size_t pos = 0;
        bool r_ex = false;
        // Lock-free
        do {
            size_t oldpos = m_insertion_position;
            pos = oldpos + slots;

            if(pos >= m_values.size())
            {
                LogError("Flushing in measurement section\n");

                // To keep it running, we just have to flush. For this, we should lock

                // Ideally, we only let the main thread flush (to avoid excess memory movements, especially with NUMA)
                // However, it's hard to guarantee termination like this. (What if the main thread is done and will never call getNewSlotPosition())
                if(m_vec_mutex.try_lock()) {
                    // We got the lock, so we can flush
                    flush();
                    m_vec_mutex.unlock();
                }
                else {
                    // We didn't get the lock, which means that somebody else is already flushing.
                    // Since we lost a lot of time already, we can just use the lock to wait instead of spinlocking.
                    LogError("Waiting for values to be written\n");
                    m_vec_mutex.lock();
                    m_vec_mutex.unlock();
                }

                // We always have to try again to get the new (correct) position
                continue;
            }
            
            r_ex = m_insertion_position.compare_exchange_weak(oldpos, pos);
            pos = oldpos;
            if(!r_ex)
            {
                ++m_contention_value;
            }
        } while(!r_ex);

        return pos;
    }

    PAPIValueSet<events...>& addEntry(const PAPIValueSet<events...>& val)
    {
        auto pos = getNewSlotPosition();

        m_values[pos] = val;

        return m_values[pos];
    }

    template<int... counterevents>
    PAPIValueSet<events...>& getNewValueSet(const PAPIPerf<counterevents...>& perf, uint32_t nodeid, uint32_t coreid, uint32_t iteration, ValueSetType type = ValueSetType::Default)
    {
        // If counterevents and store events are not the same, we have an issue.
        // The value set must have the same arguments as the store, so it will always print the same counter ids.
        // But in this case, the counterids are not the same. We therefore have to mark the entries as invalid so the store can deal with it.

        static_assert(sizeof...(counterevents) <= sizeof...(events), "Counter event size must not exceed store size");

        int codes[] = {counterevents...};
        int storecodes[] = {events...};
        bool correct_subset = true;
        for(size_t i = 0; i < sizeof...(counterevents); ++i)
        {
            if(codes[i] != storecodes[i])
            {
                correct_subset = false;
                break;
            }
        }
        if(correct_subset)
        {
            // If we have a good subset (same order, same start, same entries), we can skip this expensive operation.
            return __impl_getNewValueSet(nodeid, coreid, iteration, type);
        }

        // Get 2 slots
        auto pos = getNewSlotPosition<2>();

        // Mark the first one to override the counters.
        m_values[pos] = PAPIValueSet<events...>(nodeid, coreid, iteration, ValueSetType::CounterOverride);

        // Store the events in the "value"-fields. Write 0 to unused fields
        for(size_t i = 0; i < sizeof...(counterevents); ++i)
        {
            m_values[pos].store()[i] = codes[i];
        }
        for(size_t i = sizeof...(counterevents); i < sizeof...(events); ++i)
        {
            m_values[pos].store()[i] = 0;
        }
    }

    inline PAPIValueSet<events...>& getNewValueSet(const PAPIPerf<events...>& perf, uint32_t nodeid, uint32_t coreid, uint32_t iteration, ValueSetType type = ValueSetType::Default)
    {
        return __impl_getNewValueSet(nodeid, coreid, iteration, type);
    }

    PAPIValueSet<events...>& __impl_getNewValueSet(uint32_t nodeid, uint32_t coreid, uint32_t iteration, ValueSetType type = ValueSetType::Default)
    {
        auto& retval = addEntry(PAPIValueSet<events...>(nodeid, coreid, iteration, type));
#ifdef TEST_ALIGNMENT
        uintptr_t val = (uintptr_t)&retval;
        auto lower_bits = val & (CACHE_LINE_SIZE - 1);
        if(lower_bits != 0)
        {
            LogError("ERROR: Values not aligned. Expected lower_bits=0, got %d\n", lower_bits);
        }
        assert(lower_bits == 0);
#endif
        return retval;
    }
    
    PAPIValueSet<events...>& getNewValueSet()
    {
        return getNewValueSet(PAPIPerf<events...>(), 0, 0, 0);
    }
};


template<int... events>
class PAPIPerf
{
public:

    PAPIPerf(const bool multiplexing = false)
        : m_event_set(PAPI_NULL)
    {
        // Need to synchronize accesses because papi is not threadsafe...
        //std::lock_guard<std::recursive_mutex> guard(papi_mutex);
        auto r_create_eventset = PAPI_create_eventset(&m_event_set);
        #ifndef SKIP_RETVAL_CHECKS
        if(r_create_eventset != PAPI_OK)
        {
            LogError("Failed to create event set with code %d\n", r_create_eventset);
        }
        #endif

        if(multiplexing)
        {
            // We need this because multiplexing will otherwise act up.
            PAPI_assign_eventset_component(m_event_set, 0);
            this->enable_multiplexing();
        }
        int evarr[] = {events...};
        auto r_add_events = PAPI_add_events(m_event_set, evarr, sizeof...(events));
        #ifndef SKIP_RETVAL_CHECKS
        if(r_add_events != PAPI_OK)
        {
            PAPI_cleanup_eventset(m_event_set);
            LogError("Failed to add events to event set with code %d\n", r_add_events);
        }
        #endif
#ifdef PAPI_EXPLICIT_THREADS
        const auto r_reg_thread = PAPI_register_thread();
        #ifndef SKIP_RETVAL_CHECKS
        if(r_reg_thread != PAPI_OK)
        {
            LogError("Failed to register thread, code %d\n", r_reg_thread);
        }
        #endif
#endif
    }

    

    ~PAPIPerf()
    {
        if(m_event_set != PAPI_NULL)
        {
            PAPI_cleanup_eventset(m_event_set);
            PAPI_destroy_eventset(&m_event_set);
#ifdef PAPI_EXPLICIT_THREADS
            PAPI_unregister_thread();
#endif
        }

    }

protected:
    void enable_multiplexing()
    {
        const auto r_multiplex = PAPI_set_multiplex(m_event_set);
        if(r_multiplex != PAPI_OK)
        {
            std::cerr << "Failed to enable multiplexing, code " << r_multiplex << std::endl;
            exit(-1);
        }
    }
public:

    static PAPIValueSet<events...> ValueSet()
    {
        return PAPIValueSet<events...>();
    }

    void enterCritical()
    {
        static bool error_reported = false;
        DACE_PERF_mfence; // Fence before starting to keep misses outside
        const auto r_start = PAPI_start(m_event_set);
        #ifndef SKIP_RETVAL_CHECKS
        if(r_start != PAPI_OK && !error_reported) {
            LogError("Failed to start counters with code %d\n", r_start);
            error_reported = true;
        }
        #endif
    }

    template<int... e>
    void leaveCritical(PAPIValueSet<e...>& values)
    {
        static bool error_reported = false;
        constexpr auto num_events = sizeof...(events);
        
        // Make sure we have the correct sizes
        static_assert(sizeof...(e) >= num_events); 

        // Fence before stopping to keep misses inside
        DACE_PERF_mfence;
        const auto r_stop = PAPI_stop(m_event_set, values.store());
        #ifndef SKIP_RETVAL_CHECKS
        if(r_stop != PAPI_OK && !error_reported)
        {
            LogError("Failed to stop counters with code %d\n", r_stop);
            error_reported = true;
        }
        #endif
        // Since we allow storing less, we have to make sure there's no stale data in the value store. Set to an invalid value, i.e. max.
        for(auto i = num_events; i < sizeof...(e); ++i)
        {
            values.store()[i] = std::numeric_limits<long long>::max();
        }
    }
    
private:
    int m_event_set;
};

}  // namespace perf
}  // namespace dace

#endif  // __DACE_PERF_PAPI_H
