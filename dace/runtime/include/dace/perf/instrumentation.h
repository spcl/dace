#pragma once


#include <stdio.h>
#include <stdlib.h>
#include <papi.h>
#include <string>
#include <future>
#include <mutex>
#include <omp.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <array>

#ifdef __WIN32__
#include <processthreadsapi.h>
#endif


#ifdef __x86_64__ // We don't support i386 (macro: __i386__)
#ifdef __GNUC__
#include <x86intrin.h>
#define DACE_PERF_mfence _mm_mfence()
#else
// Left #TODO for other compilers
#define DACE_PERF_mfence /* Default: NO FENCE AVAILABLE*/

#endif
#else
#define DACE_PERF_mfence /* Default: NO FENCE AVAILABLE*/
#endif

#ifndef DACE_INSTRUMENTATION_FAST_AND_DANGEROUS
#define TEST_ALIGNMENT
#endif
//#define PAPI_EXPLICIT_THREADS // Define to use explicit thread assignment. Docs say it's not necessary.

#ifdef DACE_INSTRUMENTATION_FAST_AND_DANGEROUS
#define SKIP_RETVAL_CHECKS
#endif
#ifndef DACE_INSTRUMENTATION_FAST_AND_DANGEROUS
#define CHECK_BOUNDS
#endif
//#define LOG_ERRORS // define to create errors.log
#define FAST_ASSERTS // disable some of the slower asserts.

#define NO_RUNTIME_BYTEMOVEMENT_ACCUMULATION // Define to disable byte movement recording. Defining this can reduce cache line ping pong

#ifndef OVERHEAD_REPETITIONS
#define OVERHEAD_REPETITIONS 100
#endif

#ifndef DACE_INSTRUMENTATION_SUPERSECTION_FLUSH_THRESHOLD
// Define a threshhold for flushing (= don't flush at every supersection, but only if the buffer is filled to a certain percentage.)
#define DACE_INSTRUMENTATION_SUPERSECTION_FLUSH_THRESHOLD 0.5f
#endif

//#define ASSIGN_COMPONENT // Assigns the component explicitly. This should not be enabled for 2 reasons: 1) PAPI_start() already does this, and 2) there might be a tiny to medium overhead when enabling twice
namespace dace_perf
{

    constexpr uint32_t invalid_node_id = std::numeric_limits<uint32_t>::max();

void logError(const std::string& str) 
{
    #ifdef LOG_ERRORS
    FILE* f = fopen("errors.log", "a");
    if(f) {
        fprintf(f, "%s\n", str.c_str());
        fclose(f);
    }
    #endif
}

template<int... events> class PAPIPerfLowLevel;
template<bool, int... events> class PAPIValueSetInternal;
template<int... events> class PAPIValueStore;

template<int... events>
using PAPIValueSet = PAPIValueSetInternal<true, events...>;

template<int... events> using PAPIPerf = PAPIPerfLowLevel<events...>;



constexpr size_t CACHE_LINE_SIZE = 64;

template<typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedElement
{
public:

    static constexpr size_t alignment_padding = (sizeof(T) == Alignment) ? (0) : (Alignment - (sizeof(T) & (Alignment - 1)));

    AlignedElement() = default;
    AlignedElement(const T& x) : m_elem(x) {}
    AlignedElement(T&& x) : m_elem(x) {}

    operator T&() {
        static_assert(sizeof(*this) % Alignment == 0, "Aligned Element is bugged");
        return m_elem;
    }

    ~AlignedElement() = default;

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
        logError("Buffer resized");
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
            logError("Failed to allocate buffer");
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
    //uint8_t* m_rawdat;
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
    #elif __WIN32__
    // More than 64 CPUs are put into processor groups, so the normal GetCurrentProcessorNumber() does not work with > 64 Threads.
    PROCESSOR_NUMBER pn;
    GetCurrentProcessorNumberEx(&pn);
    const auto thread_id = static_cast<size_t>(pn.Group) * 64 + static_cast<size_t>(pn.Number) ;
    #else
    #error Unsupported platform (maybe MacOS?), provide code to get hardware thread number here.
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
        // Notify the blocking threads

        std::cout << "Releasing thread " << dace_perf::getThreadID() << std::endl;
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
    #error Implement this lockThreadID() function to lock a task to a given thread.
    #endif
}

// This class provides an interface to lock threads in order to prevent multiple concurrent measurements on the same core
class ThreadLockProvider
{
public:
    ThreadLockProvider()
    {
        //size_t thread_count = omp_get_num_threads();
        size_t thread_count = std::thread::hardware_concurrency();
        std::cout << "Creating locks for " << thread_count << " threads." << std::endl;
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

        const auto thread_id = dace_perf::getThreadID();

        // Lock it
        dace_perf::lockThreadID(thread_id);

        std::cout << "Running on thread " << thread_id << std::endl;
        
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

        logError("Papi initialized");
    }

    static void init_library()
    {
        const auto r_init = PAPI_library_init(PAPI_VER_CURRENT);
        #ifndef SKIP_RETVAL_CHECKS
        if(r_init != PAPI_VER_CURRENT && r_init != PAPI_OK)
        {
            std::cerr << "Failed to init PAPI" << std::endl;
            PAPI_perror("Error: ");
        }
        #endif
    }

    static void init_threads()
    {
        const auto r_init = ::PAPI_thread_init((long unsigned int (*)())omp_get_thread_num);
        #ifndef SKIP_RETVAL_CHECKS
        if(r_init != PAPI_VER_CURRENT && r_init != PAPI_OK)
        {
            std::cerr << "Failed to init PAPI threads code " << r_init << std::endl;
            PAPI_perror("Error: ");
        }
        #endif
    }

    static void init_multiplexing()
    {
        const auto r_init = ::PAPI_multiplex_init();
        #ifndef SKIP_RETVAL_CHECKS
        if(r_init != PAPI_VER_CURRENT && r_init != PAPI_OK)
        {
            std::cerr << "Failed to init PAPI multiplexing, code " << r_init << std::endl;
            PAPI_perror("Error: ");
        }
        #endif
    }

    static double getTimePerCycle();
private:
};


enum class ValueSetType
    : uint32_t
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

template<bool standalone, int... events>
class PAPIValueSetInternal
{
public:

    PAPIValueSetInternal()
        : m_flags(ValueSetType::Default)
    {

    }

    PAPIValueSetInternal(uint32_t nodeid, uint32_t coreid, uint32_t iteration, ValueSetType flags = ValueSetType::Default)
        : m_nodeid(nodeid), m_coreid(coreid), m_iteration(iteration), m_flags(flags)
    {

    }

    ~PAPIValueSetInternal()
    {
        if(standalone)
        {
            // report in destructor
            std::cout << "Value set destroyed" << std::endl;
            size_t index = 0;
            PAPI_event_info_t info;
            for(const auto& e : {events...})
            {
                PAPI_get_event_info(e, &info);
                std::cout << info.symbol << ": " << m_values[index] << std::endl;
                ++index;
            }
        }
    }

    std::string toStoreFormat(int* event_override = nullptr) const
    {
        int event_tags[] = {events...};
        if(event_override)
        {
            std::copy(event_override, event_override + sizeof...(events), event_tags);
        }
        std::string ret = "# entry";

        if(m_flags == ValueSetType::Default || m_flags == ValueSetType::Copy)
        {
            ret += " (" + std::to_string(m_nodeid) + ", " + std::to_string(m_coreid) + ", " + std::to_string(m_iteration) + ", " + std::to_string((int)m_flags) + ")\n";
        }
        else if(m_flags == ValueSetType::OverheadComp)
        {
            ret = "# Overhead comp\n";
            // Values are written out below.
        }
        else if(m_flags == ValueSetType::OMP_marker_parfor_start)
        {
            ret = "# LOOP START\n";
            return ret;
        }
        else if(m_flags == ValueSetType::marker_section_start)
        {
            ret = "# Section start (node " + std::to_string(m_nodeid) + ", core " + std::to_string(m_coreid) + ")\n";
            ret += "bytes: " + std::to_string(m_values[0]) + "\n";
            if(m_values[1] != 0) // The input data (stored in slot 1) is implicit-0, so only add it if it adds information
                ret += "input_bytes: " + std::to_string(m_values[1]) + "\n";
            return ret;
        }
        else if(m_flags == ValueSetType::marker_supersection_start)
        {
            ret = "# Supersection start (node " + std::to_string(m_nodeid) + ")\n";
            return ret;
        }
        else
            ret += "\n";

        //constexpr auto evcount = sizeof...(events);
        size_t i = 0;

        for(const auto& e : event_tags)
        {
            if(e == 0) continue; // Skip unnecessary/invalid entries
            ret += std::to_string(e) + ": " + std::to_string(m_values[i]) + "\n";
            ++i;
        }
        return ret;
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
    // Note: If we keep adding variables we will at some point run into issues. Right now, we should probably aim for 64 bytes, so we have enough room for 6 PMCs.
    alignas(CACHE_LINE_SIZE) long long m_values[sizeof...(events)];
    uint32_t m_nodeid;      // The node in the graph
    uint32_t m_coreid;      // The ID of the core.
    uint32_t m_iteration;   // The iteration (in a given loop)
    ValueSetType m_flags;   // The flags (mode) for this value set.
};


    constexpr auto store_path = "instrumentation_results.txt";
// Class to store the value sets during execution and writing it to disk after execution
// For now, the store is dynamic
template<int... events>
class PAPIValueStore
{
    using byte_counter_size_t = uint64_t;
public:
    static constexpr size_t store_reserve_size = 4096 * 1024;
    PAPIValueStore()
    {
        assert(m_moved_bytes.is_lock_free() && "Moved byte counter is not lockfree!");
        // Skip first few growth operations
        //m_values.reserve(store_reserve_size);
        m_values.resize(store_reserve_size);

        // Remove previous instrumentation data
        //m_store_file = fopen(store_path, "wb");
        m_store_file = fopen(store_path, "ab");
        if(!m_store_file)
        {
            std::cerr << "Failed to open result file" << std::endl;
        }

        m_insertion_position = 0;
        m_moved_bytes = 0;
        m_contention_value = 0;

        m_time = -1.0;

        logError("Value store created");
    }

    ~PAPIValueStore()
    {
    
        flush();
        fclose(m_store_file);
    }

    // Does a couple of empty runs (starting and immediately stopping counters) to get a rough estimate of overheads
    void getMeasuredOverhead()
    {
        auto _min = PAPIValueSetInternal<false, events...>(-1, 0, 0, ValueSetType::OverheadComp);
        auto _max = PAPIValueSetInternal<false, events...>();

        // flush values
        for(auto& x : _min.store()) x = std::numeric_limits<long long>::max();
        for(auto& x : _max.store()) x = std::numeric_limits<long long>::min();

        auto set_min_max = [&](const auto& vs) {
            // TODO(later): Vectorize?
            for(size_t i = 0; i < sizeof(vs.cstore()) / sizeof(vs.cstore()[0]); ++i)
            {
                _min.store()[i] = std::min(_min.store()[i], vs.cstore()[i]);
                _max.store()[i] = std::max(_max.store()[i], vs.cstore()[i]);
            }
        };

        for(auto i = 0; i < OVERHEAD_REPETITIONS; ++i) {
            PAPIPerf<events...> perfctr;
            PAPIValueSetInternal<false, events...> vs;
            perfctr.enterCritical();
            perfctr.leaveCritical(vs);

            set_min_max(vs);   
        }

        addEntry(_min);
        
    }

    void set_time(double t)
    {
        this->m_time = t;
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
#ifdef LOG_ERRORS
        FILE* f = fopen("errors.log", "a");
        if(f) {
            fprintf(f, "Flushing with pos %ld\n", size_t(this->m_insertion_position));
            fclose(f);
        }
#endif
        if(!m_store_file)
            return;
        // We want to store information about what we actually measured
        int event_override[sizeof...(events)];
        bool override_events = false;
        std::string resultstring;
        constexpr size_t result_reserve_size = 1024ul * 1024ul * 16;
        resultstring.reserve(result_reserve_size);
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
                    resultstring.append(e.toStoreFormat(event_override));
                    override_events = false;
                }
                else
                {
                    resultstring.append(e.toStoreFormat());
                }
                if(resultstring.length() > result_reserve_size / 2)
                {
                    fwrite(resultstring.data(), resultstring.length(), 1, m_store_file);
                    resultstring.clear();
                    resultstring.reserve(result_reserve_size);
                }
            }
        }
        // This may write an empty string, but that does not matter
        fwrite(resultstring.data(), resultstring.length(), 1, m_store_file);
        if(m_insertion_position > 0)
        {
#ifndef NO_RUNTIME_BYTEMOVEMENT_ACCUMULATION
            // Quite expensive using std::to_string, but it adapts to different types...
            auto bm = collectBytesMoved();
            fprintf(m_store_file, "# moved_bytes: %s\n", std::to_string(bm).c_str());
#endif
            // Also store contention
            uint64_t cont = 0;
            cont = m_contention_value.exchange(cont);
            
            if(cont != 0)
                fprintf(m_store_file, "# contention: %s\n", std::to_string(cont).c_str());
        }
        // Write collected time as well.
        if(m_time != -1.0)
        {
            fprintf(m_store_file, "# Timer recorded %f secs\n", m_time);
            m_time = -1.0;
        }
        m_values.clear();
        //m_values.reserve(store_reserve_size);
        m_values.resize(store_reserve_size);
        m_insertion_position = 0;
    }

    // This is to provide a default sync point. Its effect on the output can be disregarded
    void markSuperSectionStart(uint32_t nodeid, ValueSetType flags = ValueSetType::marker_supersection_start)
    {
        if(this->m_insertion_position >= static_cast<size_t>(store_reserve_size * DACE_INSTRUMENTATION_SUPERSECTION_FLUSH_THRESHOLD))
            flush();
        
        PAPIValueSetInternal<false, events...> set(nodeid, 0, 0, flags);
        addEntry(set);
    }

    // This marks sections in a threadsafe way. In principle, instead of being a "barrier" syncing threads, it now just guarantees serial properties per thread
    void markSectionStart(uint32_t nodeid, long long SizeInBytes, long long InputSize, uint32_t threadid, uint32_t iteration = 0, ValueSetType flags = ValueSetType::marker_section_start)
    {
        // Difference SizeInBytes and InputSize: InputSize is just the amount of bytes moved INTO the section, while sizeInBytes is the amount of bytes MOVED inside a section (without reuses of the same data)
        PAPIValueSetInternal<false, events...> set(nodeid, threadid, iteration, flags);
        set.store()[0] = SizeInBytes; // Use the first slot for memory information
        set.store()[1] = InputSize; // Use the second slot for memory input information
        static_assert(sizeof...(events) >= 2, "Must have at least 2 counters specified");
        addEntry(set);
    }

    // # TODO: Maybe with forward args?
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
                logError("Flushing in measurement section");
                std::cerr << "Array not big enough!" << std::endl;

                // To keep it running, we just have to flush. For this, we should lock

                // Ideally, we only let the main thread flush (to avoid excess memory movements, especially with NUMA)
                // However, it's hard to guarantee termination like this. (What if the main thread is done and will never call getNewSlotPosition())
                if(m_vec_mutex.try_lock()) {
                    // We got the lock, so we can flush
                    logError("Flushing...");
                    flush();
                    m_vec_mutex.unlock();
                    logError("Continuing...");
                }
                else {
                    // We didn't get the lock, which means that somebody else is already flushing.
                    // Since we lost a lot of time already, we can just use the lock to wait instead of spinlocking.
                    logError("Waiting for values to be written");
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

    PAPIValueSetInternal<false, events...>& addEntry(const PAPIValueSetInternal<false, events...>& val)
    {
        auto pos = getNewSlotPosition();

        m_values[pos] = val;

        return m_values[pos];
    }

    template<int... counterevents>
    PAPIValueSetInternal<false, events...>& getNewValueSet([[maybe_unused]] const PAPIPerfLowLevel<counterevents...>& perf, uint32_t nodeid, uint32_t coreid, uint32_t iteration, ValueSetType type = ValueSetType::Default)
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
        m_values[pos] = PAPIValueSetInternal<false, events...>(nodeid, coreid, iteration, ValueSetType::CounterOverride);

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

    inline PAPIValueSetInternal<false, events...>& getNewValueSet([[maybe_unused]] const PAPIPerfLowLevel<events...>& perf, uint32_t nodeid, uint32_t coreid, uint32_t iteration, ValueSetType type = ValueSetType::Default)
    {
        return __impl_getNewValueSet(nodeid, coreid, iteration, type);
    }

    PAPIValueSetInternal<false, events...>& __impl_getNewValueSet(uint32_t nodeid, uint32_t coreid, uint32_t iteration, ValueSetType type = ValueSetType::Default)
    {
        auto& retval = addEntry(PAPIValueSetInternal<false, events...>(nodeid, coreid, iteration, type));
#ifdef TEST_ALIGNMENT
        uintptr_t val = (uintptr_t)&retval;
        auto lower_bits = val & (CACHE_LINE_SIZE - 1);
        if(lower_bits != 0)
        {
            std::cerr << "ERROR: LOWER BITS EXPECTED 0, got " << lower_bits << ". This means the values are not aligned!" << std::endl;
        }
        assert(lower_bits == 0);
#endif
        return retval;
    }
    
    PAPIValueSetInternal<false, events...>& getNewValueSet()
    {
        return getNewValueSet(PAPIPerf<events...>(), 0, 0, 0);
    }
private:
    std::recursive_mutex m_vec_mutex;
    //aligned_vector<PAPIValueSetInternal<false, events...>> m_values;
    //std::vector<PAPIValueSetInternal<false, events...>> m_values;
    AlignedContainer<PAPIValueSetInternal<false, events...>> m_values;
    
    std::atomic<size_t> m_insertion_position;
    std::atomic<uint64_t> m_contention_value;
    std::atomic<byte_counter_size_t> m_moved_bytes;

    double m_time;

    FILE* m_store_file;
};


// Class similar to PAPIPerf, but allows for much finer grained control
template<int... events>
class PAPIPerfLowLevel
{
public:

    PAPIPerfLowLevel(const bool multiplexing = false)
        : m_event_set(PAPI_NULL)
    {
        // Need to synchronize accesses because papi is not threadsafe...
        //std::lock_guard<std::recursive_mutex> guard(papi_mutex);
        auto r_create_eventset = PAPI_create_eventset(&m_event_set);
        #ifndef SKIP_RETVAL_CHECKS
        if(r_create_eventset != PAPI_OK)
        {
            std::cerr << "Failed to create event set, code " << r_create_eventset << std::endl;
            #ifdef LOG_ERRORS
            FILE* f = fopen("errors.log", "a");
            if(f) {
                fprintf(f, "Failed to create event set with code %d\n", r_create_eventset);
                fclose(f);
            }
            #endif
        }
        #endif

        #ifdef ASSIGN_COMPONENT
        // We need this because multiplexing will otherwise act up.
        // Issue is that if we don't do it in the general case as well, the starting will take longer. It's not really a good solution to put it here, though.
        PAPI_assign_eventset_component(m_event_set, 0);
        #endif
        if(multiplexing)
        {
            #ifndef ASSIGN_COMPONENT
            // We need this because multiplexing will otherwise act up.
            PAPI_assign_eventset_component(m_event_set, 0);
            #endif
            this->enable_multiplexing();
        }
        int evarr[] = {events...};
        auto r_add_events = PAPI_add_events(m_event_set, evarr, sizeof...(events));
        #ifndef SKIP_RETVAL_CHECKS
        if(r_add_events != PAPI_OK)
        {
            PAPI_cleanup_eventset(m_event_set);
            std::cerr << "Failed to add events to event set, code " << r_add_events << std::endl;
            #ifdef LOG_ERRORS
            FILE* f = fopen("errors.log", "a");
            if(f) {
                fprintf(f, "Failed to add events to event set with code %d\n", r_add_events);
                fclose(f);
            }
            #endif
        }
        #endif
#ifdef PAPI_EXPLICIT_THREADS
        const auto r_reg_thread = PAPI_register_thread();
        #ifndef SKIP_RETVAL_CHECKS
        if(r_reg_thread != PAPI_OK)
        {
            std::cerr << "Failed to register thread, code " << r_reg_thread << std::endl;
        }
        #endif
#endif
    }

    

    ~PAPIPerfLowLevel()
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
        //constexpr auto num_events = sizeof...(events);
        DACE_PERF_mfence; // Fence before starting to keep misses outside
        const auto r_start = PAPI_start(m_event_set);
        // #TODO: Check if we can just omit checking on the return value...
        #ifndef SKIP_RETVAL_CHECKS
        if(r_start != PAPI_OK)
        {
            std::cerr << "Failed to start counters with code " << r_start << std::endl;
            #ifdef LOG_ERRORS
            FILE* f = fopen("errors.log", "a");
            if(f) {
                fprintf(f, "Failed to start counters with code %d\n", r_start);
                fclose(f);
            }
            #endif
        }
        #endif
    }

    template<bool standalone, int... e>
    void leaveCritical(PAPIValueSetInternal<standalone, e...>& values)
    {
        constexpr auto num_events = sizeof...(events);
        
        // Make sure we have the correct sizes
        static_assert(sizeof...(e) >= num_events); 

        // Fence before stopping to keep misses inside
        DACE_PERF_mfence;
        const auto r_stop = PAPI_stop(m_event_set, values.store());
        #ifndef SKIP_RETVAL_CHECKS
        // #TODO: Check if we can just omit checking on the return value...
        if(r_stop != PAPI_OK)
        {
            std::cerr << "Failed to stop counters with code " << r_stop << std::endl;
            #ifdef LOG_ERRORS
            FILE* f = fopen("errors.log", "a");
            if(f) {
                fprintf(f, "Failed to stop counters with code %d\n", r_stop);
                fclose(f);
            }
            #endif
        }
        #endif
        // Since we allow storing less, we have to make sure there's no stale data in the value store. Set to an invalid value, i.e. max.
        for(auto i = num_events; i < sizeof...(e); ++i)
        {
            values.store()[i] = std::numeric_limits<long long>::max();
        }
    }
    
    std::string listEvents() const 
    {
        std::string ret = "===== Events =====\n";
        PAPI_event_info_t info;
        int evlist[128];
        int evsize = sizeof(evlist) / sizeof(*evlist);
        const auto r_list_events = PAPI_list_events(m_event_set, evlist, &evsize);
        if(r_list_events != PAPI_OK)
        {
            std::cerr << "Failed in list events, code " << r_list_events << std::endl;
            return ret;
        }
        for(size_t i = 0; i < evsize; ++i) 
        {
            const auto& e = evlist[i];
            const auto r = PAPI_get_event_info(e, &info);
            if(r != PAPI_OK)
                ret += "<ERROR determining event info>";
            else
                ret += info.symbol;
            ret +=  "\n";
        }

        return ret;
    }    
private:
    int m_event_set;
};


// Convenience typedef
typedef PAPIPerfLowLevel<PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM> PAPIPerfAllMisses;
typedef PAPIPerfLowLevel<PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L3_TCM> PAPIPerfDataMisses;

typedef PAPIValueStore<PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM> PAPIPerfStoreAllMisses;



inline double PAPI::getTimePerCycle()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;

    PAPIPerfLowLevel<PAPI_TOT_CYC, PAPI_REF_CYC> perf;
    auto vs = perf.ValueSet();

    perf.enterCritical();
    auto start = high_resolution_clock::now();
    for(size_t i = 0; i < 10000000l; ++i)
        __asm__ __volatile__ ("");
    auto stop = high_resolution_clock::now();
    perf.leaveCritical(vs);

    return double(duration_cast<std::chrono::microseconds>(std::chrono::duration<double>(stop - start)).count()) / double(vs.store()[0]) / 1e6;
}

// Small wrapper for an std timer
class Timer {
public:
    Timer()
        : m_start(std::chrono::high_resolution_clock::now()), m_finished(false)
    {
        
    }

    double collect()
    {
        const auto n = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> timediff = n - m_start;
        m_finished = true;
        return timediff.count();
    }

    ~Timer()
    {
        if(m_finished)
            return;


        auto timediff = collect();

        FILE* f = fopen(store_path, "ab");
        if(f == nullptr)
        {
            std::cerr << "Timer failed to open result file" << std::endl;
        }
        else
        {
            fprintf(f, "# Timer recorded %f secs\n", timediff);

            fclose(f);
        }
    }
private:
    std::chrono::high_resolution_clock::time_point m_start;
    bool m_finished;
};

};