// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_PERF_REPORTING_H
#define __DACE_PERF_REPORTING_H

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#define DACE_REPORT_BUFFER_SIZE     2048
#define DACE_REPORT_EVENT_NAME_LEN  64
#define DACE_REPORT_EVENT_CAT_LEN   10

namespace dace {
namespace perf {

    struct TraceEvent {
        char ph;
        char name[DACE_REPORT_EVENT_NAME_LEN];
        char cat[DACE_REPORT_EVENT_CAT_LEN];
        unsigned long int tstart;
        unsigned long int tend;
        size_t tid;
        struct _element_id {
            int cfg_id;
            int state_id;
            int el_id;
        } element_id;
        struct _counter {
            char name[DACE_REPORT_EVENT_NAME_LEN];
            unsigned long int val;
        } counter;
    };

    /**
     * An event of an element that a GPU thread block ran, with times in nanoseconds.
     */
    struct DeviceTraceEvent {
        char name[DACE_REPORT_EVENT_NAME_LEN];
        unsigned long long tstart;
        unsigned long long duration;
        size_t block;  // The thread block that ran the element (the event's thread in the trace)
        unsigned int sm;  // The multiprocessor that ran it
        unsigned long long cycles;  // Of the multiprocessor's clock
        TraceEvent::_element_id element_id;
    };

    /**
     * Simple instrumentation report class that can save to JSON.
     */
    class Report {
    protected:
        std::mutex _mutex;
        std::vector<TraceEvent> _events;
        std::vector<DeviceTraceEvent> _device_events;
    public:
        ~Report() {}

        /**
         * Clears the report.
         */
        void reset() {
            std::lock_guard<std::mutex> guard (this->_mutex);
            this->_events.clear();
            this->_events.reserve(DACE_REPORT_BUFFER_SIZE);
            this->_device_events.clear();
        }

        /**
         * Appends an event of an element that a GPU thread block ran.
         * @param name:     Name of the element.
         * @param tstart:   Start time, in nanoseconds on the time axis of the other events.
         * @param duration: Duration in nanoseconds.
         * @param block:    Linear index of the thread block.
         * @param sm:       Multiprocessor that ran the thread block.
         * @param cycles:   Clock cycles of the multiprocessor from start to end.
         * @param cfg_id:   Control flow graph ID of the element.
         * @param state_id: State ID of the element.
         * @param el_id:    ID of the element.
         */
        void add_device_completion(
            const char *name,
            unsigned long long tstart,
            unsigned long long duration,
            size_t block,
            unsigned int sm,
            unsigned long long cycles,
            int cfg_id,
            int state_id,
            int el_id
        ) {
            std::lock_guard<std::mutex> guard (this->_mutex);
            DeviceTraceEvent event = {"", tstart, duration, block, sm, cycles, { cfg_id, state_id, el_id }};
            strncpy(event.name, name, DACE_REPORT_EVENT_NAME_LEN);
            event.name[DACE_REPORT_EVENT_NAME_LEN - 1] = '\0';
            this->_device_events.push_back(event);
        }

        void add_counter(
            const char *name,
            const char *cat,
            const char *counter_name,
            unsigned long int counter_val
        ) {
            std::thread::id thread_id = std::this_thread::get_id();
            size_t tid = std::hash<std::thread::id>{}(thread_id);
            add_counter(name, cat, counter_name, counter_val, tid, -1, -1, -1);
        }

        void add_counter(
            const char *name,
            const char *cat,
            const char *counter_name,
            unsigned long int counter_val,
            size_t tid,
            int cfg_id,
            int state_id,
            int el_id
        ) {
            long unsigned int tstart = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();
            std::lock_guard<std::mutex> guard (this->_mutex);
            struct TraceEvent event = {
                'C',
                "",
                "",
                tstart,
                0,
                tid,
                { cfg_id, state_id, el_id },
                { "", counter_val }
            };
            strncpy(event.name, name, DACE_REPORT_EVENT_NAME_LEN);
            event.name[DACE_REPORT_EVENT_NAME_LEN - 1] = '\0';
            strncpy(event.cat, cat, DACE_REPORT_EVENT_CAT_LEN);
            event.cat[DACE_REPORT_EVENT_CAT_LEN - 1] = '\0';
            strncpy(event.counter.name, counter_name, DACE_REPORT_EVENT_NAME_LEN);
            event.counter.name[DACE_REPORT_EVENT_NAME_LEN - 1] = '\0';
            this->_events.push_back(event);
        }

        /**
         * Appends a single completion event to the report.
         * @param name:     Name of the event.
         * @param cat:      Comma separated categories the event belongs to.
         * @param tstart:   Start timestamp of the event.
         * @param tend:     End timestamp of the event.
         * @param cfg_id:   Control flow graph ID of the element associated with this event.
         * @param state_id: State ID of the element associated with this event.
         * @param el_id:    ID of the element associated with this event.
         */
        void add_completion(
            const char *name,
            const char *cat,
            unsigned long int tstart,
            unsigned long int tend,
            int cfg_id,
            int state_id,
            int el_id
        ) {
            std::thread::id thread_id = std::this_thread::get_id();
            size_t tid = std::hash<std::thread::id>{}(thread_id);
            add_completion(name, cat, tstart, tend, tid, cfg_id, state_id, el_id);
        }

        void add_completion(
            const char *name,
            const char *cat,
            unsigned long int tstart,
            unsigned long int tend,
            size_t tid,
            int cfg_id,
            int state_id,
            int el_id
        ) {
            std::lock_guard<std::mutex> guard (this->_mutex);
            struct TraceEvent event = {
                'X',
                "",
                "",
                tstart,
                tend,
                tid,
                { cfg_id, state_id, el_id },
                { "", 0 }
            };
            strncpy(event.name, name, DACE_REPORT_EVENT_NAME_LEN);
            event.name[DACE_REPORT_EVENT_NAME_LEN - 1] = '\0';
            strncpy(event.cat, cat, DACE_REPORT_EVENT_CAT_LEN);
            event.cat[DACE_REPORT_EVENT_CAT_LEN - 1] = '\0';
            this->_events.push_back(event);
        }

        /**
         * Saves the report to a timestamped JSON file.
         * @param path: Path to folder where the output JSON file will be stored.
         * @param hash: Hash of the SDFG.
         */
        void save(const char *path, const char *hash) {
            std::lock_guard<std::mutex> guard (this->_mutex);

            // Create report filename
            std::stringstream ss;
            std::chrono::milliseconds ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                );
            ss << path << "/" << "report-" << ms.count() << ".json";

            // Dump report as JSON
            {
                bool first = true;
                std::ofstream ofs (ss.str(), std::ios::binary);

                ofs << "{" << std::endl;
                ofs << "  \"traceEvents\": [" << std::endl;

                int pid = getpid();

                for (const auto& event : this->_events) {
                    if (first)
                        first = false;
                    else
                        ofs << "," << std::endl;

                    ofs << "    {";
                    ofs << "\"name\": \"" << event.name << "\", ";
                    ofs << "\"cat\": \"" << event.cat << "\", ";
                    ofs << "\"ph\": \"" << event.ph << "\", ";

                    ofs << "\"ts\": " << event.tstart << ", ";

                    if (event.ph == 'X')
                        ofs << "\"dur\": " << event.tend - event.tstart << ", ";

                    ofs << "\"pid\": " << pid << ", ";
                    ofs << "\"tid\": " << event.tid << ", ";

                    ofs << "\"args\": {";
                    ofs << "\"cfg_id\": " << event.element_id.cfg_id;

                    if (event.element_id.state_id > -1) {
                        ofs << ", \"state_id\": ";
                        ofs << event.element_id.state_id;
                    }

                    if (event.element_id.el_id > -1) {
                        ofs << ", \"id\": " << event.element_id.el_id;
                    }
                     if (event.ph == 'C') {
                        ofs << ", \"" << event.counter.name << "\": ";
                        ofs << event.counter.val;
                    }

                    ofs << "}}";
                }

                // Device events: a thread of the trace per thread block
                for (const auto& event : this->_device_events) {
                    if (first)
                        first = false;
                    else
                        ofs << "," << std::endl;

                    ofs << "    {";
                    ofs << "\"name\": \"" << event.name << "\", ";
                    ofs << "\"cat\": \"GPU\", ";
                    ofs << "\"ph\": \"X\", ";
                    // Microseconds, to the nanosecond
                    ofs << "\"ts\": " << event.tstart / 1000 << "." << std::setfill('0') << std::setw(3)
                        << event.tstart % 1000 << ", ";
                    ofs << "\"dur\": " << event.duration / 1000 << "." << std::setw(3) << event.duration % 1000
                        << std::setfill(' ') << ", ";
                    ofs << "\"pid\": " << pid << ", ";
                    ofs << "\"tid\": " << event.block << ", ";
                    ofs << "\"args\": {";
                    ofs << "\"cfg_id\": " << event.element_id.cfg_id;
                    ofs << ", \"state_id\": " << event.element_id.state_id;
                    ofs << ", \"id\": " << event.element_id.el_id;
                    ofs << ", \"block\": " << event.block;
                    ofs << ", \"sm\": " << event.sm;
                    ofs << ", \"cycles\": " << event.cycles;
                    ofs << "}}";
                }

                ofs << std::endl << "  ]," << std::endl;

                ofs << "  \"sdfgHash\": \"";
                ofs << hash;
                ofs << "\"" << std::endl;

                ofs << "}" << std::endl;
            }
        }
    };

    extern Report report;

}  // namespace perf
}  // namespace dace

#undef DACE_REPORT_BUFFER_SIZE
#undef DACE_REPORT_EVENT_NAME_LEN
#undef DACE_REPORT_EVENT_CAT_LEN

#endif  // __DACE_PERF_REPORTING_H
