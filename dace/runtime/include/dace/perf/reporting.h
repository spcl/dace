// Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_PERF_REPORTING_H
#define __DACE_PERF_REPORTING_H

#include <chrono>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

#ifdef __unix__
#include <unistd.h>
#endif

#define DACE_REPORT_BUFFER_SIZE 2048

namespace dace {
namespace perf {

    struct TraceEvent {
        std::string name;
        std::string cat;
        unsigned long int tstart;
        unsigned long int tend;
        std::thread::id tid;
        struct _element_id {
            int sdfg_id;
            int state_id;
            int el_id;
        } element_id;
    };

    /**
     * Simple instrumentation report class that can save to JSON.
     */
    class Report {
    protected:
        std::mutex _mutex;
        std::vector<TraceEvent> _events;
    public:
        ~Report() {}

        /**
         * Clears the report.
         */
        void reset() {
            std::lock_guard<std::mutex> guard (this->_mutex);
            this->_events.clear();
            this->_events.reserve(DACE_REPORT_BUFFER_SIZE);
        }

        /**
         * Appends a single completion event to the report.
         * @param name:     Name of the event.
         * @param cat:      Comma separated categories the event belongs to.
         * @param tstart:   Start timestamp of the event.
         * @param tend:     End timestamp of the event.
         * @param sdfg_id:  SDFG ID of the element associated with this event.
         * @param state_id: State ID of the element associated with this event.
         * @param el_id:    ID of the element associated with this event.
         */
        void add_completion(
            const char *name,
            const char *cat,
            unsigned long int tstart,
            unsigned long int tend,
            int sdfg_id,
            int state_id,
            int el_id
        ) {
            std::thread::id tid = std::this_thread::get_id();
            std::lock_guard<std::mutex> guard (this->_mutex);
            this->_events.push_back({
                name,
                cat,
                tstart,
                tend,
                tid,
                {
                    sdfg_id,
                    state_id,
                    el_id
                }
            });
        }

        /**
         * Saves the report to a timestamped JSON file.
         * @param path: Path to folder where the output JSON file will be stored.
         */
        void save(const char *path) {
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
                    ofs << "\"ph\": \"" << 'X' << "\", ";

                    ofs << "\"ts\": " << event.tstart << ", ";
                    ofs << "\"dur\": " << event.tend - event.tstart << ", ";

                    ofs << "\"pid\": " << pid << ", ";
                    ofs << "\"tid\": " << event.tid << ", ";

                    ofs << "\"args\": {";
                    ofs << "\"sdfg_id\": " << event.element_id.sdfg_id;
                    if (event.element_id.state_id > -1)
                        ofs << ", \"state_id\": " << event.element_id.state_id;
                    if (event.element_id.el_id > -1)
                        ofs << ", \"id\": " << event.element_id.el_id;
                    ofs << "}";

                    ofs << "}";
                }

                ofs << std::endl << "  ]" << std::endl;
                ofs << "}" << std::endl;
            }
        }
    };

    extern Report report;

}  // namespace perf
}  // namespace dace

#undef DACE_REPORT_BUFFER_SIZE

#endif  // __DACE_PERF_REPORTING_H
