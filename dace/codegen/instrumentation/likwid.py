# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the LIKWID counter performance instrumentation provider.
    Used for collecting CPU performance counters. """

import dace
from dace import dtypes, registry
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.config import Config

from dace.transformation import helpers as xfh

from pathlib import Path


@registry.autoregister_params(type=dtypes.InstrumentationType.LIKWID_Counters)
class LIKWIDInstrumentation(InstrumentationProvider):
    """ Instrumentation provider that reports CPU performance counters using
        the Likwid tool.
    """

    perf_whitelist_schedules = [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Sequential]

    def __init__(self):
        self._likwid_used = False
        self._regions = []

    def configure_likwid(self):
        Config.append('compiler', 'cpu', 'args', value=' -DLIKWID_PERFMON -fopenmp ')

        # Link with liblikwid
        Config.append('compiler', 'cpu', 'libs', value=' likwid ')

        try:
            self._default_events = Config.get('instrumentation', 'likwid', 'default_events')
        except KeyError:
            self._default_events = "CLOCK"

        self._likwid_used = True

    def on_sdfg_begin(self, sdfg, local_stream, global_stream, codegen):
        if not sdfg.parent is None:
            return

        # Configure CMake project and counters
        self.configure_likwid()

        if not self._likwid_used:
            return

        likwid_marker_file = Path(sdfg.build_folder) / "perf" / "likwid_marker.out"

        # Add instrumentation includes and initialize LIKWID
        header_code = '''
#include <omp.h>
#include <likwid.h>
#include <likwid-marker.h>

#include <unistd.h>
#include <string>

#define MAX_NUM_EVENTS 20
'''
        global_stream.write(header_code, sdfg)

        init_code = f'''
if(getenv("LIKWID_PIN"))
{{
    printf("ERROR: Instrumentation must not be wrapped by likwid-perfctr.\\n");
    exit(1);
}}

setenv("LIKWID_FILEPATH", "{likwid_marker_file.absolute()}", 0);
setenv("LIKWID_MODE", "1", 0);
setenv("LIKWID_FORCE", "1", 1);
setenv("LIKWID_EVENTS", "{self._default_events}", 0);

int num_threads = 1;
int num_procs = 1;
#pragma omp parallel
{{
    #pragma omp single
    {{
        num_threads = omp_get_num_threads();
        num_procs = omp_get_num_procs();
    }}
}}

if (num_threads > num_procs)
{{
    printf("ERROR: Number of threads larger than number of processors.\\n");
    exit(1);
}}

std::string thread_pinning = "0";
for (int i = 1; i < num_threads; i++)
{{
    thread_pinning += "," + std::to_string(i);
}}
const char* thread_pinning_c = thread_pinning.c_str();
setenv("LIKWID_THREADS", thread_pinning_c, 1);

LIKWID_MARKER_INIT;

#pragma omp parallel
{{
    int thread_id = omp_get_thread_num();
    likwid_pinThread(thread_id);
    LIKWID_MARKER_THREADINIT;
}}
'''
        local_stream.write(init_code)

    def on_sdfg_end(self, sdfg, local_stream, global_stream):
        if not self._likwid_used:
            return

        outer_code = f'''
double events[num_threads][MAX_NUM_EVENTS];
double time[num_threads];
'''
        local_stream.write(outer_code, sdfg)

        for region, sdfg_id, state_id, node_id in self._regions:
            report_code = f'''
#pragma omp parallel
{{
    num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    int nevents = MAX_NUM_EVENTS;
    int count = 0;

    LIKWID_MARKER_GET("{region}", &nevents, events[thread_id], time + thread_id, &count);

    #pragma omp barrier

    #pragma omp single
    {{
        int gid = perfmon_getIdOfActiveGroup();
        char* group_name = perfmon_getGroupName(gid);

        for (int t = 0; t < num_threads; t++)
        {{
            __state->report.add_completion("Timer", "likwid", 0, time[t] * 1000 * 1000, t, {sdfg_id}, {state_id}, {node_id});
        }}

        for (int i = 0; i < nevents; i++)
        {{
            char* event_name = perfmon_getEventName(gid, i); 
            
            for (int t = 0; t < num_threads; t++)
            {{
                __state->report.add_counter("{region}", "likwid", event_name, events[t][i], t, {sdfg_id}, {state_id}, {node_id});
            }}
        }}
    }}
}}
'''
            local_stream.write(report_code)

        local_stream.write("LIKWID_MARKER_CLOSE;", sdfg)

    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        if not self._likwid_used:
            return

        if state.instrument == dace.InstrumentationType.LIKWID_Counters:
            sdfg_id = sdfg.sdfg_id
            state_id = sdfg.node_id(state)
            node_id = -1
            region = f"state_{sdfg_id}_{state_id}_{node_id}"
            self._regions.append((region, sdfg_id, state_id, node_id))

            marker_code = f'''
#pragma omp parallel
{{
    LIKWID_MARKER_REGISTER("{region}");

    #pragma omp barrier
    LIKWID_MARKER_START("{region}");
}}
'''
            local_stream.write(marker_code)

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if not self._likwid_used:
            return

        if state.instrument == dace.InstrumentationType.LIKWID_Counters:
            sdfg_id = sdfg.sdfg_id
            state_id = sdfg.node_id(state)
            node_id = -1
            region = f"state_{sdfg_id}_{state_id}_{node_id}"

            marker_code = f'''
#pragma omp parallel
{{
    LIKWID_MARKER_STOP("{region}");
}}
'''
            local_stream.write(marker_code)

    def on_scope_entry(self, sdfg, state, node, outer_stream, inner_stream, global_stream):
        if not self._likwid_used or node.instrument != dace.InstrumentationType.LIKWID_Counters:
            return

        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(state, node) is not None:
            raise TypeError("Only top-level map scopes supported")
        elif node.schedule not in LIKWIDInstrumentation.perf_whitelist_schedules:
            raise TypeError("Unsupported schedule on scope")

        sdfg_id = sdfg.sdfg_id
        state_id = sdfg.node_id(state)
        node_id = state.node_id(node)
        region = f"scope_{sdfg_id}_{state_id}_{node_id}"

        self._regions.append((region, sdfg_id, state_id, node_id))
        marker_code = f'''
#pragma omp parallel
{{
    LIKWID_MARKER_REGISTER("{region}");

    #pragma omp barrier
    LIKWID_MARKER_START("{region}");
}}
'''
        outer_stream.write(marker_code)

    def on_scope_exit(self, sdfg, state, node, outer_stream, inner_stream, global_stream):
        entry_node = state.entry_node(node)
        if not self._likwid_used or entry_node.instrument != dace.InstrumentationType.LIKWID_Counters:
            return

        sdfg_id = sdfg.sdfg_id
        state_id = sdfg.node_id(state)
        node_id = state.node_id(entry_node)
        region = f"scope_{sdfg_id}_{state_id}_{node_id}"

        marker_code = f'''
#pragma omp parallel
{{
    LIKWID_MARKER_STOP("{region}");
}}
'''
        outer_stream.write(marker_code)
