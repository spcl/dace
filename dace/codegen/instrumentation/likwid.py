# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the LIKWID counter performance instrumentation provider.
    Used for collecting CPU performance counters. """

import dace
from dace import dtypes, registry
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.config import Config

from typing import Optional, Set


@registry.autoregister_params(type=dtypes.InstrumentationType.LIKWID_Counters)
class LIKWIDInstrumentation(InstrumentationProvider):
    """ Instrumentation provider that reports CPU performance counters using
        the Likwid tool. """

    _counters: Optional[Set[str]] = None

    perf_whitelist_schedules = [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Sequential]

    def __init__(self):
        self._likwid_used = False

    def configure_likwid(self):
        Config.append('compiler', 'cpu', 'args', value=' -DLIKWID_PERFMON -fopenmp ')

        # Link with liblikwid
        Config.append('compiler', 'cpu', 'libs', value=' likwid ')

        self._likwid_used = True

    def on_sdfg_begin(self, sdfg, local_stream, global_stream, codegen):
        if not sdfg.parent is None:
            return

        # Configure CMake project and counters
        self.configure_likwid()

        if not self._likwid_used:
            return

        # Add instrumentation includes and initialize PAPI
        header_code = '''
#include <omp.h>
#include <likwid.h>
#include <likwid-marker.h>
#include <unistd.h>

#define MAX_NUM_EVENTS 20
'''
        global_stream.write(header_code, sdfg)

        init_code = '''
if(getenv("LIKWID_PIN"))
{
    printf("ERROR: it appears you are running this SDFG with likwid-perfctr. This instrumentation is supposed to be used standalone.");
    exit(1);
}

setenv("LIKWID_FILEPATH", "/tmp/likwid_marker.out", 0);
setenv("LIKWID_MODE", "1", 0);
setenv("LIKWID_FORCE", "1", 1);

int num_threads = 1;
#pragma omp parallel
{
    num_threads = omp_get_num_threads();
}

setenv("LIKWID_THREADS", "0,1", 1);

LIKWID_MARKER_INIT;

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    likwid_pinThread(thread_id);
    LIKWID_MARKER_THREADINIT;
}
'''
        local_stream.write(init_code)

    def on_sdfg_end(self, sdfg, local_stream, global_stream):
        if not self._likwid_used:
            return

        report_code = '''
double events[num_threads][MAX_NUM_EVENTS];
double time[num_threads];

#pragma omp parallel
{
    num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    int nevents = MAX_NUM_EVENTS;
    int count = 0;

    LIKWID_MARKER_GET("state_0", &nevents, events[thread_id], time + thread_id, &count);

    #pragma omp barrier

    #pragma omp single
    {
        int gid = perfmon_getIdOfActiveGroup();
        char* group_name = perfmon_getGroupName(gid);

        for (int t = 0; t < num_threads; t++)
        {
            __state->report.add_completion("Time", "likwid", 0, time[t] * 1000 * 1000, 0, 0, -1);
        }

        for (int i = 0; i < nevents; i++)
        {
            char* event_name = perfmon_getEventName(gid, i); 
            
            for (int t = 0; t < num_threads; t++)
            {
                __state->report.add_completion(event_name, "likwid", 0, events[t][i] * 1000, 0, 0, -1);
            }
        }

        /*
        // TODO: Figure out region id by tag
        int region_id = 0;
        for (int k = 0; k < perfmon_getNumberOfMetrics(gid); k++) 
        {
            char* metric_name = perfmon_getMetricName(gid, k);

            for (int t = 0; t < num_threads; t++)
            {
                double metric_value = perfmon_getMetricOfRegionThread(region_id, k, t);
                __state->report.add_completion(metric_name, "likwid", 0, metric_value, 0, 0, -1);
            }
        }
        */
    }
}
'''
        local_stream.write(report_code)
        local_stream.write("LIKWID_MARKER_CLOSE;", sdfg)

    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        if not self._likwid_used:
            return

        if state.instrument == dace.InstrumentationType.LIKWID_Counters:
            marker_code = '''
#pragma omp parallel
{
    LIKWID_MARKER_REGISTER("state_0");

    #pragma omp barrier
    LIKWID_MARKER_START("state_0");
}
'''
            local_stream.write(marker_code)

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if not self._likwid_used:
            return

        if state.instrument == dace.InstrumentationType.LIKWID_Counters:
            marker_code = '''
#pragma omp parallel
{
    LIKWID_MARKER_STOP("state_0");
}
'''
            local_stream.write(marker_code)

