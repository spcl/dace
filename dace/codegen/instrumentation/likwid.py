# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Implements the LIKWID counter performance instrumentation provider.
    Used for collecting CPU performance counters.
"""

import dace
import os
import ctypes.util

from pathlib import Path

from dace import dtypes, registry, library
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.config import Config
from dace.transformation import helpers as xfh


@library.environment
class LIKWID:
    """ 
    An environment for LIKWID
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []

    headers = ["likwid.h"]
    state_fields = []
    init_code = ""
    finalize_code = ""
    dependencies = []

    @staticmethod
    def cmake_includes():
        # Anaconda
        if 'CONDA_PREFIX' in os.environ:
            base_path = os.environ['CONDA_PREFIX']
            # Anaconda on Windows
            candpath = os.path.join(base_path, 'Library', 'include')
            if os.path.isfile(os.path.join(candpath, 'likwid.h')):
                return [candpath]
            # Anaconda on other platforms
            candpath = os.path.join(base_path, 'include')
            if os.path.isfile(os.path.join(candpath, 'likwid.h')):
                return [candpath]

        return []

    @staticmethod
    def cmake_libraries():
        path = ctypes.util.find_library('likwid')
        if path:
            return [path]

        return []

    @staticmethod
    def is_installed():
        return len(LIKWID.cmake_libraries()) > 0


@registry.autoregister_params(type=dtypes.InstrumentationType.LIKWID_CPU)
class LIKWIDInstrumentationCPU(InstrumentationProvider):
    """ Instrumentation provider that reports CPU performance counters using
        the Likwid tool.
    """

    perf_whitelist_schedules = [dtypes.ScheduleType.CPU_Multicore, dtypes.ScheduleType.Sequential]

    def __init__(self):
        self._likwid_used = False
        self._regions = []

        try:
            self._default_events = Config.get('instrumentation', 'likwid', 'default_events')
        except KeyError:
            self._default_events = "CLOCK"

    def on_sdfg_begin(self, sdfg, local_stream, global_stream, codegen):
        if sdfg.parent is not None:
            return

        self.codegen = codegen

        # Configure CMake project and counters
        self._likwid_used = LIKWID.is_installed()
        if not self._likwid_used:
            return

        codegen.dispatcher.used_environments.add(LIKWID.full_class_path())
        Config.append('compiler', 'cpu', 'args', value=' -DLIKWID_PERFMON -fopenmp ')

        self.codegen = codegen

        likwid_marker_file = Path(sdfg.build_folder) / "perf" / "likwid_marker.out"

        # Add instrumentation includes and initialize LIKWID
        header_code = '''
#include <likwid.h>

#include <iostream>

#include <csignal>
#include <unistd.h>
#include <string>
#include <sys/types.h>

int* cpu_list = nullptr;
int LIKWID_ACTIVE_GID = -1;

void __dace_generate_report();

void signalHandler( int signum ) {
   std::cout << "Interrupt signal (" << signum << ") received.\\n";

   // cleanup and close up stuff here  
   // terminate program  

    __dace_generate_report();
   
   exit(signum);  
}

'''
        global_stream.write(header_code, sdfg)

        init_code = f'''
signal(SIGINT, signalHandler);  

// Initialize topology module
int error = topology_init();
if (error < 0)
{{
    printf("Failed to initialize LIKWID's topology module\\n");
    exit(1);
}}

// Create affinity domains: Enables reading uncore counters
affinity_init();

// Create list of CPU ids.
CpuTopology_t topo = get_cpuTopology();
cpu_list = (int*) malloc(topo->numHWThreads * sizeof(int));
if (!cpu_list)
{{
    exit(1);
}}
for (int i = 0;i < topo->numHWThreads; i++)
{{
    cpu_list[i] = topo->threadPool[i].apicId;
}}

// Initialize likwid's perfmon module
error = perfmon_init(topo->numHWThreads, cpu_list);
if (error < 0)
{{
    printf("Failed to initialize LIKWID's perfmon module\\n");
    topology_finalize();
    exit(1);
}}
free(cpu_list);

// Configure group via env variable LIKWID_EVENTS
LIKWID_ACTIVE_GID = perfmon_addEventSet(NULL);
if (LIKWID_ACTIVE_GID < 0)
{{
    printf("Failed to add group %s to LIKWID's perfmon module\\n", getenv("LIKWID_EVENTS"));
    perfmon_finalize();
    topology_finalize();
    exit(1);
}}

error = perfmon_setupCounters(LIKWID_ACTIVE_GID);
if (error < 0)
{{
    printf("Failed to setup group in LIKWID's perfmon module\\n");
    perfmon_finalize();
    topology_finalize();
    exit(1);
}}

printf("SETUP");
'''
        codegen._initcode.write(init_code)

    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        if not self._likwid_used:
            return

        if state.instrument == dace.InstrumentationType.LIKWID_CPU:
            sdfg_id = sdfg.sdfg_id
            state_id = sdfg.node_id(state)
            node_id = -1
            region = f"state_{sdfg_id}_{state_id}_{node_id}"
            self._regions.append((region, sdfg_id, state_id, node_id))

            start_counters_code = f'''
int error = perfmon_startCounters();
if (error < 0)
{{
    printf("Failed to start counters for group %d for thread %d\\n", LIKWID_ACTIVE_GID, (-1*error)-1);
    perfmon_finalize();
    topology_finalize();
    exit(1);
}}
std::cout << "STARTED\\n";
'''
            local_stream.write(start_counters_code)

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if not self._likwid_used:
            return

        if state.instrument == dace.InstrumentationType.LIKWID_CPU:
            sdfg_id = sdfg.sdfg_id
            state_id = sdfg.node_id(state)
            node_id = -1
            region = f"state_{sdfg_id}_{state_id}_{node_id}"

            sdfg_path = sdfg.build_folder.replace('\\', '/')
            report_code = f'''
void __dace_generate_report() {{
    int error = perfmon_stopCounters();
    if (error < 0)
    {{
        printf("Failed to stop counters for group %d for thread %d\\n", LIKWID_ACTIVE_GID, (-1*error)-1);
        perfmon_finalize();
        topology_finalize();
        exit(1);
    }}

    dace::perf::Report report;
    report.reset();

    CpuTopology_t topo = get_cpuTopology();
    int num_events = perfmon_getNumberOfEvents(LIKWID_ACTIVE_GID);
    for (int j = 0; j < num_events; j++)
    {{
        char* event_name = perfmon_getEventName(LIKWID_ACTIVE_GID, j);
        for (int i = 0; i < topo->numHWThreads; i++)
        {{
            double result = perfmon_getResult(LIKWID_ACTIVE_GID, j, i);
            std::cout << result << "\\n";
            report.add_counter("{region}", "likwid", event_name, result, i, {sdfg_id}, {state_id}, {node_id});
        }}
    }}

    report.save("{sdfg_path}/perf", __HASH_{sdfg.name});

    perfmon_finalize();
    affinity_finalize();
    topology_finalize();
}}
'''
            global_stream.write(report_code, sdfg)

            call_code = "__dace_generate_report();\n"
            local_stream.write(call_code, sdfg)


@registry.autoregister_params(type=dtypes.InstrumentationType.LIKWID_GPU)
class LIKWIDInstrumentationGPU(InstrumentationProvider):
    """ Instrumentation provider that reports CPU performance counters using
        the Likwid tool.
    """

    perf_whitelist_schedules = [dtypes.ScheduleType.GPU_Default, dtypes.ScheduleType.GPU_Device]

    def __init__(self):
        self._likwid_used = False
        self._regions = []

        try:
            self._default_events = Config.get('instrumentation', 'likwid', 'default_events')
        except KeyError:
            self._default_events = "FLOPS_SP"

    def on_sdfg_begin(self, sdfg, local_stream, global_stream, codegen):
        if sdfg.parent is not None:
            return

        # Configure CMake project and counters
        self._likwid_used = LIKWID.is_installed()
        if not self._likwid_used:
            return

        codegen.dispatcher.used_environments.add(LIKWID.full_class_path())
        Config.append('compiler', 'cpu', 'args', value=' -DLIKWID_NVMON ')

        self.codegen = codegen

        likwid_marker_file_gpu = Path(sdfg.build_folder) / "perf" / "likwid_marker_gpu.out"

        # Add instrumentation includes and initialize LIKWID
        header_code = '''
#include <likwid-marker.h>

#include <unistd.h>
#include <string>

#define MAX_NUM_EVENTS 1024
#define MAX_NUM_GPUS 1
'''
        global_stream.write(header_code, sdfg)

        init_code = f'''
setenv("LIKWID_GPUS", "0", 0);
setenv("LIKWID_GEVENTS", "{self._default_events}", 0);
setenv("LIKWID_GPUFILEPATH", "{likwid_marker_file_gpu.absolute()}", 0);

LIKWID_NVMARKER_INIT;
'''
        codegen._initcode.write(init_code)

    def on_sdfg_end(self, sdfg, local_stream, global_stream):
        if not self._likwid_used or sdfg.parent is not None:
            return

        for region, sdfg_id, state_id, node_id in self._regions:
            report_code = f'''
{{
    double *events = (double*) malloc(MAX_NUM_EVENTS * sizeof(double));
    double time = 0.0;
    int nevents = MAX_NUM_EVENTS;
    int ngpus = MAX_NUM_GPUS;
    int count = 0;

    LIKWID_NVMARKER_GET("{region}", &ngpus, &nevents, &events, &time, &count);

    __state->report.add_completion("Timer", "likwid_gpu", 0, time * 1000 * 1000, 0, {sdfg_id}, {state_id}, {node_id});
    
    int gid = nvmon_getIdOfActiveGroup();
    for (int i = 0; i < nevents; i++)
    {{
        char* event_name = nvmon_getEventName(gid, i); 
        
        __state->report.add_counter("{region}", "likwid_gpu", event_name, events[i], 0, {sdfg_id}, {state_id}, {node_id});
    }}

    free(events);
}}
'''
            local_stream.write(report_code)

        exit_code = '''
LIKWID_NVMARKER_CLOSE;
'''
        self.codegen._exitcode.write(exit_code, sdfg)

    def on_state_begin(self, sdfg, state, local_stream, global_stream):
        if not self._likwid_used:
            return

        if state.instrument == dace.InstrumentationType.LIKWID_GPU:
            sdfg_id = sdfg.sdfg_id
            state_id = sdfg.node_id(state)
            node_id = -1
            region = f"state_{sdfg_id}_{state_id}_{node_id}"
            self._regions.append((region, sdfg_id, state_id, node_id))

            marker_code = f'''
LIKWID_NVMARKER_REGISTER("{region}");

LIKWID_NVMARKER_START("{region}");
LIKWID_NVMARKER_STOP("{region}");
LIKWID_NVMARKER_RESET("{region}");

LIKWID_NVMARKER_START("{region}");
'''
            local_stream.write(marker_code)

    def on_state_end(self, sdfg, state, local_stream, global_stream):
        if not self._likwid_used:
            return

        if state.instrument == dace.InstrumentationType.LIKWID_GPU:
            sdfg_id = sdfg.sdfg_id
            state_id = sdfg.node_id(state)
            node_id = -1
            region = f"state_{sdfg_id}_{state_id}_{node_id}"

            marker_code = f'''
LIKWID_NVMARKER_STOP("{region}");
'''
            local_stream.write(marker_code)

    def on_scope_entry(self, sdfg, state, node, outer_stream, inner_stream, global_stream):
        if not self._likwid_used or node.instrument != dace.InstrumentationType.LIKWID_GPU:
            return

        if not isinstance(node, dace.nodes.MapEntry) or xfh.get_parent_map(state, node) is not None:
            raise TypeError("Only top-level map scopes supported")
        elif node.schedule not in LIKWIDInstrumentationGPU.perf_whitelist_schedules:
            raise TypeError("Unsupported schedule on scope")

        sdfg_id = sdfg.sdfg_id
        state_id = sdfg.node_id(state)
        node_id = state.node_id(node)
        region = f"scope_{sdfg_id}_{state_id}_{node_id}"

        self._regions.append((region, sdfg_id, state_id, node_id))
        marker_code = f'''
LIKWID_NVMARKER_REGISTER("{region}");

LIKWID_NVMARKER_START("{region}");
LIKWID_NVMARKER_STOP("{region}");
LIKWID_NVMARKER_RESET("{region}");

LIKWID_NVMARKER_START("{region}");
'''
        outer_stream.write(marker_code)

    def on_scope_exit(self, sdfg, state, node, outer_stream, inner_stream, global_stream):
        entry_node = state.entry_node(node)
        if not self._likwid_used or entry_node.instrument != dace.InstrumentationType.LIKWID_GPU:
            return

        sdfg_id = sdfg.sdfg_id
        state_id = sdfg.node_id(state)
        node_id = state.node_id(entry_node)
        region = f"scope_{sdfg_id}_{state_id}_{node_id}"

        marker_code = f'''
LIKWID_NVMARKER_STOP("{region}");
'''
        outer_stream.write(marker_code)
