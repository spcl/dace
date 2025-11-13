# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import numpy as np

import dace.codegen.instrumentation.papi as pp
M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')


@dace.program
def matmul_papi(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    C = A@B


##### DaCe + PAPI: Matmul Instrumentation #####
# This sample demonstrates the PAPI instrumentation in DaCe.
#
# In order to run the sample, make sure that PAPI is installed on your system and msr access 
# is granted. (Either set kernel paranoia to 0 or run as root) 
# It is recommended to set the OMP_NUM_THREADS environment variable
#
# Example: 'OMP_NUM_THREADS=2 python matmul_papi.py'
#
# To check which events are available on your machine, run papi_avail in your command line.
# To check which native events are PAPI exposes on your system, use papi_native_avail
#
# Note that native events are only supported by the whole sdfg instrumentation!
#
# You can check whether a set of events can be measured simultaneously by using the command:
# papi_event_chooser <PRESET|NATIVE> <event1> <event2> ...
# Example: papi_event_chooser PRESET PAPI_DP_OPS PAPI_TOT_INS

## 1. Setup: SDFG + data
# Convert to SDFG
sdfg = matmul_papi.to_sdfg()
sdfg.expand_library_nodes()
sdfg.simplify()

# Specialize SDFG for input sizes
m = 512
k = 512
n = 512

# Create arrays
A = np.random.rand(m, k).astype(np.float32)
B = np.random.rand(k, n).astype(np.float32)
C = np.zeros((m, n), dtype=np.float32)

## 2. Instrumentation

# set which events should be counted
pp.PAPIInstrumentation._counters = {'PAPI_SP_OPS', 'PAPI_TOT_INS'}

## 2.1 Run with whole sdfg instrumentation
sdfg_complete_papi = copy.deepcopy(sdfg)
sdfg_complete_papi.name = sdfg_complete_papi.name + "_complete"
# set the instrumentation of the top-level SDFG to PAPI
sdfg_complete_papi.instrument = dace.InstrumentationType.PAPI_Counters

## 2.2 Run with instrumentation only at spcific nodes 
# Node Types that can be instrumented here are: SDFGState, MapEntry, 

sdfg_selected_papi = copy.deepcopy(sdfg)
sdfg_selected_papi.name = sdfg_selected_papi.name+"_slected"
for node in sdfg_selected_papi.nodes():
    node.instrument = dace.InstrumentationType.PAPI_Counters
    if node.name == "SDFGState _MatMult_gemm_state_0":
        for sub_node in node.nodes():
            try:
                sub_node.instrument = dace.InstrumentationType.PAPI_Counters
            except Exception as e:
                print("Exception while trying to instrument", sub_node, ":", e)
## 3. Compile and execute
# During execution, the counters for different parts of the SDFG and different
# threads are measured by PAPI and written into a performance report
# in form of events. This report is saved at .dacecache/matmul/perf.
csdfg_complete_papi = sdfg_complete_papi.compile()
csdfg_complete_papi(A=A, B=B, C=C, K=k, M=m, N=n)

csdfg_selected_papi = sdfg_selected_papi.compile()
csdfg_selected_papi(A=A, B=B, C=C, K=k, M=m, N=n)

## 4. Report
# We can now parse the performance report into a python-object
# and read different counters or timers. Furthermore, the report
# provides a table-like print.
report_complete_papi = sdfg_complete_papi.get_latest_report()
report_selected_papi = sdfg_selected_papi.get_latest_report()


# Print human-readable table
# Tip: Try this feature with only a 1-2 on instrumented states/nodes.
print("Complete SDFG PAPI instrumentation report:")
print(report_complete_papi)


print("Selected SDFG element instrumentation report:")
print(report_selected_papi)

# Access counters
# We will now demonstrate how to access the raw values from the report
# on the example of number of SP FLOPS. Those are measured
# when executing the sample with LIKWID_EVENTS="FLOPS_SP".
#
# Counter values are grouped by the SDFG element which defines the scope
# of the intrumentation. Those elements are described as the triplet
# (cfg_id, state_id, node_id).

# ~ expected FLOPS
expected_flops = m * k * (n * 2)

print(f"Expected {expected_flops} FLOPS, measured {"measured_flops"} FLOPS, diff: {"measured_flops" - expected_flops}")
