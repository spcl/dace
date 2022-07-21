# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

import dace.transformation.helpers as xfh

M = dace.symbol('M')
K = dace.symbol('K')
N = dace.symbol('N')


@dace.program
def matmul(A: dace.float32[M, K], B: dace.float32[K, N], C: dace.float32[M, N]):
    tmp = np.ndarray([M, N, K], dtype=A.dtype)
    for i, j, k in dace.map[0:M, 0:N, 0:K]:
        with dace.tasklet:
            in_A << A[i, k]
            in_B << B[k, j]
            out >> tmp[i, j, k]

            out = in_A * in_B
    dace.reduce(lambda a, b: a + b, tmp, C, axis=2, identity=0)


##### DaCe + Likwid: Matmul Instrumentation #####
# This sample demonstrates the likwid instrumentation in DaCe.
#
# In order to run the sample, specific environment variables must be set
# - OMP_NUM_THREADS: number of threads [1, num procs]
# - LIKWID_EVENTS: set of counters to be measured [FLOPS_SP, CACHE, MEM, ...]
#
# Example: 'OMP_NUM_THREADS=2 LIKWID_EVENTS="FLOPS_SP" python matmul_likwid.py'
#
# The available event set for your architecture can be found in the likwid
# groups folder: https://github.com/RRZE-HPC/likwid/tree/master/groups

## 1. Setup: SDFG + data
# Convert to SDFG
sdfg = matmul.to_sdfg()
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
# We will now iterate through the SDFG and set the instrumentation
# type to LIKWID_Counters for all states and top-level map entries.
# Non-top-level map entries are currently not supported!
for nsdfg in sdfg.all_sdfgs_recursive():
    for state in nsdfg.nodes():
        state.instrument = dace.InstrumentationType.LIKWID_Counters
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry) and xfh.get_parent_map(state, node) is None:
                node.instrument = dace.InstrumentationType.LIKWID_Counters

## 3. Compile and execute
# During execution, the counters for different parts of the SDFG and different
# threads are measured by likwid and written into a performance report
# in form of events. This report is saved at .dacecache/matmul/perf.
csdfg = sdfg.compile()
csdfg(A=A, B=B, C=C, K=k, M=m, N=n)

## 4. Report
# We can now parse the performance report into a python-object
# and read different counters or timers. Furthermore, the report
# provides a table-like print.
report = sdfg.get_latest_report()

# Print human-readable table
# Tip: Try this feature with only a 1-2 on instrumented states/nodes.
print(report)

# Access counters
# We will now demonstrate how to access the raw values from the report
# on the example of number of SP FLOPS. Those are measured
# when executing the sample with LIKWID_EVENTS="FLOPS_SP".
#
# Counter values are grouped by the SDFG element which defines the scope
# of the intrumentation. Those elements are described as the triplet
# (sdfg_id, state_id, node_id).

measured_flops = 0
flops_report = report.counters[(0, 0, -1)]["RETIRED_SSE_AVX_FLOPS_SINGLE_ALL"]
for tid in flops_report:
    measured_flops += flops_report[tid][0]

# ~ expected FLOPS
expected_flops = m * k * (n * 2)

print(f"Expected {expected_flops} FLOPS, measured {measured_flops} FLOPS, diff: {measured_flops - expected_flops}")
