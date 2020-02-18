""" Tests that generate various instrumentation reports with timers and
    performance counters. """

import numpy as np
import sys

import dace
from dace.graph import nodes
from dace.transformation.interstate import GPUTransformSDFG

N = dace.symbol('N')


@dace.program
def slowmm(A: dace.float64[N, N], B: dace.float64[N, N],
           C: dace.float64[N, N]):
    for t in range(20):

        @dace.map
        def mult(i: _[0:N], j: _[0:N], k: _[0:N]):
            a << A[i, k]
            b << B[k, j]
            c >> C(1, lambda a, b: a + b)[i, j]
            c = a * b


def onetest(instrumentation: dace.InstrumentationType, size=128):
    N.set(size)
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.zeros([size, size], dtype=np.float64)

    sdfg: dace.SDFG = slowmm.to_sdfg()
    sdfg.apply_strict_transformations()

    # Set instrumentation both on the state and the map
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.label == 'mult':
            node.map.instrument = instrumentation
            state.instrument = instrumentation

    if instrumentation == dace.InstrumentationType.CUDA_Events:
        sdfg.apply_transformations(GPUTransformSDFG)

    sdfg(A=A, B=B, C=C, N=N)

    # Check for correctness
    assert np.allclose(C, 20 * A @ B)

    # Print instrumentation report
    if sdfg.is_instrumented():
        print('Instrumentation report')
        report = sdfg.get_latest_report()
        print(report)


def test_timer():
    onetest(dace.InstrumentationType.Timer)


def test_papi():
    # Run a lighter load for the sake of performance
    onetest(dace.InstrumentationType.PAPI_Counters, 4)


def test_cuda_events():
    onetest(dace.InstrumentationType.CUDA_Events)


if __name__ == '__main__':
    test_timer()
    test_papi()
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        test_cuda_events()
