# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests that generate various instrumentation reports with timers and
    performance counters. """

import pytest
import numpy as np
import re
import sys

import dace
from dace.sdfg import nodes
from dace.transformation.interstate import GPUTransformSDFG

N = dace.symbol('N')


@dace.program
def slowmm(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for t in range(20):

        @dace.map
        def mult(i: _[0:N], j: _[0:N], k: _[0:N]):
            a << A[i, k]
            b << B[k, j]
            c >> C(1, lambda a, b: a + b)[i, j]
            c = a * b


def onetest(instrumentation: dace.InstrumentationType, size=128):
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.zeros([size, size], dtype=np.float64)

    sdfg: dace.SDFG = slowmm.to_sdfg()
    sdfg.name = f"instrumentation_test_{instrumentation.name}"
    sdfg.simplify()

    # Set instrumentation both on the state and the map
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.label == 'mult':
            node.map.instrument = instrumentation
            state.instrument = instrumentation

    if instrumentation in [dace.InstrumentationType.GPU_Events, dace.InstrumentationType.GPU_TX_MARKERS]:
        sdfg.apply_transformations(GPUTransformSDFG)

    with dace.instrument(instrumentation,
                         filter='*',
                         annotate_maps=True,
                         annotate_tasklets=False,
                         annotate_states=True,
                         annotate_sdfgs=True):
        sdfg(A=A, B=B, C=C, N=size)

    # Check for correctness
    assert np.allclose(C, 20 * A @ B)

    # Print instrumentation report
    if sdfg.is_instrumented():
        print('Instrumentation report')
        report = sdfg.get_latest_report()
        print(report)

    # Check that the NVTX/rocTX range wrapper is present in the generated CPU code
    if instrumentation == dace.InstrumentationType.GPU_TX_MARKERS:
        code = sdfg.generate_code()[0].clean_code
        tx_include = re.search(r'#include <(nvtx3/nvToolsExt|roctx).h>', code)
        assert tx_include is not None
        range_push = re.search(r'(nvtx|roctx)RangePush\("sdfg', code) is not None
        range_push &= re.search(r'(nvtx|roctx)RangePush\("copy', code) is not None
        range_push &= re.search(r'(nvtx|roctx)RangePush\("state', code) is not None
        range_push &= re.search(r'(nvtx|roctx)RangePush\("alloc', code) is not None
        range_push &= re.search(r'(nvtx|roctx)RangePush\("dealloc', code) is not None
        range_push &= re.search(r'(nvtx|roctx)RangePush\("init', code) is not None
        range_push &= re.search(r'(nvtx|roctx)RangePush\("exit', code) is not None
        assert range_push
        range_pop = re.search(r'(nvtx|roctx)RangePop\b', code)
        assert range_pop is not None


def test_timer():
    onetest(dace.InstrumentationType.Timer)


@pytest.mark.papi
def test_papi():
    # Run a lighter load for the sake of performance
    onetest(dace.InstrumentationType.PAPI_Counters, 4)


@pytest.mark.gpu
def test_gpu_events():
    onetest(dace.InstrumentationType.GPU_Events)


@pytest.mark.gpu
def test_gpu_tx_markers():
    onetest(dace.InstrumentationType.GPU_TX_MARKERS)


if __name__ == '__main__':
    test_timer()
    test_papi()
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        test_gpu_events()
        test_gpu_tx_markers()
