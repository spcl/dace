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

# State struct field that carries the instrumentation report
REPORT_DECL = 'dace::perf::Report report;'


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

        # GPU_TX_MARKERS never writes to the report, so it must not be declared nor used when it is
        # the only instrumentation in use. Both save sites are checked, as they are guarded by the
        # `instrumentation.report_each_invocation` configuration entry.
        for each_invocation in (True, False):
            with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=each_invocation):
                code = sdfg.generate_code()[0].clean_code
            assert REPORT_DECL not in code
            assert re.search(r'__state->report\b', code) is None


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


@pytest.mark.gpu
def test_gpu_tx_markers_with_timer():
    """ The report is still needed when another instrumentation type is used next to GPU_TX_MARKERS. """
    sdfg: dace.SDFG = slowmm.to_sdfg()
    sdfg.name = 'instrumentation_test_GPU_TX_MARKERS_with_timer'
    sdfg.simplify()

    # Mark the map with GPU_TX_MARKERS and the state containing it with a timer
    sdfg.instrument = dace.InstrumentationType.GPU_TX_MARKERS
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.label == 'mult':
            node.map.instrument = dace.InstrumentationType.GPU_TX_MARKERS
            state.instrument = dace.InstrumentationType.Timer

    sdfg.apply_transformations(GPUTransformSDFG)

    # Both providers are in use, so the ranges are emitted and the report is kept at either save site
    for each_invocation in (True, False):
        with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=each_invocation):
            code = sdfg.generate_code()[0].clean_code
        assert re.search(r'(nvtx|roctx)RangePush\(', code) is not None
        assert REPORT_DECL in code
        assert re.search(r'__state->report\b', code) is not None


if __name__ == '__main__':
    test_timer()
    test_papi()
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        test_gpu_events()
        test_gpu_tx_markers()
        test_gpu_tx_markers_with_timer()
