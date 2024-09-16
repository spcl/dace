# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests analysis passes related to control flow regions (control_flow_region_analysis.py). """


import dace
from dace.memlet import Memlet
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.analysis.control_flow_region_analysis import CFGDataDependence, StateDataDependence


def test_simple_state_data_dependence_with_self_contained_read():
    N = dace.symbol('N')

    @dace.program
    def myprog(A: dace.float64[N], B: dace.float64):
        for i in dace.map[0:N/2]:
            with dace.tasklet:
                in1 << B[i]
                out1 >> A[i]
                out1 = in1 + 1
            with dace.tasklet:
                in1 << B[i]
                out1 >> B[N - (i + 1)]
                out1 = in1 - 1
        for i in dace.map[0:N/2]:
            with dace.tasklet:
                in1 << A[i]
                out1 >> B[i]
                out1 = in1 * 2

    sdfg = myprog.to_sdfg()

    res = {}
    Pipeline([StateDataDependence()]).apply_pass(sdfg, res)
    state_data_deps = res[StateDataDependence.__name__][0][sdfg.states()[0]]

    assert len(state_data_deps[0]) == 1
    read_memlet: Memlet = list(state_data_deps[0])[0]
    assert read_memlet.data == 'B'
    assert read_memlet.subset[0][0] == 0
    assert read_memlet.subset[0][1] == 0.5 * N - 1 or read_memlet.subset[0][1] == N / 2 - 1

    assert len(state_data_deps[1]) == 3


'''
def test_nested_cf_region_data_dependence():
    N = dace.symbol('N')

    @dace.program
    def myprog(A: dace.float64[N], B: dace.float64):
        for i in range(N):
            with dace.tasklet:
                in1 << B[i]
                out1 >> A[i]
                out1 = in1 + 1
        for i in range(N):
            with dace.tasklet:
                in1 << A[i]
                out1 >> B[i]
                out1 = in1 * 2

    myprog.use_experimental_cfg_blocks = True

    sdfg = myprog.to_sdfg()

    res = {}
    pipeline = Pipeline([CFGDataDependence()])
    pipeline.__experimental_cfg_block_compatible__ = True
    pipeline.apply_pass(sdfg, res)

    print(sdfg)
    '''


if __name__ == '__main__':
    test_simple_state_data_dependence_with_self_contained_read()
    #test_nested_cf_region_data_dependence()
