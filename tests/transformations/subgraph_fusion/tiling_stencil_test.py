import dace
from dace.sdfg import SDFG
from dace.transformation.subgraph.stencil_tiling import StencilTiling
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg.graph import SubgraphView

import numpy as np

N = dace.symbol('N')
N.set(100)

@dace.program
def stencil(A: dace.float64[N], B: dace.float64[N]):
    tmp1 = np.ndarray(shape=[N], dtype = dace.float64)
    tmp2 = np.ndarray(shape=[N], dtype = dace.float64)
    tmp3 = np.ndarray(shape=[N], dtype = dace.float64)

    @dace.map
    def m1(i: _[1:N]):
        in1 << A[i]
        in2 << A[i-1]
        out1 >> tmp1[i]
        out1 = (in1 - in2)/float(2.0)

    @dace.map
    def m2(i: _[0:N-1]):
        in1 << A[i]
        in2 << A[i+1]
        out1 >> tmp2[i]
        out1 = (in2 - in1)/float(2.0)

    @dace.map
    def m3(i: _[1:N-1]):
        in1_1 << tmp1[i]
        in1_2 << tmp1[i+1]

        in2_1 << tmp2[i]
        in2_0 << tmp2[i-1]

        out1 >> tmp3[i]

        out1 = in1_1 - float(0.2)*in1_2 + in2_1 + float(0.8)*in2_0

    @dace.map
    def m4(i: _[2:N-2]):
        in1 << tmp3[i-1:i+2]
        out1 >> B[i]

        out1 = in1[1]-0.5*(in1[0] + in1[2])


@dace.program
def stencil_offset(A: dace.float64[N], B: dace.float64[N]):
    tmp1 = np.ndarray(shape=[N], dtype = dace.float64)
    tmp2 = np.ndarray(shape=[N], dtype = dace.float64)
    tmp3 = np.ndarray(shape=[N], dtype = dace.float64)

    @dace.map
    def m1(i: _[0:N-1]):
        in1 << A[i+1]
        in2 << A[i-1+1]
        out1 >> tmp1[i+1]
        out1 = (in1 - in2)/float(2.0)

    @dace.map
    def m2(i: _[1:N]):
        in1 << A[i-1]
        in2 << A[i+1-1]
        out1 >> tmp2[i-1]
        out1 = (in2 - in1)/float(2.0)

    @dace.map
    def m3(i: _[2:N]):
        in1_1 << tmp1[i-1]
        in1_2 << tmp1[i+1-1]

        in2_1 << tmp2[i-1]
        in2_0 << tmp2[i-1-1]

        out1 >> tmp3[i-1]

        out1 = in1_1 - float(0.2)*in1_2 + in2_1 + float(0.8)*in2_0

    @dace.map
    def m4(i: _[0:N-4]):
        in1 << tmp3[i-1+2:i+2+2]
        out1 >> B[i+2]

        out1 = in1[1]-0.5*(in1[0] + in1[2])

def test_stencil(tile_size, offset = False, unroll = False, view = False):

    A = np.random.rand(N.get()).astype(np.float64)
    B1 = np.zeros((N.get()), dtype = np.float64)
    B2 = np.zeros((N.get()), dtype = np.float64)
    B3 = np.zeros((N.get()), dtype = np.float64)

    if offset:
        sdfg = stencil_offset.to_sdfg()
    else:
        sdfg = stencil.to_sdfg()
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]

    if view:
        sdfg.view()
    # baseline
    sdfg._name = 'baseline'
    csdfg = sdfg.compile()
    csdfg(A=A, B=B1, N=N)
    del csdfg

    subgraph = SubgraphView(graph, [n for n in graph.nodes()])
    st = StencilTiling(subgraph)
    st.tile_size = (tile_size,)
    st.schedule = dace.dtypes.ScheduleType.Sequential
    assert st.can_be_applied(sdfg, subgraph)
    if unroll:
        st.unroll_loops = True
    st.apply(sdfg)
    if view:
        sdfg.view()
    sdfg._name = 'tiled'
    sdfg.validate()
    csdfg = sdfg.compile()
    csdfg(A=A, B=B2, N=N)
    del csdfg
    assert np.allclose(B1, B2)

    sdfg.apply_strict_transformations()
    subgraph = SubgraphView(graph, [n for n in graph.nodes()])
    sf = SubgraphFusion(subgraph)
    # also test consolidation
    sf.consolidate = True
    sf.apply(sdfg)
    sdfg._name = 'fused'
    csdfg = sdfg.compile()
    csdfg(A=A, B=B3, N=N)
    del csdfg

    print(np.linalg.norm(B1))
    print(np.linalg.norm(B3))
    assert np.allclose(B1, B2)
    assert np.allclose(B1, B3)
    print("PASS")


if __name__ == '__main__':
    test_stencil(1, offset = False, unroll = False)
    test_stencil(8, offset = False, unroll = False)
    test_stencil(1, offset = True, unroll = False)
    test_stencil(8, offset = True, unroll = False)
    test_stencil(1, offset = False, unroll = True)
    test_stencil(8, offset = False, unroll = True)
    test_stencil(1, offset = True, unroll = True)
    test_stencil(8, offset = True, unroll = True)
