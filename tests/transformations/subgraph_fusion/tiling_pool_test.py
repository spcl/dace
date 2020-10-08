import dace
from dace.sdfg import SDFG
from dace.transformation.subgraph.stencil_tiling import StencilTiling
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg.graph import SubgraphView

import numpy as np

N = dace.symbol('N')
N.set(100)

@dace.program
def stencil(A: dace.float32[2*N], B: dace.float32[N]):
    tmp1 = np.ndarray([N], dtype = dace.float32)

    @dace.map
    def m1(i: _[0:N]):
        in1 << A[2*i]
        in2 << A[2*i+1]
        out1 >> tmp1[i]
        out1 = (in1 + in2)/float(2.0)

    @dace.map
    def m2(i: _[1:N-1]):
        in1 << tmp1[i]
        in2 << tmp1[i+1]
        in3 << tmp1[i-1]
        out1 >> B[i]
        out1 = (in2 - 0.2 * in1 - 0.2 * in3)

@dace.program
def stencil_offset(A: dace.float32[2*N], B: dace.float32[N]):
    tmp1 = np.ndarray(shape=[N], dtype = dace.float32)

    @dace.map
    def m1(i: _[0:N]):
        in1 << A[2*i]
        in2 << A[2*i+1]
        out1 >> tmp1[i]
        out1 = (in1 + in2)/float(2.0)

    @dace.map
    def m2(i: _[0:N-2]):
        in1 << tmp1[i+1]
        in2 << tmp1[i+1+1]
        in3 << tmp1[i-1+1]
        out1 >> B[i+1]
        out1 = (in2 - 0.2 * in1 - 0.2 * in3)



def test_stencil(tile_size, offset = False):

    A = np.random.rand(N.get()*2).astype(np.float32)
    B1 = np.zeros((N.get()), dtype = np.float32)
    B2 = np.zeros((N.get()), dtype = np.float32)
    B3 = np.zeros((N.get()), dtype = np.float32)

    if offset:
        sdfg = stencil_offset.to_sdfg()
    else:
        sdfg = stencil.to_sdfg()
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]

    # baseline
    sdfg._name = 'baseline'
    csdfg = sdfg.compile()
    csdfg(A=A, B=B1, N=N)

    subgraph = SubgraphView(graph, [n for n in graph.nodes()])
    st = StencilTiling(subgraph)
    st.tile_size = (tile_size,)
    st.schedule = dace.dtypes.ScheduleType.Sequential
    assert st.can_be_applied(sdfg, subgraph)
    # so that OMP never fails
    st.schedule = dace.dtypes.ScheduleType.Sequential
    st.apply(sdfg)

    sdfg._name = 'tiled'
    csdfg = sdfg.compile()
    csdfg(A=A, B=B2, N=N)

    sdfg.apply_strict_transformations()
    subgraph = SubgraphView(graph, [n for n in graph.nodes()])
    sf = SubgraphFusion(subgraph)

    sdfg._name = 'fused'
    csdfg = sdfg.compile()
    csdfg(A=A, B=B3, N=N)

    assert np.allclose(B1, B2)
    assert np.allclose(B1, B3)



if __name__ == '__main__':
    test_stencil(1, offset = False)
    test_stencil(8, offset = False)
    test_stencil(1, offset = True)
    test_stencil(8, offset = True)
