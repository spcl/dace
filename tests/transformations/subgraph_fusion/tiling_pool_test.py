# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
from dace.sdfg import SDFG
from dace.transformation.subgraph.stencil_tiling import StencilTiling
from dace.transformation.subgraph import SubgraphFusion
from dace.sdfg.graph import SubgraphView

import numpy as np
import pytest
import itertools

N = dace.symbol('N')
N.set(100)


@dace.program
def stencil(A: dace.float64[2 * N], B: dace.float64[N]):
    tmp1 = np.ndarray([N], dtype=dace.float64)

    @dace.map
    def m1(i: _[0:N]):
        in1 << A[2 * i]
        in2 << A[2 * i + 1]
        out1 >> tmp1[i]
        out1 = (in1 + in2) / float(2.0)

    @dace.map
    def m2(i: _[1:N - 1]):
        in1 << tmp1[i]
        in2 << tmp1[i + 1]
        in3 << tmp1[i - 1]
        out1 >> B[i]
        out1 = (in2 - 0.2 * in1 - 0.2 * in3)


@dace.program
def stencil_offset(A: dace.float64[2 * N], B: dace.float64[N]):
    tmp1 = np.ndarray(shape=[N], dtype=dace.float64)

    @dace.map
    def m1(i: _[0:N]):
        in1 << A[2 * i]
        in2 << A[2 * i + 1]
        out1 >> tmp1[i]
        out1 = (in1 + in2) / float(2.0)

    @dace.map
    def m2(i: _[0:N - 2]):
        in1 << tmp1[i + 1]
        in2 << tmp1[i + 1 + 1]
        in3 << tmp1[i - 1 + 1]
        out1 >> B[i + 1]
        out1 = (in2 - 0.2 * in1 - 0.2 * in3)


def invoke_stencil(tile_size, offset=False, unroll=False):

    A = np.random.rand(N.get() * 2).astype(np.float64)
    B1 = np.zeros((N.get()), dtype=np.float64)
    B2 = np.zeros((N.get()), dtype=np.float64)
    B3 = np.zeros((N.get()), dtype=np.float64)

    if offset:
        sdfg = stencil_offset.to_sdfg()
    else:
        sdfg = stencil.to_sdfg()
    sdfg.simplify()
    graph = sdfg.nodes()[0]

    # baseline
    sdfg.name = f'baseline_{tile_size}_{offset}_{unroll}'
    csdfg = sdfg.compile()
    csdfg(A=A, B=B1, N=N)
    del csdfg

    subgraph = SubgraphView(graph, [n for n in graph.nodes()])
    st = StencilTiling(subgraph)
    st.tile_size = (tile_size, )
    st.unroll_loops = unroll
    assert st.can_be_applied(sdfg, subgraph)
    # change schedule so that OMP never fails
    st.schedule = dace.dtypes.ScheduleType.Sequential
    st.apply(sdfg)

    sdfg.name = f'tiled_{tile_size}_{offset}_{unroll}'
    csdfg = sdfg.compile()
    csdfg(A=A, B=B2, N=N)
    del csdfg

    sdfg.simplify()
    subgraph = SubgraphView(graph, [n for n in graph.nodes()])
    sf = SubgraphFusion(subgraph)
    assert sf.can_be_applied(sdfg, subgraph)
    sf.apply(sdfg)

    sdfg.name = f'fused_{tile_size}_{offset}_{unroll}'
    csdfg = sdfg.compile()
    csdfg(A=A, B=B3, N=N)
    del csdfg

    print(np.linalg.norm(B1))
    print(np.linalg.norm(B3))

    print("PASS")


test_settings = list(itertools.product([1, 8], [False, True], [False, True]))


@pytest.mark.parametrize(["tile", "offset", "unroll"], test_settings)
def test_all(tile, offset, unroll):
    invoke_stencil(tile, offset, unroll)


if __name__ == '__main__':
    for (t, o, u) in itertools.product([1, 8], [False, True], [False, True]):
        print(f"Testing config {t}, {o}, {u}")
        test_all(t, o, u)
