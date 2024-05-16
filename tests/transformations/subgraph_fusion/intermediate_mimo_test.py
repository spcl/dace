# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
from dace.sdfg import nodes
from dace.sdfg.graph import SubgraphView
from dace.transformation.helpers import nest_state_subgraph
import numpy as np
import unittest
import sys

from dace.transformation.subgraph import MultiExpansion, SubgraphFusion

from typing import Union, List
from dace.sdfg.graph import SubgraphView

N = dace.symbol('N')


@dace.program
def mimo(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):

    for i in dace.map[0:N // 2]:
        with dace.tasklet:
            in1 << A[2 * i]
            in2 << A[2 * i + 1]
            out >> C[2 * i]

            out = in1 + in2

    for i in dace.map[0:N // 2]:
        with dace.tasklet:
            in1 << B[2 * i]
            in2 << B[2 * i + 1]
            out >> C[2 * i + 1]

            out = in1 + in2

    for i in dace.map[0:N // 2]:
        with dace.tasklet:
            in1 << C[2 * i:2 * i + 2]
            out1 >> D[2 * i:2 * i + 2]

            out1[0] = in1[0] * in1[0]
            out1[1] = in1[1] * in1[1]


def _test_quantitatively(sdfg):
    graph = sdfg.nodes()[0]
    A = np.random.rand(1000).astype(np.float64)
    B = np.random.rand(1000).astype(np.float64)
    C1 = np.random.rand(1000).astype(np.float64)
    C2 = np.random.rand(1000).astype(np.float64)
    D1 = np.random.rand(1000).astype(np.float64)
    D2 = np.random.rand(1000).astype(np.float64)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C1, D=D1, N=1000)
    del csdfg

    subgraph = SubgraphView(graph, [node for node in graph.nodes()])

    me = MultiExpansion()
    me.setup_match(subgraph)
    assert me.can_be_applied(sdfg, subgraph) == True
    me.apply(sdfg)

    sf = SubgraphFusion()
    sf.setup_match(subgraph)
    assert sf.can_be_applied(sdfg, subgraph) == True
    sf.apply(sdfg)

    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C2, D=D2, N=1000)

    assert np.allclose(C1, C2)
    assert np.allclose(D1, D2)


def test_mimo():
    sdfg = mimo.to_sdfg()
    from dace.transformation.interstate.state_fusion import StateFusion
    sdfg.apply_transformations_repeated(StateFusion, permissive=True)
    # merge the C array
    C1 = None
    C2 = None
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.AccessNode) and node.data == 'C':
            if not C1:
                C1 = node
            elif not C2:
                C2 = node
                break
    if C1 is not None and C2 is not None:
        dace.sdfg.utils.change_edge_dest(sdfg.nodes()[0], C2, C1)
        dace.sdfg.utils.change_edge_src(sdfg.nodes()[0], C2, C1)
        sdfg.nodes()[0].remove_node(C2)

    sdfg.validate()
    _test_quantitatively(sdfg)


def test_single_data_multiple_intermediate_accesses():

    @dace.program
    def sdmi_accesses(ZSOLQA: dace.float64[1, 5, 5], ZEPSEC: dace.float64, ZQX: dace.float64[1, 137, 5],
                      LLINDEX3: dace.bool[1, 5, 5], ZRATIO: dace.float64[1, 5], ZSINKSUM: dace.float64[1, 5]):
        
        for i in dace.map[0:5]:
            ZSINKSUM[0, i] = 0.0
            for j in dace.map[0:5]:
                LLINDEX3[0, j, i] = False
        
        for i in dace.map[0:5]:
            for k in range(5):
                ZSINKSUM[0, i] = ZSINKSUM[0, i] - ZSOLQA[0, 0, k]
        
        for i in dace.map[0:5]:
            t0 = max(ZEPSEC, ZQX[0, 0, i])
            t1 = max(t0, ZSINKSUM[0, i])
            ZRATIO[0, i] = t0 / t1
    
    sdfg = sdmi_accesses.to_sdfg(simplify=True)
    assert len(sdfg.states()) == 1

    rng = np.random.default_rng(42)
    ZSOLQA = rng.random((1, 5, 5))
    ZEPSEC = rng.random()
    ZQX = rng.random((1, 137, 5))
    ref_LLINDEX3 = rng.random((1, 5, 5)) > 0.5
    ref_ZRATIO = rng.random((1, 5))
    ref_ZSINKSUM = rng.random((1, 5))
    val_LLINDEX3 = ref_LLINDEX3.copy()
    val_ZRATIO = ref_ZRATIO.copy()
    val_ZSINKSUM = ref_ZSINKSUM.copy()

    sdfg(ZSOLQA=ZSOLQA, ZEPSEC=ZEPSEC, ZQX=ZQX, LLINDEX3=ref_LLINDEX3, ZRATIO=ref_ZRATIO, ZSINKSUM=ref_ZSINKSUM)

    graph = sdfg.states()[0]
    subgraph = SubgraphView(graph, [node for node in graph.nodes()])

    me = MultiExpansion()
    me.setup_match(subgraph)
    assert me.can_be_applied(sdfg, subgraph) == True
    me.apply(sdfg)

    sf = SubgraphFusion()
    sf.setup_match(subgraph)
    assert sf.can_be_applied(sdfg, subgraph) == True
    sf.apply(sdfg)

    sdfg(ZSOLQA=ZSOLQA, ZEPSEC=ZEPSEC, ZQX=ZQX, LLINDEX3=val_LLINDEX3, ZRATIO=val_ZRATIO, ZSINKSUM=val_ZSINKSUM)

    assert np.allclose(ref_LLINDEX3, val_LLINDEX3)
    assert np.allclose(ref_ZRATIO, val_ZRATIO)
    assert np.allclose(ref_ZSINKSUM, val_ZSINKSUM)


if __name__ == '__main__':
    test_mimo()
    test_single_data_multiple_intermediate_accesses()
