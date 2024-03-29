# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import List, Union

import numpy as np
from util import expand_maps, expand_reduce, fusion

import dace
import dace.libraries.standard as stdlib
import dace.sdfg.nodes as nodes
import dace.transformation.subgraph.helpers as helpers
from dace.sdfg.graph import SubgraphView
from dace.transformation.dataflow import ReduceExpansion
from dace.transformation.subgraph import SubgraphFusion

M = dace.symbol('M')
N = dace.symbol('N')


# TRUE
@dace.program
def disjoint_test_1(A: dace.float64[M, 2]):

    for i in dace.map[0:M]:
        with dace.tasklet:
            in1 << A[i, 0]
            out1 >> A[i, 1]
            out1 = in1 * 2

    for i in dace.map[0:M]:
        with dace.tasklet:
            in1 << A[i, 1]
            out1 >> A[i, 0]
            out1 = in1 * 1.5 + 3

    for i in dace.map[0:M]:
        with dace.tasklet:
            in1 << A[i, 1]
            out1 >> A[i, 0]
            out1 = in1 * 3 + 1


# FALSE
@dace.program
def disjoint_test_2(A: dace.float64[M, N]):

    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i, N - 1 - j]
            out1 >> A[i, j]
            out1 = in1 * 2

    for i, j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i, j]
            out1 >> A[i, j]
            out1 = in1 * 1.5 + 3


# FALSE
@dace.program
def disjoint_test_3(A: dace.float64[M]):
    tmp1 = dace.ndarray(shape=[N], dtype=dace.float64)
    for i in dace.map[0:M]:
        with dace.tasklet:
            in1 << A[i]
            out1 >> tmp1[i]
            out1 = in1 * 2

    for i in dace.map[0:M]:
        with dace.tasklet:
            in1 << tmp1[i]
            out1 >> A[M - 1 - i]
            out1 = in1 * 1.5 + 3

    for i in dace.map[0:M]:
        with dace.tasklet:
            in1 << A[M - 1 - i]
            out1 >> A[i]
            out1 = in1 * 3 + 1


def test_p1():
    sdfg = disjoint_test_1.to_sdfg()
    sdfg.simplify()
    state = sdfg.nodes()[0]
    assert len(sdfg.nodes()) == 1
    A = np.random.rand(30, 2).astype(np.float64)
    A1 = A.copy()
    A2 = A.copy()

    csdfg = sdfg.compile()
    csdfg(A=A1, N=20, M=30)
    del csdfg

    subgraph = SubgraphView(state, state.nodes())
    sf = SubgraphFusion()
    sf.setup_match(subgraph)
    assert sf.can_be_applied(sdfg, subgraph)
    sf.apply(sdfg)

    csdfg = sdfg.compile()
    csdfg(A=A2, M=30)
    del csdfg

    assert np.allclose(A1, A2)


def test_p2():
    sdfg = disjoint_test_2.to_sdfg()
    sdfg.simplify()
    state = sdfg.nodes()[0]
    assert len(sdfg.nodes()) == 1

    subgraph = SubgraphView(state, state.nodes())
    sf = SubgraphFusion()
    sf.setup_match(subgraph)
    assert not sf.can_be_applied(sdfg, subgraph)


def test_p3():
    sdfg = disjoint_test_3.to_sdfg()
    sdfg.simplify()
    state = sdfg.nodes()[0]
    assert len(sdfg.nodes()) == 1

    subgraph = SubgraphView(state, state.nodes())
    sf = SubgraphFusion()
    sf.setup_match(subgraph)
    assert not sf.can_be_applied(sdfg, subgraph)


if __name__ == "__main__":
    test_p1()
    test_p2()
    test_p3()
