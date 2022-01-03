# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.transformation.subgraph.helpers as helpers
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView

import sys

from dace.transformation.subgraph import SubgraphFusion
from util import expand_maps, expand_reduce, fusion

N, M, O = [dace.symbol(s) for s in ['N', 'M', 'O']]
N.set(50)
M.set(60)
O.set(70)


@dace.program
def expansion1(A: dace.float64[M, N, O], B: dace.float64[M, N, O], C: dace.float64[M, N, O], out1: dace.float64[M, N,
                                                                                                                O]):

    tmp1 = np.ndarray([M, N, O], dtype=dace.float64)
    tmp2 = np.ndarray([M, N, O], dtype=dace.float64)

    for i, j, k in dace.map[0:M, 0:N, 0:O]:
        with dace.tasklet:
            in1 << A[i, j, k]
            in2 << B[i, j, k]
            out >> tmp1[i, j, k]

            out = in1 * 3 + in2

    for j, k, i in dace.map[0:M, 0:N, 0:O]:
        with dace.tasklet:
            in1 << A[j, k, i]
            in2 << tmp1[j, k, i]
            out >> out1[j, k, i]

            out = in1 + in2 + 42


@dace.program
def expansion2(A: dace.float64[M, N], B: dace.float64[M, O], out1: dace.float64[M, N], out2: dace.float64[M, O]):

    tmp1 = np.ndarray([M, N], dtype=dace.float64)
    tmp2 = np.ndarray([M, O], dtype=dace.float64)

    for i, p in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i, p]
            out >> out1[i, p]

            out = in1 * 2 + 42

    for p, i in dace.map[0:M, 0:O]:
        with dace.tasklet:
            in1 << B[p, i]
            out >> out2[p, i]

            out = in1 * 3 + 31


def run(sdfg, graph, kwargs):
    csdfg = sdfg.compile()
    csdfg(**kwargs)
    del csdfg


def test_expansion2():
    sdfg = expansion2.to_sdfg()
    graph = sdfg.nodes()[0]
    kwargs = {
        'A': np.random.rand(M.get(), N.get()).astype(np.float64),
        'B': np.random.rand(M.get(), O.get()).astype(np.float64),
        'out1': np.ndarray((M.get(), N.get()), dtype=np.float64),
        'out2': np.ndarray((M.get(), O.get()), dtype=np.float64),
        'N': N,
        'M': M,
        'O': O
    }

    run(sdfg, graph, kwargs)
    out1 = kwargs['out1'].copy()
    out2 = kwargs['out2'].copy()

    kwargs['out1'].fill(0)
    kwargs['out2'].fill(0)

    expand_maps(sdfg, graph)
    run(sdfg, graph, kwargs)
    out3 = kwargs['out1'].copy()
    out4 = kwargs['out2'].copy()
    assert np.linalg.norm(out1) > 0.01
    assert np.allclose(out1, out3)
    assert np.allclose(out2, out4)


def test_expansion1():
    sdfg = expansion1.to_sdfg()
    graph = sdfg.nodes()[0]
    kwargs = {
        'A': np.random.rand(M.get(), N.get(), O.get()).astype(np.float64),
        'B': np.random.rand(M.get(), N.get(), O.get()).astype(np.float64),
        'C': np.random.rand(M.get(), N.get(), O.get()).astype(np.float64),
        'out1': np.ndarray((M.get(), N.get(), O.get()), dtype=np.float64),
        'N': N,
        'M': M,
        'O': O
    }
    run(sdfg, graph, kwargs)
    out1 = kwargs['out1'].copy()
    kwargs['out1'].fill(0)
    expand_maps(sdfg, graph)
    run(sdfg, graph, kwargs)
    out2 = kwargs['out1'].copy()

    assert np.linalg.norm(out1) > 0.01
    assert np.allclose(out1, out2)


if __name__ == "__main__":
    test_expansion1()
    test_expansion2()
