# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for ``@dace.mapscope``-decorated function desugaring in the
next-generation Python frontend's canonicalization pass. ``@dace.mapscope``
is the classic explicit-dataflow scope syntax:

    @dace.mapscope(_[0:N])
    def rows(i):
        <arbitrary statements using i>

which must desugar to the canonical equivalent of ``for i in dace.map[0:N]:``
with the body statements recursively canonicalized (so nested
``@dace.map``/``@dace.tasklet``/``@dace.mapscope`` functions inside the body
desugar too).
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')
M = dace.symbol('M')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_mapscope_structure():

    @dace.program
    def mapscope_prog(A: dace.float64[N, M], B: dace.float64[N, M]):

        @dace.mapscope(_[0:N])
        def rows(i):

            @dace.map(_[0:M])
            def cols(j):
                a << A[i, j]
                b >> B[i, j]
                b = a * 2.0

    tree = nextgen.parse_program(mapscope_prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 2
    outer = maps[0]
    assert outer.node.map.params == ['i']
    assert [(str(r[0]), str(r[1])) for r in outer.node.map.range] == [('0', 'N - 1')]

    inner = maps[1]
    assert inner.node.map.params == ['j']

    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert tasklets[0].in_memlets['a'].data == 'A'
    assert tasklets[0].out_memlets['b'].data == 'B'


def test_mapscope_execution():

    @dace.program
    def mapscope_prog(A: dace.float64[N, M], B: dace.float64[N, M]):

        @dace.mapscope(_[0:N])
        def rows(i):

            @dace.map(_[0:M])
            def cols(j):
                a << A[i, j]
                b >> B[i, j]
                b = a * 2.0

    tree = nextgen.parse_program(mapscope_prog)
    sdfg = tree.as_sdfg()
    func = sdfg.compile()

    n, m = 8, 6
    A = np.random.rand(n, m)
    B = np.zeros((n, m))
    func(A=A, B=B, N=n, M=m)
    assert np.allclose(B, A * 2.0)


def test_mapscope_multidim():

    @dace.program
    def mapscope_prog(A: dace.float64[N, M], B: dace.float64[N, M]):

        @dace.mapscope(_[0:N, 0:M])
        def block(i, j):

            @dace.tasklet
            def compute():
                a << A[i, j]
                b >> B[i, j]
                b = a * 2.0

    tree = nextgen.parse_program(mapscope_prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert maps[0].node.map.params == ['i', 'j']

    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1

    sdfg = tree.as_sdfg()
    func = sdfg.compile()
    n, m = 5, 4
    A = np.random.rand(n, m)
    B = np.zeros((n, m))
    func(A=A, B=B, N=n, M=m)
    assert np.allclose(B, A * 2.0)


def test_mapscope_statements_body():

    @dace.program
    def mapscope_prog(A: dace.float64[N, M], B: dace.float64[N, M]):

        @dace.mapscope(_[0:N])
        def rows(i):
            B[i, :] = A[i, :] * 2.0

    tree = nextgen.parse_program(mapscope_prog)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    # The mapscope produces the outer map; the elementwise slice assignment
    # in its body ('B[i, :] = A[i, :] * 2.0') lowers to its own (inner) map,
    # so at least the outer mapscope map must be present.
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) >= 1
    assert maps[0].node.map.params == ['i']

    sdfg = tree.as_sdfg()
    func = sdfg.compile()
    n, m = 6, 5
    A = np.random.rand(n, m)
    B = np.zeros((n, m))
    func(A=A, B=B, N=n, M=m)
    assert np.allclose(B, A * 2.0)


def test_consumescope_fallback():

    @dace.program
    def fibonacci(iv: dace.int32[1], res: dace.float32[1]):
        S = dace.define_stream(dace.int32, 0)
        with dace.tasklet:
            i << iv
            s >> S
            s = i

        @dace.consume(S, 4)
        def scope(elem, p):
            sout >> S(-1)
            val >> res(-1, lambda a, b: a + b)
            if elem == 1:
                val = 1
            elif elem > 1:
                sout = elem - 1
                sout = elem - 2

    # dace.consume/dace.consumescope are not yet recognized as explicit
    # dataflow; they must fall through to the interpreter-callback path
    # without crashing lowering.
    tree = nextgen.parse_program(fibonacci)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) >= 1


if __name__ == '__main__':
    test_mapscope_structure()
    test_mapscope_execution()
    test_mapscope_multidim()
    test_mapscope_statements_body()
    test_consumescope_fallback()
