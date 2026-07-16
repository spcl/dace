# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for explicit-dataflow lowering in the next-generation Python frontend:
``with dace.tasklet:`` blocks, ``@dace.tasklet`` functions, ``@dace.map``
functions, and the ``<<``/``>>`` memlet syntax.
"""
import ast

import dace
from dace import dtypes
from dace.frontend.python import nextgen
from dace.frontend.python.nextgen.canonical import cpa
from dace.frontend.python.nextgen.canonical.passes import default_passes
from dace.frontend.python.nextgen.pipeline import CanonicalizationPipeline, PipelineContext
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def test_with_tasklet_in_map():

    @dace.program
    def tasklet_in_map(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                b = a * 2.0
                b >> B[i]

    tree = nextgen.parse_program(tasklet_in_map)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    tasklet = tasklets[0]
    assert set(tasklet.in_memlets.keys()) == {'a'}
    assert set(tasklet.out_memlets.keys()) == {'b'}
    assert tasklet.in_memlets['a'].data == 'A'
    assert tasklet.out_memlets['b'].data == 'B'
    assert 'a * 2.0' in tasklet.node.code.as_string
    # Memlet statements are stripped from the tasklet code
    assert '<<' not in tasklet.node.code.as_string
    assert '>>' not in tasklet.node.code.as_string
    assert not _nodes_of_type(tree, tn.StatementNode)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_tasklet_wcr_output():

    @dace.program
    def wcr_tasklet(A: dace.float64[N], B: dace.float64[1]):
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << A[i]
                out = a
                out >> B(1, lambda x, y: x + y)[0]

    tree = nextgen.parse_program(wcr_tasklet)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    out_memlet = tasklets[0].out_memlets['out']
    assert out_memlet.data == 'B'
    assert out_memlet.wcr is not None
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_tasklet_decorator():

    @dace.program
    def decorated(A: dace.float64[N], B: dace.float64[N]):

        @dace.tasklet
        def compute():
            a << A[0]
            b = a + 1.0
            b >> B[0]

    tree = nextgen.parse_program(decorated)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert tasklets[0].node.label == 'compute'
    assert tasklets[0].in_memlets['a'].data == 'A'
    assert tasklets[0].out_memlets['b'].data == 'B'
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_map_decorator():

    @dace.program
    def mapped(A: dace.float64[N], B: dace.float64[N]):

        @dace.map(_[0:N])
        def scale(i):
            a << A[i]
            b = a * 3.0
            b >> B[i]

    tree = nextgen.parse_program(mapped)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert maps[0].node.map.params == ['i']
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert tasklets[0].in_memlets['a'].data == 'A'
    assert tasklets[0].out_memlets['b'].data == 'B'
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)


def test_tasklet_cpp_intrinsic():

    @dace.program
    def cpp_tasklet(A: dace.float64[N], B: dace.float64[N]):
        with dace.tasklet:
            a << A[0]
            'b = a * 2;'
            b >> B[0]

    tree = nextgen.parse_program(cpp_tasklet)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert tasklets[0].node.language == dtypes.Language.CPP
    assert 'b = a * 2;' in tasklets[0].node.code.as_string


def test_tasklet_language_argument():

    @dace.program
    def explicit_language(A: dace.float64[N], B: dace.float64[N]):
        with dace.tasklet(language='CPP'):
            a << A[0]
            b = a
            b >> B[0]

    tree = nextgen.parse_program(explicit_language)
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    assert tasklets[0].node.language == dtypes.Language.CPP


def test_bare_call_falls_back_to_callback():

    @dace.program
    def bare_call(A: dace.float64[N]):
        print(A[0])

    tree = nextgen.parse_program(bare_call)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    assert callbacks[0].reason
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_local_name_tasklet_not_hijacked():
    # A name that merely spells "tasklet" without resolving to dace.tasklet
    # must not be recognized as explicit dataflow during canonicalization.
    source = ('def f(A):\n'
              '    with tasklet:\n'
              '        A[0] = 1.0\n')
    tree = ast.parse(source).body[0]
    context = PipelineContext('test', '<test>', {'dace': dace}, {})
    result = CanonicalizationPipeline(default_passes()).run(tree, context)
    assert not any(isinstance(s, cpa.ExplicitTasklet) for s in ast.walk(result))
    assert any(isinstance(s, cpa.OpaqueStmt) for s in ast.walk(result))


def test_resolved_bare_tasklet_recognized():
    # `from dace import tasklet` style: the bare name resolves to the builtin.
    source = ('def f(A):\n'
              '    with tasklet:\n'
              '        a << A[0]\n'
              '        b = a\n'
              '        b >> A[1]\n')
    tree = ast.parse(source).body[0]
    context = PipelineContext('test', '<test>', {'tasklet': dace.tasklet}, {})
    result = CanonicalizationPipeline(default_passes()).run(tree, context)
    assert any(isinstance(s, cpa.ExplicitTasklet) for s in ast.walk(result))


if __name__ == '__main__':
    test_with_tasklet_in_map()
    test_tasklet_wcr_output()
    test_tasklet_decorator()
    test_map_decorator()
    test_tasklet_cpp_intrinsic()
    test_tasklet_language_argument()
    test_bare_call_falls_back_to_callback()
    test_local_name_tasklet_not_hijacked()
    test_resolved_bare_tasklet_recognized()
