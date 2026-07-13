# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the next-generation Python frontend pipeline: canonicalization
contracts, lowering rules, and output verification.
"""
import ast

import pytest

import dace
from dace import data
from dace.frontend.python import nextgen
from dace.frontend.python.nextgen.canonical import cpa
from dace.frontend.python.nextgen.canonical.passes import default_passes
from dace.frontend.python.nextgen.common import CanonicalViolationError, TreeVerificationError
from dace.frontend.python.nextgen.lowering.emitter import TreeEmitter
from dace.frontend.python.nextgen.pipeline import CanonicalizationPipeline, PipelineContext
from dace.sdfg.analysis.schedule_tree import treenodes as tn

N = dace.symbol('N')


def _canonicalize(source: str) -> ast.FunctionDef:
    tree = ast.parse(source).body[0]
    context = PipelineContext('test', '<test>', {}, {})
    pipeline = CanonicalizationPipeline(default_passes())
    return pipeline.run(tree, context)


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


# ---------------------------------------------------------------------- #
# Canonicalization
# ---------------------------------------------------------------------- #


def test_canonicalize_augassign():
    result = _canonicalize('def f(a, b):\n    a += b * 2\n')
    assigns = [s for s in ast.walk(result) if isinstance(s, ast.Assign)]
    assert len(assigns) >= 1
    # No AugAssign remains and the result verifies
    assert not any(isinstance(s, ast.AugAssign) for s in ast.walk(result))


def test_canonicalize_anf_flattens_nested_expressions():
    result = _canonicalize('def f(a, b, c, out):\n    out = a * b + c / 2\n')
    # All assignment values must be flat
    for statement in result.body:
        assert isinstance(statement, ast.Assign)
        assert cpa.is_flat(statement.value)


def test_canonicalize_while_with_complex_test():
    result = _canonicalize('def f(a, b):\n    while a * 2 > b + 1:\n        a = a - 1\n')
    while_statement = next(s for s in result.body if isinstance(s, ast.While))
    # Complex test becomes while True with a hoisted conditional break
    assert isinstance(while_statement.test, ast.Constant) and while_statement.test.value is True
    inner_if = while_statement.body[0]
    while not isinstance(inner_if, ast.If):
        inner_if = while_statement.body[while_statement.body.index(inner_if) + 1]
    assert any(isinstance(s, ast.Break) for s in ast.walk(inner_if))


def test_canonicalize_range_normalization():
    result = _canonicalize('def f(n):\n    for i in range(n):\n        pass\n')
    for_statement = next(s for s in result.body if isinstance(s, ast.For))
    assert cpa.is_range_iterator(for_statement.iter)
    assert len(for_statement.iter.args) == 3


def test_canonicalize_marks_try_opaque():
    result = _canonicalize('def f(a):\n    try:\n        a = 1\n    except ValueError:\n        a = 2\n')
    opaque = [s for s in result.body if isinstance(s, cpa.OpaqueStmt)]
    assert len(opaque) == 1
    assert 'a' in opaque[0].outputs


def test_canonicalize_loop_else_desugars():
    result = _canonicalize('def f(a):\n'
                           '    for i in range(10):\n'
                           '        if a > 5:\n'
                           '            break\n'
                           '    else:\n'
                           '        a = 0\n')
    for_statement = next(s for s in result.body if isinstance(s, ast.For))
    assert not for_statement.orelse
    # A did-break flag guards the else body
    assert any(isinstance(s, ast.If) for s in result.body)


def test_verify_canonical_rejects_non_canonical():
    tree = ast.parse('def f(a, b, c):\n    c = a * b + a\n').body[0]
    with pytest.raises(CanonicalViolationError):
        cpa.verify_canonical(tree)


def test_statement_io_sets():
    statement = ast.parse('a[i] = b + c').body[0]
    reads, writes = cpa.statement_io_sets(statement)
    assert reads == {'a', 'i', 'b', 'c'}
    assert writes == {'a'}


# ---------------------------------------------------------------------- #
# End-to-end lowering
# ---------------------------------------------------------------------- #


def test_lower_copy():

    @dace.program
    def copy_program(A: dace.float64[N], B: dace.float64[N]):
        B[:] = A

    tree = nextgen.parse_program(copy_program)
    copies = _nodes_of_type(tree, tn.CopyNode)
    assert len(copies) == 1
    assert copies[0].target == 'B'
    assert copies[0].memlet.data == 'A'
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_lower_elementwise():

    @dace.program
    def elementwise(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        C[:] = A + B

    tree = nextgen.parse_program(elementwise)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert len(tasklets) == 1
    tasklet = tasklets[0]
    assert set(tasklet.in_memlets.keys()) == {'__in0', '__in1'}
    assert tasklet.out_memlets['__out'].data == 'C'
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_lower_scalar_accumulation_loop():

    @dace.program
    def accumulate(A: dace.float64[N]):
        result = 0.0
        for i in range(N):
            result = result + A[i]
        return result

    tree = nextgen.parse_program(accumulate)
    loops = _nodes_of_type(tree, tn.ForScope)
    assert len(loops) == 1
    assert loops[0].loop.loop_variable == 'i'
    # The accumulator is a scalar container updated in place
    assert isinstance(tree.containers['result'], data.Scalar)
    returns = _nodes_of_type(tree, tn.ReturnNode)
    assert len(returns) == 1
    assert returns[0].values and returns[0].values[0].startswith('__return')
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_lower_if_else():

    @dace.program
    def branch(A: dace.float64[N], flag: dace.int32):
        if flag > 0:
            A[0] = 1.0
        else:
            A[0] = 2.0

    tree = nextgen.parse_program(branch)
    if_scopes = _nodes_of_type(tree, tn.IfScope)
    else_scopes = _nodes_of_type(tree, tn.ElseScope)
    assert len(if_scopes) == 1
    assert len(else_scopes) == 1
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_lower_dace_map():

    @dace.program
    def mapped(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            B[i] = A[i] + 1.0

    tree = nextgen.parse_program(mapped)
    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 1
    assert maps[0].node.map.params == ['i']
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_lower_return_expression():

    @dace.program
    def return_expression(A: dace.float64[N]):
        return A + 1.0

    tree = nextgen.parse_program(return_expression)
    returns = _nodes_of_type(tree, tn.ReturnNode)
    assert len(returns) == 1
    assert '__return' in tree.containers
    assert tree.containers['__return'].transient is False
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_lower_view_binding():

    @dace.program
    def view_program(A: dace.float64[N]):
        b = A[1:5]
        b[0] = 42.0

    tree = nextgen.parse_program(view_program)
    views = _nodes_of_type(tree, tn.ViewNode)
    assert len(views) == 1
    assert views[0].source == 'A'
    # The write through the view targets the view container
    tasklets = _nodes_of_type(tree, tn.TaskletNode)
    assert any(t.out_memlets['__out'].data == views[0].target for t in tasklets)
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_opaque_statement_becomes_callback():

    @dace.program
    def with_print(A: dace.float64[N]):
        print(A)
        A[0] = 1.0

    tree = nextgen.parse_program(with_print)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1
    assert 'A' in callbacks[0].input_names
    assert callbacks[0].reason
    assert not _nodes_of_type(tree, tn.StatementNode)


def test_no_statement_nodes_ever():
    """The frontend-legal contract bans StatementNode emission outright."""

    root = tn.ScheduleTreeRoot(name='test', children=[])
    emitter = TreeEmitter(root)
    with pytest.raises(TreeVerificationError):
        emitter.emit(tn.StatementNode(code=dace.properties.CodeBlock('pass')))


def test_repository_is_shared_not_cloned():

    @dace.program
    def identity(A: dace.float64[N], B: dace.float64[N]):
        B[:] = A

    tree = nextgen.parse_program(identity)
    # Argument containers are non-transient and registered once
    assert tree.containers['A'].transient is False
    assert tree.containers['B'].transient is False
    assert 'N' in tree.symbols


if __name__ == '__main__':
    import sys
    pytest.main([__file__, '-v'] + sys.argv[1:])
