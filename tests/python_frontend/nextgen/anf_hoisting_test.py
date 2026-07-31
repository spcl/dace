# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for A-normal form hoisting of compound subexpressions the canonical
grammar bounds: a computed base under an attribute or a method call, and
subscript nesting deeper than one level of indirection.

The CPA grammar admits an attribute over a name and one nested subscript in an
index; anything deeper is expected to arrive already hoisted into a temporary.
These pin that the ANF pass actually performs the hoisting the grammar's bounds
assume, rather than leaving the statement to the interpreter fallback.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_attribute_of_a_computed_value():
    """``(a @ b).T`` — the attribute's base is an expression (Issue #1295)."""

    @dace.program
    def transposed(a: dace.float64[20, 20], b: dace.float64[20, 20], c: dace.float64[20, 20]):
        c[:, :] = (a @ b).T

    a, b = np.random.rand(20, 20), np.random.rand(20, 20)
    c = np.zeros((20, 20))
    tree = nextgen.parse_program(transposed, a, b, c)
    assert not _callbacks(tree)
    tree.as_sdfg().compile()(a=a, b=b, c=c)
    assert np.allclose(c, (a @ b).T)


def test_method_call_on_a_computed_receiver():
    """``numpy.arange(10).reshape(10, 1)`` — the receiver is itself a call."""

    @dace.program
    def chained():
        return np.arange(10).reshape(10, 1)

    tree = nextgen.parse_program(chained)
    assert not _callbacks(tree)
    assert np.allclose(chained(), np.arange(10).reshape(10, 1))


def test_doubly_nested_index():
    """``A[x[x[i]]]`` — one level deeper than an index admits."""

    @dace.program
    def indirect(A: dace.float64[20], x: dace.int64[20], i: dace.int64):
        return A[x[x[i]]]

    A = np.random.rand(20)
    x = np.random.randint(0, 20, size=20).astype(np.int64)
    tree = nextgen.parse_program(indirect, A, x, np.int64(0))
    assert not _callbacks(tree)
    for i in (0, 3, 7):
        assert np.allclose(indirect(A, x, i), A[x[x[i]]])


def test_indirect_range_bound_is_not_hoisted():
    """
    The one level of indirection an index DOES admit must stay in place:
    ``A[ptr[i]:ptr[i + 1]]`` is the dynamic-map-range pattern, and hoisting its
    bounds would turn a map range into a data read.
    """

    @dace.program
    def segments(A: dace.float64[20], ptr: dace.int64[3], out: dace.float64[2]):
        for i in dace.map[0:2]:
            out[i] = np.sum(A[ptr[i]:ptr[i + 1]])

    A = np.random.rand(20)
    ptr = np.array([0, 8, 20], dtype=np.int64)
    out = np.zeros(2)
    tree = nextgen.parse_program(segments, A, ptr, out)
    assert not _callbacks(tree)
    assert any(isinstance(node, tn.MapScope) for node in tree.preorder_traversal())


def _helper(value):
    return value * 2.0


def _undefined_temporaries(tree) -> set:
    """
    Names a callback reads that no statement in the tree ever assigns, and that
    are not program containers either -- i.e. temporaries left behind by a
    rewrite whose defining assignment was dropped.
    """
    assigned = set(tree.containers)
    for node in tree.preorder_traversal():
        assigned.update(getattr(node, 'outputs', None) or ())
    reads = set()
    for node in _callbacks(tree):
        reads.update(getattr(node, 'inputs', None) or ())
    return {name for name in reads - assigned if name.startswith('__anf')}


def test_abandoned_flattening_leaves_no_dangling_temporary():
    """
    A statement ANF starts flattening and then abandons (a short-circuit
    position it cannot hoist through) must be handed to the interpreter
    EXACTLY as written.

    Flattening rewrites subexpressions in place while collecting the hoisted
    assignments; abandoning it drops those assignments, so the rewrites must be
    undone with them. Left in, the callback body read a ``__anf`` temporary
    that nothing ever assigned.
    """

    @dace.program
    def short_circuit(A: dace.float64[10], c: dace.bool):
        b = 1.0
        d = 2.0
        A[1] = _helper(A[0]) + (b if c else d)

    tree = nextgen.parse_program(short_circuit)
    callbacks = _callbacks(tree)
    assert callbacks, 'the conditional expression is expected to stay opaque'
    assert not any('__anf' in node.reason for node in callbacks), [node.reason for node in callbacks]
    assert not _undefined_temporaries(tree)


def test_abandoned_target_flattening_leaves_no_dangling_temporary():
    """
    The same, abandoned on the assignment TARGET rather than the value.

    A chained target hoists its base first (``__anf0 = A[:, 1:5]``) and only
    then flattens the index, so a hazard in the index leaves the base rewrite
    behind unless it is rolled back.
    """

    @dace.program
    def short_circuit_target(A: dace.float64[10, 10], c: dace.bool):
        A[:, 1:5][1 if c else 2] = 5.0

    tree = nextgen.parse_program(short_circuit_target)
    callbacks = _callbacks(tree)
    assert callbacks, 'the conditional index is expected to stay opaque'
    assert not any('__anf' in node.reason for node in callbacks), [node.reason for node in callbacks]
    assert not _undefined_temporaries(tree)


if __name__ == '__main__':
    test_attribute_of_a_computed_value()
    test_method_call_on_a_computed_receiver()
    test_doubly_nested_index()
    test_indirect_range_bound_is_not_hoisted()
    test_abandoned_flattening_leaves_no_dangling_temporary()
    test_abandoned_target_flattening_leaves_no_dangling_temporary()
