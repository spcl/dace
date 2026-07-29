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


if __name__ == '__main__':
    test_attribute_of_a_computed_value()
    test_method_call_on_a_computed_receiver()
    test_doubly_nested_index()
    test_indirect_range_bound_is_not_hoisted()
