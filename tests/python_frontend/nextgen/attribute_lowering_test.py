# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for registry ATTRIBUTE-family reads (``A.T``, ``.real``, ``.imag``,
``.flat``) in expression positions other than a whole right-hand side bound to
a name.

Such a read computes data, so it has to be materialized into a container before
anything lowers it. The elementwise mechanism has no resolution for an
attribute and would otherwise substitute the access verbatim into the generated
tasklet (``__out = __in0.T``), so these tests check EXECUTION, not just the
absence of a callback -- a callback-free lowering of the wrong program is
exactly what this path used to produce.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_attribute_into_a_subscript_target():
    """``c[:, :] = a.T`` — the target is not a name, so the dedicated
    name-binding path does not apply."""

    @dace.program
    def transposed(a: dace.float64[4, 4], c: dace.float64[4, 4]):
        c[:, :] = a.T

    a, c = np.random.rand(4, 4), np.zeros((4, 4))
    tree = nextgen.parse_program(transposed, a, c)
    assert not _callbacks(tree)
    tree.as_sdfg().compile()(a=a, c=c)
    assert np.allclose(c, a.T)


def test_attribute_as_one_operand():
    """``c[:, :] = a.T + b`` — the read is one operand of a larger expression."""

    @dace.program
    def shifted(a: dace.float64[4, 4], b: dace.float64[4, 4], c: dace.float64[4, 4]):
        c[:, :] = a.T + b

    a, b, c = np.random.rand(4, 4), np.random.rand(4, 4), np.zeros((4, 4))
    tree = nextgen.parse_program(shifted, a, b, c)
    assert not _callbacks(tree)
    tree.as_sdfg().compile()(a=a, b=b, c=c)
    assert np.allclose(c, a.T + b)


def test_attribute_inside_a_dataflow_scope_degrades():
    """
    Materializing the attribute means emitting a deferred replacement call,
    which is not a legal node inside a map scope. The statement becomes a
    callback there rather than generated code containing the attribute access.
    """

    @dace.program
    def in_scope(a: dace.float64[4, 4], c: dace.float64[4]):
        for i in dace.map[0:4]:
            c[i] = a.T[i, 0]

    tree = nextgen.parse_program(in_scope, np.random.rand(4, 4), np.zeros(4))
    assert len(_callbacks(tree)) == 1


if __name__ == '__main__':
    test_attribute_into_a_subscript_target()
    test_attribute_as_one_operand()
    test_attribute_inside_a_dataflow_scope_degrades()
