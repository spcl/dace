# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for subscripts whose bounds canonicalization hoisted into a temporary.

``O[i * 96:(i + 1) * 96]`` becomes ``__anf0 = i + 1; O[i * 96:__anf0 * 96]``,
and that temporary used to be materialized as a scalar CONTAINER. In a subset a
container name is meaningless: inside a dataflow scope it reads as a
data-dependent index and was refused outright, and even outside one it kept
sizes like ``__anf0 * 96 - i * 96`` from folding.

A quantity set once and consumed in a range is a SYMBOL
(``doc/extensions/symbolic.rst``), so such a temporary binds as one
(``rules.assign._bind_index_symbol``): a compile-time symbolic value IS the
expression and nothing is emitted, while a value read from data defines a real
symbol through an interstate assignment.

Also here because it shares the shape: a full reduction assigned to an array
subset broadcasts, which is what NumPy does with ``out[a:b] = numpy.sum(x)``.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def test_map_dependent_slice_target():
    """A slice whose bounds are an expression over the map parameter: the
    hoisted bound stays symbolic, so no container is materialized for it."""

    @dace.program
    def tiled(A: dace.float64[4], O: dace.float64[8]):
        for i in dace.map[0:2]:
            O[i * 4:(i + 1) * 4] = A + i

    A, O = np.arange(4, dtype=np.float64), np.zeros(8)
    tree = nextgen.parse_program(tiled, A, O)
    assert not _callbacks(tree)
    assert not [name for name in tree.containers if name.startswith('__anf')]

    tree.as_sdfg().compile()(A=A, O=O)
    assert np.allclose(O, np.concatenate([A, A + 1]))


def test_map_dependent_slice_source():
    """The same, reading through a hoisted bound.

    Asserted on the TREE only: the view this binds inside the map body then
    hits a tree-to-SDFG gap (a view bound inside a scope needs its source
    imported into the nested SDFG the body becomes, and the passthrough memlet
    carries the source's size rather than the view's), recorded in the plan.
    """

    @dace.program
    def gather(A: dace.float64[8], O: dace.float64[2]):
        for i in dace.map[0:2]:
            O[i] = np.sum(A[i * 4:(i + 1) * 4])

    tree = nextgen.parse_program(gather, np.zeros(8), np.zeros(2))
    assert not _callbacks(tree)
    views = [node for node in tree.preorder_traversal() if isinstance(node, tn.ViewNode)]
    assert len(views) == 1
    assert str(views[0].memlet.subset) == '4*i:4*i + 4'


def test_full_reduction_broadcasts_into_a_slice():
    """``out[a:b] = numpy.sum(x)``: one value written to every element of the
    target subset, as NumPy broadcasts it."""

    @dace.program
    def filled(A: dace.float64[8], O: dace.float64[8]):
        for i in dace.map[0:2]:
            O[i * 4:i * 4 + 4] = np.sum(A)

    tree = nextgen.parse_program(filled, np.zeros(8), np.zeros(8))
    assert not _callbacks(tree)
    # Reduce into a scalar temporary, then fill the target subset from it.
    scopes = [node for node in tree.preorder_traversal() if isinstance(node, tn.MapScope)]
    assert len(scopes) == 3  # the program's own map, the reduction, the broadcast
    # Asserted on the TREE only: inside a map body the scalar temporary hits
    # the recorded tree-to-SDFG gap where a body-local transient is also given
    # a nested-SDFG connector ("is a connector but its corresponding array is
    # transient").


def test_range_length_from_data_becomes_a_symbol():
    """A scalar that determines a subset's LENGTH is promoted to a symbol,
    defined by an interstate assignment -- the same mechanism dynamic map
    bounds use. Left as a container it printed as data where the code
    generator expects a symbol, and inside a dataflow scope it was refused
    outright as a data-dependent subscript."""

    @dace.program
    def fill_bounded(bounds: dace.int64[2], O: dace.float64[10]):
        n = bounds[0]
        O[0:n] = 1.0

    bounds, O = np.array([4, 7]), np.zeros(10)
    tree = nextgen.parse_program(fill_bounded, bounds, O)
    assert not _callbacks(tree)
    assigns = [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    assert len(assigns) == 1 and assigns[0].name.startswith('__idx_n')

    tree.as_sdfg().compile()(bounds=bounds, O=O)
    assert np.allclose(O[:4], 1.0) and np.allclose(O[4:], 0.0)


def test_data_length_inside_a_map_scope():
    """The same inside a map scope, where a scalar in a subset used to be
    refused outright as a data-dependent subscript."""

    @dace.program
    def bounded_rows(b: dace.int64[4], O: dace.float64[4, 20]):
        for i in dace.map[0:4]:
            n = b[i] + 1
            O[i, 0:n] = 1.0

    b, O = np.array([0, 1, 2, 3]), np.zeros((4, 20))
    tree = nextgen.parse_program(bounded_rows, b, O)
    assert not _callbacks(tree)
    assert [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]

    tree.as_sdfg().compile()(b=b, O=O)
    assert all(row[:count + 1].all() and not row[count + 1:].any() for count, row in zip(b, O))


if __name__ == '__main__':
    test_map_dependent_slice_target()
    test_map_dependent_slice_source()
    test_full_reduction_broadcasts_into_a_slice()
    test_range_length_from_data_becomes_a_symbol()
    test_data_length_inside_a_map_scope()
