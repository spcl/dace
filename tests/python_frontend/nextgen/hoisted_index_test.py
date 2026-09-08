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

The same rule decides what happens to a data read that appears in an index
position directly (``lowering.access.promote_index_reads``), and the index FORM
decides it: a slice BOUND is a range bound and becomes a symbol, while an
element index is a data read and stays indirection -- except where the
subscript also keeps a range, which indirection cannot express at all.

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
    # Reduce into a scalar temporary, then fill the target subset from it. The
    # reduction itself is a deferred replacement (a Reduce library node), so
    # only the program's own map and the broadcast are map scopes here.
    scopes = [node for node in tree.preorder_traversal() if isinstance(node, tn.MapScope)]
    assert len(scopes) == 2
    reductions = [node for node in tree.preorder_traversal() if isinstance(node, tn.ReplacementCallNode)]
    assert [node.qualname for node in reductions] == ['numpy.sum']
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


def test_read_through_a_data_bound_is_not_indirection():
    """The READ side of ``O[0:n] = A[0:n] + 1.0``.

    A scalar in a slice bound used to be collected as an indirect index read,
    which put the whole source behind a pointer connector and left the slice in
    the tasklet code (``__out = __in0[0:__in1] + 1.0``). The code generator
    drops such a subscript entirely and emits ``__in0_0 + 1.0`` -- pointer
    arithmetic where an element was meant, which nothing before the C++
    compiler rejected."""

    @dace.program
    def bounded_offset(bounds: dace.int64[2], A: dace.float64[10], O: dace.float64[10]):
        n = bounds[0]
        O[0:n] = A[0:n] + 1.0

    bounds, A, O = np.array([4, 7]), np.arange(10, dtype=np.float64), np.zeros(10)
    tree = nextgen.parse_program(bounded_offset, bounds, A, O)
    assert not _callbacks(tree)
    # One symbol for the whole statement: the write and the read subsets have
    # to agree on an extent, not merely hold equal values.
    assigns = [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    assert len(assigns) == 1
    symbol = assigns[0].name
    tasklets = [node for node in tree.preorder_traversal() if isinstance(node, tn.TaskletNode)]
    read = tasklets[-1]
    assert set(read.in_memlets) == {'__in0'} and read.in_memlets['__in0'].data == 'A'
    assert str(read.in_memlets['__in0'].subset) == '__i0'  # elementwise, not a whole-array pointer
    maps = [node for node in tree.preorder_traversal() if isinstance(node, tn.MapScope)]
    assert str(maps[0].node.map.range) == f'0:{symbol}'

    tree.as_sdfg().compile()(bounds=bounds, A=A, O=O)
    assert np.allclose(O[:4], A[:4] + 1.0) and np.allclose(O[4:], 0.0)


def test_single_element_slice_is_still_a_range():
    """``A[n:n + 1]`` spans one element but is slice-formed, so it is a range.

    Deciding by extent instead of by form sent it down the indirection path,
    where a slice cannot be expressed."""

    @dace.program
    def one_wide_slice(bounds: dace.int64[2], A: dace.float64[10], O: dace.float64[10]):
        n = bounds[0]
        O[n:n + 1] = A[n:n + 1] + 1.0

    bounds, A, O = np.array([4, 7]), np.arange(10, dtype=np.float64), np.zeros(10)
    tree = nextgen.parse_program(one_wide_slice, bounds, A, O)
    assert not _callbacks(tree)
    tasklet = [node for node in tree.preorder_traversal() if isinstance(node, tn.TaskletNode)][-1]
    assert str(tasklet.in_memlets['__in0'].subset).startswith('__idx_n')

    tree.as_sdfg().compile()(bounds=bounds, A=A, O=O)
    assert O[4] == A[4] + 1.0 and not O[:4].any() and not O[5:].any()


def test_range_bound_read_from_an_array():
    """A bound read straight from an array (``O[i, 0:ends[i]]``), with no
    scalar temporary in between."""

    @dace.program
    def data_bounded_rows(ends: dace.int64[4], A: dace.float64[4, 20], O: dace.float64[4, 20]):
        for i in dace.map[0:4]:
            O[i, 0:ends[i]] = A[i, 0:ends[i]] + 1.0

    ends = np.array([2, 4, 6, 8])
    A, O = np.arange(80, dtype=np.float64).reshape(4, 20).copy(), np.zeros((4, 20))
    tree = nextgen.parse_program(data_bounded_rows, ends, A, O)
    assert not _callbacks(tree)
    assigns = [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    assert len(assigns) == 1 and assigns[0].name.startswith('__idx_ends')

    tree.as_sdfg().compile()(ends=ends, A=A, O=O)
    for row, count in enumerate(ends):
        assert np.allclose(O[row, :count], A[row, :count] + 1.0) and not O[row, count:].any()


def test_a_bound_read_through_a_scalar_index():
    """``O[0:ends[k]]`` with a scalar ``k``: the symbol is defined from
    ``ends[k]`` on an interstate edge, which can name symbols but not data, so
    ``k`` has to be promoted first -- the definitions chain."""

    @dace.program
    def chained_bound(idx: dace.int64[2], ends: dace.int64[4], A: dace.float64[20], O: dace.float64[20]):
        k = idx[0]
        O[0:ends[k]] = A[0:ends[k]] + 1.0

    idx, ends = np.array([1, 0]), np.array([2, 4, 6, 8])
    A, O = np.arange(20, dtype=np.float64), np.zeros(20)
    tree = nextgen.parse_program(chained_bound, idx, ends, A, O)
    assert not _callbacks(tree)
    assigns = [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    assert len(assigns) == 2
    # A scalar is named on its own -- it generates as a value, not a pointer.
    assert assigns[0].name.startswith('__idx_k') and assigns[0].value.as_string == 'k'
    assert assigns[1].value.as_string == f'ends[{assigns[0].name}]'

    tree.as_sdfg().compile()(idx=idx, ends=ends, A=A, O=O)
    assert np.allclose(O[:4], A[:4] + 1.0) and not O[4:].any()


def test_a_step_read_from_data():
    """A slice's STEP is a range bound too, not just its endpoints."""

    @dace.program
    def data_strided(bounds: dace.int64[2], A: dace.float64[20], O: dace.float64[20]):
        s = bounds[0]
        O[0:20:s] = A[0:20:s] + 1.0

    bounds, A, O = np.array([2, 7]), np.arange(20, dtype=np.float64), np.zeros(20)
    tree = nextgen.parse_program(data_strided, bounds, A, O)
    assert not _callbacks(tree)

    tree.as_sdfg().compile()(bounds=bounds, A=A, O=O)
    expected = np.zeros(20)
    expected[0:20:2] = A[0:20:2] + 1.0
    assert np.allclose(O, expected)


def test_element_index_beside_a_slice_becomes_a_symbol():
    """``A[col[i], 0:5]``: an element index, but the subscript keeps a range.

    Indirection lowers a whole-ELEMENT read, so it cannot express this shape --
    it reached SDFG construction as ``__in0[__in1, 0:5]`` and failed there. The
    row index becomes a symbol instead, leaving an ordinary strided copy."""

    @dace.program
    def symbolic_row_gather(col: dace.int64[4], A: dace.float64[10, 5], O: dace.float64[4, 5]):
        for i in dace.map[0:4]:
            O[i, 0:5] = A[col[i], 0:5]

    col = np.array([0, 2, 4, 6])
    A, O = np.arange(50, dtype=np.float64).reshape(10, 5).copy(), np.zeros((4, 5))
    tree = nextgen.parse_program(symbolic_row_gather, col, A, O)
    assert not _callbacks(tree)
    assigns = [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    assert len(assigns) == 1 and assigns[0].name.startswith('__idx_col')

    tree.as_sdfg().compile()(col=col, A=A, O=O)
    assert np.allclose(O, A[col])


def test_unindexed_dimensions_count_as_a_range():
    """``G[k, E, neigh[a, b]]`` on a 5-D ``G``: no slice is written, but the
    two unindexed dimensions stay whole, so the subscript keeps a range and
    indirection cannot express it (the shape sse uses).

    Judging by written slices alone left this on the indirection path, where
    the trailing dimensions have nowhere to go, and it fell back instead."""

    @dace.program
    def contract(neigh: dace.int32[2, 2], G: dace.float64[4, 3, 2], O: dace.float64[2, 2, 2]):
        for a, b in dace.map[0:2, 0:2]:
            O[a, b] = G[a + 1, neigh[a, b]]

    neigh = np.array([[0, 2], [1, 0]], dtype=np.int32)
    G, O = np.arange(24, dtype=np.float64).reshape(4, 3, 2).copy(), np.zeros((2, 2, 2))
    tree = nextgen.parse_program(contract, neigh, G, O)
    assert not _callbacks(tree)
    assigns = [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    assert len(assigns) == 1 and assigns[0].name.startswith('__idx_neigh')

    tree.as_sdfg().compile()(neigh=neigh, G=G, O=O)
    for a in range(2):
        for b in range(2):
            assert np.allclose(O[a, b], G[a + 1, neigh[a, b]])


def test_element_index_alone_stays_indirection():
    """The complement of the rule: with no range in the subscript, a
    data-dependent element index is genuine indirection and keeps lowering as
    a full-array connector plus an index connector."""

    @dace.program
    def indirect_gather(idx: dace.int64[4], A: dace.float64[10], O: dace.float64[4]):
        for i in dace.map[0:4]:
            O[i] = A[idx[i]] + 1.0

    idx, A, O = np.array([1, 3, 5, 7]), np.arange(10, dtype=np.float64), np.zeros(4)
    tree = nextgen.parse_program(indirect_gather, idx, A, O)
    assert not _callbacks(tree)
    assert not [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    tasklet = [node for node in tree.preorder_traversal() if isinstance(node, tn.TaskletNode)][-1]
    assert str(tasklet.in_memlets['__in0'].subset) == '0:10'  # the whole array, through a pointer
    assert '__in0[__in1]' in tasklet.node.code.as_string

    tree.as_sdfg().compile()(idx=idx, A=A, O=O)
    assert np.allclose(O, A[idx] + 1.0)


def test_explicit_tasklet_memlet_with_a_data_bound():
    """The same rule where explicit-dataflow syntax builds its memlets: a data
    read in a slice bound is a range bound there too. Left to the memlet parser
    it became an un-evaluated ``rows(i)`` inside the subset."""

    @dace.program
    def explicit_chunked(rows: dace.int64[4], x: dace.float64[20], O: dace.float64[4]):
        for i in dace.map[0:4]:
            with dace.tasklet:
                chunk << x[rows[i]:rows[i] + 2]
                out >> O[i]
                out = chunk[0] + chunk[1]

    rows, x, O = np.array([0, 2, 4, 6]), np.arange(20, dtype=np.float64), np.zeros(4)
    tree = nextgen.parse_program(explicit_chunked, rows, x, O)
    assert not _callbacks(tree)
    tasklet = [node for node in tree.preorder_traversal() if isinstance(node, tn.TaskletNode)][-1]
    assert str(tasklet.in_memlets['chunk'].subset).startswith('__idx_rows')

    tree.as_sdfg().compile()(rows=rows, x=x, O=O)
    assert np.allclose(O, [x[row] + x[row + 1] for row in rows])


def test_a_reassigned_bound_gets_a_fresh_symbol():
    """The per-statement symbol cache is dropped between statements, so the
    second use of a rebound scalar cannot pick up the first definition."""

    @dace.program
    def rebound_twice(bounds: dace.int64[2], A: dace.float64[10], O: dace.float64[10]):
        n = bounds[0]
        O[0:n] = A[0:n] + 1.0
        n = bounds[1]
        O[0:n] = A[0:n] + 2.0

    bounds, A, O = np.array([3, 6]), np.arange(10, dtype=np.float64), np.zeros(10)
    tree = nextgen.parse_program(rebound_twice, bounds, A, O)
    assert not _callbacks(tree)
    assigns = [node for node in tree.preorder_traversal() if isinstance(node, tn.AssignNode)]
    assert len({node.name for node in assigns}) == 2

    tree.as_sdfg().compile()(bounds=bounds, A=A, O=O)
    assert np.allclose(O[:6], A[:6] + 2.0) and not O[6:].any()


if __name__ == '__main__':
    test_map_dependent_slice_target()
    test_map_dependent_slice_source()
    test_full_reduction_broadcasts_into_a_slice()
    test_range_length_from_data_becomes_a_symbol()
    test_data_length_inside_a_map_scope()
    test_read_through_a_data_bound_is_not_indirection()
    test_single_element_slice_is_still_a_range()
    test_range_bound_read_from_an_array()
    test_a_bound_read_through_a_scalar_index()
    test_a_step_read_from_data()
    test_element_index_beside_a_slice_becomes_a_symbol()
    test_unindexed_dimensions_count_as_a_range()
    test_element_index_alone_stays_indirection()
    test_explicit_tasklet_memlet_with_a_data_bound()
    test_a_reassigned_bound_gets_a_fresh_symbol()
