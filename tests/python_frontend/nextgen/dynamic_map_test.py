# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for data-dependent ``dace.map`` bounds (dynamic map ranges) in the
next-generation Python frontend.

``for j in dace.map[A_row[i]:A_row[i + 1]]`` (and the desugared
``@dace.map(_[A_row[i]:A_row[i + 1]])`` form) must lower to a real
:class:`~dace.sdfg.nodes.MapEntry` whose range uses fresh symbols fed by
:class:`~dace.sdfg.analysis.schedule_tree.treenodes.DynScopeCopyNode` inputs
(dynamic map-range inputs), instead of falling back to an interpreter
callback for the whole loop. Index expressions are canonical in place (never
ANF-hoisted), so the lowering rule
(``control_flow.py::_bound``/``_dynamic_bound``) sees the bound expression
directly (a scalar integer element access like ``A_row[i]``, or a scalar
container name) and turns it into a dynamic map-range input, emitted as a
``DynScopeCopyNode`` sibling immediately preceding the map scope (matching
the placement SDFG-derived schedule trees use -- see ``sdfg_to_tree.py`` and
``tree_to_sdfg.py``).
"""
import numpy as np
import scipy.sparse as sp

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

H = dace.symbol('H')
nnz = dace.symbol('nnz')


def _nodes_of_type(root: tn.ScheduleTreeRoot, node_type):
    return [node for node in root.preorder_traversal() if isinstance(node, node_type)]


def _csr_reference(A_row: np.ndarray, A_val: np.ndarray, num_rows: int) -> np.ndarray:
    """Reference row-sum of a CSR-encoded matrix, computed without scipy/numpy reductions
    that could mask indexing mistakes."""
    result = np.zeros(num_rows, dtype=np.float32)
    for row in range(num_rows):
        for k in range(A_row[row], A_row[row + 1]):
            result[row] += A_val[k]
    return result


def test_dynamic_map_structure():
    """
    The exact csr_rowsum program (nested @dace.mapscope / @dace.map with
    data-dependent inner bounds) must lower with zero callbacks, exactly two
    DynScopeCopyNode inputs (for the row-start/row-end bounds) with distinct
    target symbols registered in the tree's symbol table, placed as siblings
    immediately preceding the inner map scope, and the inner map's range must
    reference both symbols.
    """

    @dace.program
    def csr_rowsum(A_row: dace.uint32[H + 1], A_val: dace.float32[nnz], b: dace.float32[H]):

        @dace.mapscope(_[0:H])
        def rows(i):

            @dace.map(_[A_row[i]:A_row[i + 1]])
            def nz(j):
                a << A_val[j]
                out >> b(1, lambda x, y: x + y)[i]
                out = a

    tree = nextgen.parse_program(csr_rowsum)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    maps = _nodes_of_type(tree, tn.MapScope)
    assert len(maps) == 2
    outer, inner = maps
    assert outer.node.map.params == ['i']
    assert inner.node.map.params == ['j']

    dyn_copies = _nodes_of_type(tree, tn.DynScopeCopyNode)
    assert len(dyn_copies) == 2

    targets = [node.target for node in dyn_copies]
    assert len(set(targets)) == 2, 'dynamic map-range symbols must be distinct'
    for target in targets:
        assert target in tree.symbols, f'"{target}" must be registered in the tree symbol table'

    # Placement: siblings immediately preceding the inner MapScope within its
    # parent scope's children (not children inside the MapScope itself).
    parent = inner.parent
    assert parent is not None
    index = parent.children.index(inner)
    assert index >= len(dyn_copies)
    preceding = parent.children[index - len(dyn_copies):index]
    assert all(isinstance(node, tn.DynScopeCopyNode) for node in preceding)
    assert {node.target for node in preceding} == set(targets)

    # The inner map's range must be expressed in terms of both dynamic symbols.
    range_symbols = {str(s) for s in inner.node.map.range.free_symbols}
    assert set(targets) <= range_symbols


def test_dynamic_map_execution():
    """
    End-to-end: a dynamic-bounds CSR row-sum with a regular Python for-loop
    as the outer row loop executes correctly against a scipy-built reference
    (the @dace.mapscope-nested variant is covered by
    test_dynamic_map_mapscope_execution).
    """

    @dace.program
    def csr_rowsum(A_row: dace.uint32[H + 1], A_val: dace.float32[nnz], b: dace.float32[H]):
        for i in range(H):

            @dace.map(_[A_row[i]:A_row[i + 1]])
            def nz(j):
                a << A_val[j]
                out >> b(1, lambda x, y: x + y)[i]
                out = a

    tree = nextgen.parse_program(csr_rowsum)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)

    sdfg = tree.as_sdfg()
    func = sdfg.compile()

    rng = np.random.default_rng(1234)
    num_rows = 7
    matrix = sp.random(num_rows, num_rows, density=0.3, format='csr', dtype=np.float32, random_state=rng)
    A_row = matrix.indptr.astype(np.uint32)
    A_val = matrix.data.astype(np.float32)
    b = np.zeros(num_rows, dtype=np.float32)

    func(A_row=A_row, A_val=A_val, b=b, H=num_rows, nnz=len(A_val))

    reference = _csr_reference(A_row, A_val, num_rows)
    assert np.allclose(b, reference, atol=1e-5)


def test_dynamic_map_plain_python_loop():
    """
    The same data-dependent bounds pattern, written with a regular
    ``for i in range(H)`` loop (no @dace.mapscope) wrapping
    ``for j in dace.map[A_row[i]:A_row[i + 1]]``, must lower with zero
    callbacks and execute correctly.
    """

    @dace.program
    def csr_rowsum_loop(A_row: dace.uint32[H + 1], A_val: dace.float32[nnz], b: dace.float32[H]):
        for i in range(H):
            for j in dace.map[A_row[i]:A_row[i + 1]]:
                with dace.tasklet:
                    a << A_val[j]
                    out >> b(1, lambda x, y: x + y)[i]
                    out = a

    tree = nextgen.parse_program(csr_rowsum_loop)
    assert not _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(_nodes_of_type(tree, tn.DynScopeCopyNode)) == 2

    sdfg = tree.as_sdfg()
    func = sdfg.compile()

    rng = np.random.default_rng(42)
    num_rows = 5
    matrix = sp.random(num_rows, num_rows, density=0.3, format='csr', dtype=np.float32, random_state=rng)
    A_row = matrix.indptr.astype(np.uint32)
    A_val = matrix.data.astype(np.float32)
    b = np.zeros(num_rows, dtype=np.float32)

    func(A_row=A_row, A_val=A_val, b=b, H=num_rows, nnz=len(A_val))

    reference = _csr_reference(A_row, A_val, num_rows)
    assert np.allclose(b, reference, atol=1e-5)


def test_dynamic_bound_unsupported_form():
    """
    Map bounds that are data-dependent but not a scalar-integer container
    (a whole array, or a floating-point scalar) must fall back to a Python
    callback for the enclosing loop rather than crash or silently emit an
    invalid range referencing an unregistered symbol.
    """

    @dace.program
    def whole_array_bound(A_row: dace.uint32[H], b: dace.float32[H]):
        for i in range(3):

            @dace.map(_[0:A_row])
            def nz(j):
                out >> b[j]
                out = 1.0

    tree = nextgen.parse_program(whole_array_bound)
    callbacks = _nodes_of_type(tree, tn.PythonCallbackNode)
    assert len(callbacks) == 1

    @dace.program
    def float_scalar_bound(b: dace.float32[H]):
        for i in range(3):
            fbound = 3.5

            @dace.map(_[0:fbound])
            def nz(j):
                out >> b[j]
                out = 1.0

    tree2 = nextgen.parse_program(float_scalar_bound)
    callbacks2 = _nodes_of_type(tree2, tn.PythonCallbackNode)
    assert len(callbacks2) == 1


def test_dynamic_map_mapscope_execution():
    """
    End-to-end: the exact CSR row-sum shape this feature targets — a
    data-dependent inner @dace.map nested in an outer @dace.mapscope.
    Index expressions stay in place (``dace.map[A_row[i]:A_row[i + 1]]``
    keeps its subscripts; ``i + 1`` resolves symbolically), so the dynamic
    inputs read ``A_row`` directly and the map compiles and runs even though
    it is nested inside another map scope.
    """

    @dace.program
    def csr_rowsum(A_row: dace.uint32[H + 1], A_val: dace.float32[nnz], b: dace.float32[H]):

        @dace.mapscope(_[0:H])
        def rows(i):

            @dace.map(_[A_row[i]:A_row[i + 1]])
            def nz(j):
                a << A_val[j]
                out >> b(1, lambda x, y: x + y)[i]
                out = a

    tree = nextgen.parse_program(csr_rowsum)
    sdfg = tree.as_sdfg()
    func = sdfg.compile()

    rng = np.random.default_rng(7)
    num_rows = 6
    matrix = sp.random(num_rows, num_rows, density=0.3, format='csr', dtype=np.float32, random_state=rng)
    A_row = matrix.indptr.astype(np.uint32)
    A_val = matrix.data.astype(np.float32)
    b = np.zeros(num_rows, dtype=np.float32)

    func(A_row=A_row, A_val=A_val, b=b, H=num_rows, nnz=len(A_val))

    reference = _csr_reference(A_row, A_val, num_rows)
    assert np.allclose(b, reference, atol=1e-5)


if __name__ == '__main__':
    test_dynamic_map_structure()
    test_dynamic_map_execution()
    test_dynamic_map_plain_python_loop()
    test_dynamic_bound_unsupported_form()
    test_dynamic_map_mapscope_execution()
