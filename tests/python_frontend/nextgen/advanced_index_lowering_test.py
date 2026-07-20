# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
NumPy advanced (array-valued) indexing.

Advanced indexing and scalar indirection look alike in the AST — both are a
subscript whose index mentions a data container — but they are different
features. Scalar indirection (``x[A_col[j]]``) reads *one element* to index
with, and the result has the subset's shape. Advanced indexing (``A[indices]``)
uses whole arrays, which broadcast against each other, and the result shape
follows NumPy's own rules.

Confusing the two is not a benign mistake. Lowering ``A[ind]`` through the
indirection mechanism yields a tree that scores as a *success* under the
callback-discrepancy check — no callbacks at all — while giving the index array
the base array's subset, so the emitted memlet is out of bounds and the
compiled program segfaults. Every test here therefore checks the *value*
produced, not just that something lowered.
"""
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

M = dace.symbol('M')
N = dace.symbol('N')


def _callbacks(root: tn.ScheduleTreeRoot):
    found = []

    def walk(node):
        for child in getattr(node, 'children', None) or []:
            if isinstance(child, tn.PythonCallbackNode):
                found.append(child)
            walk(child)

    walk(root)
    return found


def _run(program, **arguments):
    """Build the nextgen schedule tree, convert it and execute, asserting that
    nothing fell back to the interpreter along the way."""
    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    return tree.as_sdfg()(**arguments)


# --- Reads


def test_index_by_one_array():
    """The index array's shape becomes the result shape."""

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[ind, 4]

    A, ind = np.random.rand(20, 10), np.array([1, 10, 15], dtype=np.int32)
    result = np.asarray(_run(program, A=A, ind=ind))
    np.testing.assert_allclose(result.reshape(A[ind, 4].shape), A[ind, 4])


def test_index_by_two_arrays_broadcasts():
    """Several index arrays broadcast together into one result chunk."""

    @dace.program
    def program(A: dace.float64[4, 3], rows: dace.int64[2, 2], columns: dace.int64[2, 2]):
        return A[rows, columns]

    A = np.arange(12, dtype=np.float64).reshape(4, 3)
    rows = np.array([[0, 0], [3, 3]], dtype=np.int64)
    columns = np.array([[0, 2], [0, 2]], dtype=np.int64)
    result = np.asarray(_run(program, A=A, rows=rows, columns=columns))
    np.testing.assert_allclose(result.reshape(A[rows, columns].shape), A[rows, columns])


def test_index_combined_with_slices():
    """Basic and advanced indexing in one subscript. The advanced indices are
    non-contiguous here, so the broadcast chunk moves to the front."""

    @dace.program
    def program(A: dace.float64[20, 10, 30], ind: dace.int32[3]):
        return A[ind, 2:7:2, [15, 10, 1]]

    A, ind = np.random.rand(20, 10, 30), np.array([1, 10, 15], dtype=np.int32)
    expected = A[ind, 2:7:2, [15, 10, 1]]
    result = np.asarray(_run(program, A=A, ind=ind))
    np.testing.assert_allclose(result.reshape(expected.shape), expected)


def test_index_inside_a_larger_expression():
    """A gather nested in an expression materializes into a temporary. ANF
    leaves it in operand position, so the split happens at lowering."""

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[M], B: dace.float64[M]):
        return A[ind] + B

    A, ind, B = np.random.rand(20), np.array([1, 5, 9], dtype=np.int32), np.ones(3)
    result = np.asarray(_run(program, A=A, ind=ind, B=B, N=20, M=3))
    np.testing.assert_allclose(result.ravel(), A[ind] + B)


# --- Writes


def test_write_through_index_array():

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[M]):
        A[ind] = 2

    A, ind = np.random.rand(20), np.array([1, 10, 15], dtype=np.int32)
    expected = A.copy()
    expected[ind] = 2
    _run(program, A=A, ind=ind, N=20, M=3)
    np.testing.assert_allclose(A, expected)


def test_accumulate_through_index_array():
    """An accumulating scatter takes conflict resolution unconditionally: the
    index array may name the same element twice, which the frontend cannot
    rule out by inspecting subsets."""

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[M]):
        A[ind] += 1

    A, ind = np.random.rand(20), np.array([1, 10, 15], dtype=np.int32)
    expected = A.copy()
    expected[ind] += 1
    _run(program, A=A, ind=ind, N=20, M=3)
    np.testing.assert_allclose(A, expected)


def test_write_combined_with_slices():

    @dace.program
    def program(A: dace.float64[N, N, N], ind: dace.int32[M]):
        A[1:2, ind, 3:4] = 2

    A, ind = np.random.rand(20, 20, 20), np.array([1, 10, 15], dtype=np.int32)
    expected = A.copy()
    expected[1:2, ind, 3:4] = 2
    _run(program, A=A, ind=ind, N=20, M=3)
    np.testing.assert_allclose(A, expected)


def test_write_broadcasts_the_value():

    @dace.program
    def program(A: dace.float64[N, N, N], B: dace.float64[N, N], ind: dace.int32[M]):
        A[ind] = B

    A, B = np.random.rand(20, 20, 20), np.random.rand(20, 20)
    ind = np.array([1, 10, 15], dtype=np.int32)
    expected = A.copy()
    expected[ind] = B
    _run(program, A=A, B=B, ind=ind, N=20, M=3)
    np.testing.assert_allclose(A, expected)


# --- Boolean masks


def test_masked_write():
    """A mask selects by predicate, so the number of written elements is
    data-dependent -- but their positions are not, which is why a masked
    *write* lowers as a guarded update over the full array and needs no
    dynamic allocation."""

    @dace.program
    def program(A: dace.float64[20, 30], mask: dace.bool[20, 30]):
        A[mask] = 2

    A = np.tile(np.arange(30, dtype=np.float64), (20, 1))
    mask = A > 15
    expected = A.copy()
    expected[mask] = 2
    _run(program, A=A, mask=mask)
    np.testing.assert_allclose(A, expected)


def test_masked_accumulation():

    @dace.program
    def program(A: dace.float64[20, 30], mask: dace.bool[20, 30]):
        A[mask] += 5

    A = np.tile(np.arange(30, dtype=np.float64), (20, 1))
    mask = A > 15
    expected = A.copy()
    expected[mask] += 5
    _run(program, A=A, mask=mask)
    np.testing.assert_allclose(A, expected)


def test_masked_read_falls_back():
    """``b = A[mask]`` has a result length known only at runtime, so it cannot
    be allocated at build time. Matches the classic frontend, which rejects
    boolean indexing outside assignment targets outright."""

    @dace.program
    def program(A: dace.float64[20, 30], mask: dace.bool[20, 30]):
        return A[mask] + 1.0

    assert _callbacks(nextgen.parse_program(program))


# --- The boundary with scalar indirection


def test_scalar_indirection_still_lowers():
    """A genuine one-element index read is still lowered as indirection, and
    must not be diverted into the advanced-indexing mechanism."""

    @dace.program
    def program(x: dace.float64[20], col: dace.int32[10], out: dace.float64[10]):
        for i in dace.map[0:10]:
            out[i] = x[col[i]] + 1.0

    x, col, out = np.random.rand(20), np.arange(10, dtype=np.int32), np.zeros(10)
    _run(program, x=x, col=col, out=out)
    np.testing.assert_allclose(out, x[col] + 1.0)


def test_mixed_indexing_does_not_raise_syntaxerror():
    """``A[ind, 2:7:2, ...]`` used to escape as a raw ``SyntaxError`` from
    re-parsing ``__in0[(__in1, 2:7:2, __in2)]``: an unparsed index tuple carries
    parentheses, and a slice is only legal in a bare subscript."""

    @dace.program
    def program(A: dace.float64[20, 10, 30], ind: dace.int32[3]):
        return A[ind, 2:7:2, [15, 10, 1]]

    nextgen.parse_program(program)


if __name__ == '__main__':
    test_index_by_one_array()
    test_index_by_two_arrays_broadcasts()
    test_index_combined_with_slices()
    test_index_inside_a_larger_expression()
    test_write_through_index_array()
    test_accumulate_through_index_array()
    test_write_combined_with_slices()
    test_write_broadcasts_the_value()
    test_masked_write()
    test_masked_accumulation()
    test_masked_read_falls_back()
    test_scalar_indirection_still_lowers()
    test_mixed_indexing_does_not_raise_syntaxerror()
