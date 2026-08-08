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
import pytest

import dace
from dace.frontend.python import nextgen
from dace.frontend.python.nextgen.common import FrontendError
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

    A = np.arange(12, dtype=np.float64).reshape(4, 3).copy()
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


def test_index_as_a_registry_operator_operand():
    """A gather feeding an operator the elementwise mechanism cannot express
    (``vals @ x[cols]``, the shape of every spmv).

    A registry implementation resolves its operands as plain accesses and an
    array-valued index is not one, so the gather has to be materialized BEFORE
    operator dispatch; the other order sent the whole statement to a callback.
    """

    @dace.program
    def program(vals: dace.float64[M], x: dace.float64[N], cols: dace.int32[M]):
        return vals @ x[cols]

    vals, x = np.random.rand(3), np.random.rand(20)
    cols = np.array([1, 5, 9], dtype=np.int32)
    result = np.asarray(_run(program, vals=vals, x=x, cols=cols, N=20, M=3))
    np.testing.assert_allclose(result.ravel()[0], vals @ x[cols])


# --- Index expressions
#
# The shared memlet parser recognizes an array-valued index only when it is
# written as a bare NAME whose descriptor it can look up. Every other spelling
# -- `A[ind[0]]`, `A[2:4, ind[1], 3]`, `A[ind[0] + 1]` -- was taken for a
# symbolic expression, which is the damaging failure: the access typed as a
# single ELEMENT rather than failing, so an ANF temporary holding the result
# was allocated as a scalar and the error surfaced later as a broadcast
# complaint about a different statement.


def test_index_expression_read():
    """An index array reached through a subscript, not a name."""

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[2, 3], O: dace.float64[3]):
        O[:] = A[ind[0]]

    A, ind, O = np.random.default_rng(0).random(20), np.array([[1, 3, 5], [0, 2, 4]], dtype=np.int32), np.zeros(3)
    _run(program, A=A, ind=ind, O=O, N=20, M=3)
    np.testing.assert_allclose(O, A[ind[0]])


def test_computed_index_expression():
    """The index is an expression over an array, evaluated into a container."""

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[2, 3], O: dace.float64[3]):
        O[:] = A[ind[0] + 1]

    A, ind, O = np.random.default_rng(1).random(20), np.array([[1, 3, 5], [0, 2, 4]], dtype=np.int32), np.zeros(3)
    _run(program, A=A, ind=ind, O=O, N=20, M=3)
    np.testing.assert_allclose(O, A[ind[0] + 1])


def test_sliced_index_expression():
    """``A[ind[0, 0:3]]``: this spelling fails during INFERENCE rather than at
    lowering -- the parser renders the index as an applied function it cannot
    then sympify -- so it exercises the recovery in ``rules.assign``."""

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[2, 5], O: dace.float64[3]):
        O[:] = A[ind[0, 0:3]]

    A = np.random.default_rng(2).random(20)
    ind = np.array([[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]], dtype=np.int32)
    O = np.zeros(3)
    _run(program, A=A, ind=ind, O=O, N=20, M=3)
    np.testing.assert_allclose(O, A[ind[0, 0:3]])


def test_index_expression_nested_in_an_expression():
    """The result shape has to be right at inference time: an ANF temporary is
    allocated from it, and a scalar temporary silently truncates the gather."""

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[2, 3], B: dace.float64[3], O: dace.float64[3]):
        O[:] = A[ind[0]] * 2.0 + B

    rng = np.random.default_rng(3)
    A, B, O = rng.random(20), rng.random(3), np.zeros(3)
    ind = np.array([[1, 3, 5], [0, 2, 4]], dtype=np.int32)
    _run(program, A=A, ind=ind, B=B, O=O, N=20, M=3)
    np.testing.assert_allclose(O, A[ind[0]] * 2.0 + B)


def test_two_index_expressions_broadcast():
    """Two array-valued index expressions in one subscript."""

    @dace.program
    def program(A: dace.float64[6, 10], ind: dace.int32[2, 3], O: dace.float64[3]):
        O[:] = A[ind[1], ind[0]]

    A = np.arange(60, dtype=np.float64).reshape(6, 10).copy()
    ind, O = np.array([[1, 3, 5], [0, 2, 4]], dtype=np.int32), np.zeros(3)
    _run(program, A=A, ind=ind, O=O, N=6, M=10)
    np.testing.assert_allclose(O, A[ind[1], ind[0]])


def test_index_expression_through_an_attribute():
    """``A[ind.T[0]]``: the index is a subscript of a registry attribute read,
    which is not a data name either."""

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[3, 2], O: dace.float64[3]):
        O[:] = A[ind.T[0]]

    A = np.random.default_rng(4).random(20)
    ind, O = np.array([[1, 0], [3, 2], [5, 4]], dtype=np.int32), np.zeros(3)
    _run(program, A=A, ind=ind, O=O, N=20, M=3)
    np.testing.assert_allclose(O, A[ind.T[0].copy()])


def test_index_expression_write_and_accumulate():
    """Both write forms, including the accumulation whose read and write sides
    name the same index expression (evaluated once, per statement)."""

    @dace.program
    def overwrite(A: dace.float64[N], ind: dace.int32[2, 3], B: dace.float64[3]):
        A[ind[0]] = B

    @dace.program
    def accumulate(A: dace.float64[N], ind: dace.int32[2, 3], B: dace.float64[3]):
        A[ind[0]] += B

    rng = np.random.default_rng(5)
    ind = np.array([[1, 3, 5], [0, 2, 4]], dtype=np.int32)

    A, B = rng.random(20), rng.random(3)
    expected = A.copy()
    expected[ind[0]] = B
    _run(overwrite, A=A, ind=ind, B=B, N=20, M=3)
    np.testing.assert_allclose(A, expected)

    A, B = rng.random(20), rng.random(3)
    expected = A.copy()
    expected[ind[0]] += B
    _run(accumulate, A=A, ind=ind, B=B, N=20, M=3)
    np.testing.assert_allclose(A, expected)


def test_multidim_index_expression_write():
    """``A[2:4, ind[1], 3] += B``: an index expression combined with a slice
    and a scalar (the shape `augmented_assignment_to_slice_test` uses)."""

    @dace.program
    def program(A: dace.int32[20, 10, 6], ind: dace.int32[2, 5], B: dace.int32[5]):
        A[2:4, ind[1], 3] += B

    A = np.arange(20 * 10 * 6, dtype=np.int32).reshape(20, 10, 6).copy()
    ind = np.array([[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]], dtype=np.int32)
    B = np.arange(5, dtype=np.int32) + 100
    expected = A.copy()
    expected[2:4, ind[1], 3] += B
    _run(program, A=A, ind=ind, B=B, N=20, M=5)
    np.testing.assert_array_equal(A, expected)


def test_scalar_index_expression_is_not_advanced_indexing():
    """A single-element index expression selects one element, so it stays
    indirection rather than becoming a gather."""

    @dace.program
    def program(A: dace.float64[N], ind: dace.int32[2, 3], O: dace.float64[1]):
        O[0] = A[ind[0, 1]]

    A, ind, O = np.random.default_rng(6).random(20), np.array([[1, 3, 5], [0, 2, 4]], dtype=np.int32), np.zeros(1)
    _run(program, A=A, ind=ind, O=O, N=20, M=3)
    np.testing.assert_allclose(O[0], A[ind[0, 1]])


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

    A = np.tile(np.arange(30, dtype=np.float64), (20, 1)).copy()
    mask = A > 15
    expected = A.copy()
    expected[mask] = 2
    _run(program, A=A, mask=mask)
    np.testing.assert_allclose(A, expected)


def test_masked_accumulation():

    @dace.program
    def program(A: dace.float64[20, 30], mask: dace.bool[20, 30]):
        A[mask] += 5

    A = np.tile(np.arange(30, dtype=np.float64), (20, 1)).copy()
    mask = A > 15
    expected = A.copy()
    expected[mask] += 5
    _run(program, A=A, mask=mask)
    np.testing.assert_allclose(A, expected)


def test_masked_read_lowers_but_cannot_be_returned():
    """
    ``A[mask] + 1.0`` has a result length known only at runtime. It lowers --
    the mask is gathered into a symbolically sized container first, and the
    computation runs over that -- but the result cannot cross the program
    boundary, because a compiled program's caller allocates the return value
    before the call. Refused by name; see ``boolean_gather_test.py`` for the
    forms that do work (consuming it inside the same program).
    """

    @dace.program
    def program(A: dace.float64[20, 30], mask: dace.bool[20, 30]):
        return A[mask] + 1.0

    with pytest.raises(FrontendError, match='only known while the program runs'):
        nextgen.parse_program(program)


def test_refusal_names_the_source_expression_not_the_temporary():
    """
    The refusal above is only useful if the reader can tell *which* expression
    it is about. Canonicalization hoists ``A[A > 15]`` into ``__anf0 = A > 15;
    __anf1 = A[__anf0]``, and the message used to name ``__anf1`` (sized by
    ``__nnz170``) -- two generated names with no way back to the program text.
    Provenance recorded at the hoist resolves them, transitively, so nested
    temporaries render as what the user wrote rather than as each other.
    """

    @dace.program
    def program(A: dace.float64[20, 30]):
        return A[A > 15]

    with pytest.raises(FrontendError) as raised:
        nextgen.parse_program(program)
    message = str(raised.value)
    assert 'Cannot return "A[A > 15]"' in message
    assert 'the number of elements "A > 15" selects' in message
    assert '__anf' not in message and '__nnz' not in message

    @dace.program
    def nested(A: dace.float64[20, 30]):
        return np.sum(A[A > 15]) + A[A > 15]

    with pytest.raises(FrontendError) as raised:
        nextgen.parse_program(nested)
    # Two levels of hoisting deep: the outer temporary's recorded expression
    # mentions the inner ones, which resolve in turn.
    assert 'Cannot return "numpy.sum(A[A > 15]) + A[A > 15]"' in str(raised.value)


def test_refusal_inside_an_inlined_callee_names_the_callee_source():
    """
    A callee is canonicalized by its own pipeline run, whose temporary names
    restart from zero, so the caller's name-keyed provenance must not answer
    for them -- and the callee's own records must still be reachable, which
    they are because they travel on the body the call site copies.
    """

    @dace.program
    def selector(A: dace.float64[20, 30]):
        return A[A > 15]

    @dace.program
    def program(A: dace.float64[20, 30]):
        return selector(A)

    with pytest.raises(FrontendError) as raised:
        nextgen.parse_program(program)
    message = str(raised.value)
    assert 'Cannot return "selector(A)"' in message
    assert 'the number of elements "A > 15" selects' in message


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
    test_index_as_a_registry_operator_operand()
    test_index_expression_read()
    test_computed_index_expression()
    test_sliced_index_expression()
    test_index_expression_nested_in_an_expression()
    test_two_index_expressions_broadcast()
    test_index_expression_through_an_attribute()
    test_index_expression_write_and_accumulate()
    test_multidim_index_expression_write()
    test_scalar_index_expression_is_not_advanced_indexing()
    test_write_through_index_array()
    test_accumulate_through_index_array()
    test_write_combined_with_slices()
    test_write_broadcasts_the_value()
    test_masked_write()
    test_masked_accumulation()
    test_masked_read_lowers_but_cannot_be_returned()
    test_refusal_names_the_source_expression_not_the_temporary()
    test_refusal_inside_an_inlined_callee_names_the_callee_source()
    test_scalar_indirection_still_lowers()
    test_mixed_indexing_does_not_raise_syntaxerror()
