# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace
from dace.frontend.python import nextgen
from dace.sdfg.analysis.schedule_tree import treenodes as tn

#: Cases where the classic frontend answers differently from NumPy, and nextgen
#: deliberately follows NumPy instead. Kept as documentation, and so a future
#: classic fix is not mistaken for a regression here.
CLASSIC_DIVERGENCES = (
    'A[0:1, :]',
    'A[:, 3:4]',
    'A[1:2, 3:4]',
    'A[ind, 4:5]',
    'return of a size-1-dimension binding',
)


def _callbacks(tree):
    return [node for node in tree.preorder_traversal() if isinstance(node, tn.PythonCallbackNode)]


def _check(program, reference, **arguments):
    """
    Build, assert callback-free, execute, and compare against NumPy.

    :param reference: The NumPy result the program must reproduce.
    """
    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    result = tree.as_sdfg()(**arguments)
    assert result is not None, 'program returned nothing'
    assert np.shape(result) == np.shape(reference), f'shape {np.shape(result)} != NumPy {np.shape(reference)}'
    assert np.allclose(result, reference)
    return result


A2 = np.arange(200, dtype=np.float64).reshape(20, 10).copy()
A3 = np.arange(192, dtype=np.float64).reshape(8, 6, 4).copy()
A1 = np.arange(20, dtype=np.float64).copy()
IND = np.array([1, 3, 5], dtype=np.int32)
JND = np.array([0, 1, 2], dtype=np.int32)
IND2D = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

# --- Basic indexing: size-1 dimensions survive, as in NumPy


def test_slice_keeps_size_one_leading():
    """``A[0:1, :]`` is (1, 10). Classic squeezes this to (10,)."""

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[0:1, :]

    _check(program, A2[0:1, :], A=A2)


def test_slice_keeps_size_one_trailing():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[:, 3:4]

    _check(program, A2[:, 3:4], A=A2)


def test_slice_keeps_size_one_in_both_dimensions():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[1:2, 3:4]

    _check(program, A2[1:2, 3:4], A=A2)


def test_integer_index_still_drops_its_dimension():
    """The counterpart: an INTEGER index drops, a size-1 slice keeps."""

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[1, 3:4]

    _check(program, A2[1, 3:4], A=A2)


def test_size_one_binding_survives_the_return():

    @dace.program
    def program(A: dace.float64[20, 10]):
        b = A[0:1, :]
        return b

    _check(program, A2[0:1, :], A=A2)


# --- newaxis


def test_newaxis_in_the_middle():
    """``newaxis`` inserts a size-1 axis wherever it is written."""

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[:, None, :]

    _check(program, A2[:, None, :], A=A2)


def test_newaxis_leading_with_ellipsis():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[None, ...]

    _check(program, A2[None, ...], A=A2)


def test_newaxis_trailing_with_ellipsis():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[..., None]

    _check(program, A2[..., None], A=A2)


def test_newaxis_alone():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[None]

    _check(program, A2[None], A=A2)


def test_newaxis_survives_into_an_operand():
    """
    The outer-product idiom. The binding path already kept the axis; the bug
    was that consuming the value as an OPERAND went through a different shape
    derivation that dropped it, silently producing a rank-1 result.
    """

    @dace.program
    def program(A: dace.float64[20]):
        return A[:, None] * A[None, :]

    _check(program, A1[:, None] * A1[None, :], A=A1)


def test_newaxis_survives_arithmetic():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[:, None, :] + 1.0

    _check(program, A2[:, None, :] + 1.0, A=A2)


# --- newaxis combined with advanced indexing


def test_newaxis_between_index_array_and_integer():
    """``A[ind, None, 4]`` -- this used to crash out of the pipeline with a
    ``TypeError`` from descriptor validation."""

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[ind, None, 4]

    _check(program, A2[IND, None, 4], A=A2, ind=IND)


def test_newaxis_after_index_array():

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[ind, None]

    _check(program, A2[IND, None], A=A2, ind=IND)


def test_newaxis_before_index_array():

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[None, ind]

    _check(program, A2[None, IND], A=A2, ind=IND)


def test_newaxis_before_index_array_and_integer():

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[None, ind, 4]

    _check(program, A2[None, IND, 4], A=A2, ind=IND)


# --- Advanced-index chunk placement


def test_integer_does_not_separate_the_advanced_chunk():
    """
    NumPy separates the advanced chunk only on a slice, Ellipsis or newaxis --
    an integer index between two index arrays keeps them one group, so the
    chunk stays in place rather than moving to the front.
    """

    @dace.program
    def program(A: dace.float64[8, 6, 4, 5], ind: dace.int32[3], jnd: dace.int32[3]):
        return A[:, ind, 0, jnd]

    A4 = np.arange(8 * 6 * 4 * 5, dtype=np.float64).reshape(8, 6, 4, 5).copy()
    _check(program, A4[:, IND, 0, JND], A=A4, ind=IND, jnd=JND)


def test_slice_does_separate_the_advanced_chunk():
    """The contrast: a slice between them moves the chunk to the front."""

    @dace.program
    def program(A: dace.float64[8, 6, 4, 5], ind: dace.int32[3], jnd: dace.int32[3]):
        return A[:, ind, :, jnd]

    A4 = np.arange(8 * 6 * 4 * 5, dtype=np.float64).reshape(8, 6, 4, 5).copy()
    _check(program, A4[:, IND, :, JND], A=A4, ind=IND, jnd=JND)


def test_index_array_beside_a_size_one_slice():
    """``A[ind, 4:5]`` is (3, 1): the slice-formed dimension survives."""

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[ind, 4:5]

    _check(program, A2[IND, 4:5], A=A2, ind=IND)


def test_separated_index_arrays_move_chunk_to_front():

    @dace.program
    def program(A: dace.float64[8, 6, 4], ind: dace.int32[3], jnd: dace.int32[3]):
        return A[ind, :, jnd]

    _check(program, A3[IND, :, JND], A=A3, ind=IND, jnd=JND)


def test_two_dimensional_index_array():

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[2, 3]):
        return A[ind, 4]

    _check(program, A2[IND2D, 4], A=A2, ind=IND2D)


def test_ellipsis_with_index_array():

    @dace.program
    def program(A: dace.float64[8, 6, 4], ind: dace.int32[3]):
        return A[..., ind]

    _check(program, A3[..., JND], A=A3, ind=JND)


# --- Nested / chained subscripts


def test_nested_basic_then_newaxis_then_ellipsis():
    """``A[i][:, None][..., k]`` -- the chained form, each link a separate
    binding after ANF hoisting."""

    @dace.program
    def program(A: dace.float64[8, 6, 4]):
        return A[2][:, None][..., 1]

    _check(program, A3[2][:, None][..., 1], A=A3)


def test_nested_through_an_index_array():

    @dace.program
    def program(A: dace.float64[8, 6, 4], ind: dace.int32[3]):
        return A[ind][1]

    _check(program, A3[IND][1], A=A3, ind=IND)


def test_basic_slice_then_index_array():

    @dace.program
    def program(A: dace.float64[8, 6, 4], ind: dace.int32[3]):
        return A[1:5][ind]

    _check(program, A3[1:5][JND], A=A3, ind=JND)


def test_index_array_then_basic():

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[:, ind][0]

    _check(program, A2[:, IND][0], A=A2, ind=IND)


def test_view_of_a_computed_transient():
    """
    Regression test for a silent miscompilation found while building this suite.

    ``B`` is a transient written by a map; ``B[1]`` binds a ``ViewNode`` over
    it. The tree-to-SDFG conversion used to read the source through a fresh
    access node, disconnecting it from the map's output, so the returned copy
    read uninitialized memory (and simplification eliminated the map).
    """

    @dace.program
    def program(A: dace.float64[8, 6, 4]):
        B = A + 1.0
        return B[1]

    _check(program, (A3 + 1.0)[1], A=A3)


# --- Index expressions (spelling, not a feature boundary)


def test_index_by_a_slice_of_an_index_array():

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[5]):
        return A[ind[0:3], 4]

    ind5 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    _check(program, A2[ind5[0:3], 4], A=A2, ind=ind5)


def test_index_by_arithmetic_on_an_index_array():

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[ind + 1, 4]

    _check(program, A2[IND + 1, 4], A=A2, ind=IND)


# --- Negative-step slices (reversed traversal)


def test_reversed_slice():
    """``A[::-1]``. The classic frontend refuses this outright."""

    @dace.program
    def program(A: dace.float64[20]):
        return A[::-1]

    _check(program, A1[::-1], A=A1)


def test_reversed_slice_with_stride():

    @dace.program
    def program(A: dace.float64[20]):
        return A[::-2]

    _check(program, A1[::-2], A=A1)


def test_reversed_slice_with_bounds():
    """``A[10:2:-1]`` selects 8 elements, not 10 -- the exclusive stop counts
    downward."""

    @dace.program
    def program(A: dace.float64[20]):
        return A[10:2:-1]

    _check(program, A1[10:2:-1], A=A1)


def test_reversed_slice_on_one_axis_of_several():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[:, ::-1]

    _check(program, A2[:, ::-1], A=A2)


def test_reversed_slice_on_both_axes():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[::-1, ::-1]

    _check(program, A2[::-1, ::-1], A=A2)


def test_reversed_slice_beside_an_integer_index():

    @dace.program
    def program(A: dace.float64[20, 10]):
        return A[-1, ::-1]

    _check(program, A2[-1, ::-1], A=A2)


def test_reversed_slice_in_an_expression():

    @dace.program
    def program(A: dace.float64[20]):
        return A[::-1] + 1.0

    _check(program, A1[::-1] + 1.0, A=A1)


def test_reversed_slice_as_a_write_target():

    @dace.program
    def program(A: dace.float64[20], B: dace.float64[20]):
        A[::-1] = B

    A = A1.copy()
    B = np.arange(100, 120, dtype=np.float64)
    expected = A1.copy()
    expected[::-1] = B

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    tree.as_sdfg()(A=A, B=B)
    assert np.allclose(A, expected)


def test_reversed_slice_combined_with_an_index_array():

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        A[ind, ::-1] = 3.0

    A = A2.copy()
    expected = A2.copy()
    expected[IND, ::-1] = 3.0

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    tree.as_sdfg()(A=A, ind=IND)
    assert np.allclose(A, expected)


# --- Negative entries in an index array


def test_negative_index_array_wraps_when_enabled():
    """
    NumPy wraps a negative index entry. The emitted tasklet indexes the base
    pointer directly, so the wrap has to be in the tasklet body -- gated on the
    same configuration entry scalar indices already use.
    """

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[ind, 4]

    negative = np.array([-1, -3, 2], dtype=np.int32)
    with dace.config.set_temporary('frontend', 'runtime_negative_indices', value=True):
        _check(program, A2[negative, 4], A=A2, ind=negative)


def test_negative_index_array_wrap_is_off_by_default():
    """
    The default stays byte-identical: no wrap in the tasklet, so a nonnegative
    index array costs exactly what it did before.
    """

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3]):
        return A[ind, 4]

    with dace.config.set_temporary('frontend', 'runtime_negative_indices', value=False):
        tree = nextgen.parse_program(program)
        tasklets = [node for node in tree.preorder_traversal() if isinstance(node, tn.TaskletNode)]
        assert any(node.node.code.as_string.strip() == '__out = __arr[__inp0]' for node in tasklets), \
            [node.node.code.as_string for node in tasklets]
        # Nonnegative entries are unaffected by the setting either way.
        _check(program, A2[IND, 4], A=A2, ind=IND)


def test_negative_index_array_wraps_on_a_scatter():
    """The write side wraps too, not just the gather."""

    @dace.program
    def program(A: dace.float64[20, 10], ind: dace.int32[3], values: dace.float64[3]):
        A[ind, 4] = values

    negative = np.array([-1, -3, 2], dtype=np.int32)
    values = np.array([7.0, 8.0, 9.0])
    expected = A2.copy()
    expected[negative, 4] = values

    A = A2.copy()
    with dace.config.set_temporary('frontend', 'runtime_negative_indices', value=True):
        tree = nextgen.parse_program(program)
        assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
        tree.as_sdfg()(A=A, ind=negative, values=values)
    assert np.allclose(A, expected)


# --- A gather and a scatter in the same statement. The scatter's map iterates
# the TARGET's index space, so the source read cannot be emitted inside it; the
# read is gathered into a temporary first, which also gives NumPy's evaluation
# order (the whole right-hand side is read before the target is touched).


def test_gather_and_scatter_in_one_statement():

    @dace.program
    def program(A: dace.float64[20], i1: dace.int32[3], i2: dace.int32[3]):
        A[i1] = A[i2]

    A = A1.copy()
    expected = A1.copy()
    expected[IND] = expected[JND]

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    tree.as_sdfg()(A=A, i1=IND.copy(), i2=JND.copy())
    assert np.allclose(A, expected)


def test_gather_and_scatter_with_overlapping_indices():
    """The source and target index the SAME elements. NumPy reads the whole
    right-hand side first, so the result must not depend on write order."""

    @dace.program
    def program(A: dace.float64[20], ind: dace.int32[3]):
        A[ind] = A[ind] + 1.0

    A = A1.copy()
    expected = A1.copy()
    expected[IND] = expected[IND] + 1.0

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    tree.as_sdfg()(A=A, ind=IND.copy())
    assert np.allclose(A, expected)


def test_gather_and_scatter_across_two_containers():

    @dace.program
    def program(A: dace.float64[20], B: dace.float64[20], i1: dace.int32[3], i2: dace.int32[3]):
        A[i1] = B[i2] + 1.0

    A = A1.copy()
    B = np.arange(100, 120, dtype=np.float64).copy()
    expected = A1.copy()
    expected[IND] = B[JND] + 1.0

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    tree.as_sdfg()(A=A, B=B, i1=IND.copy(), i2=JND.copy())
    assert np.allclose(A, expected)


def test_gather_and_scatter_on_a_two_dimensional_target():

    @dace.program
    def program(A: dace.float64[20, 10], i1: dace.int32[3], i2: dace.int32[3]):
        A[i1, 4] = A[i2, 5]

    A = A2.copy()
    expected = A2.copy()
    expected[IND, 4] = expected[JND, 5]

    tree = nextgen.parse_program(program)
    assert not _callbacks(tree), [node.reason for node in _callbacks(tree)]
    tree.as_sdfg()(A=A, i1=IND.copy(), i2=JND.copy())
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_slice_keeps_size_one_leading()
    test_slice_keeps_size_one_trailing()
    test_slice_keeps_size_one_in_both_dimensions()
    test_integer_index_still_drops_its_dimension()
    test_size_one_binding_survives_the_return()
    test_newaxis_in_the_middle()
    test_newaxis_leading_with_ellipsis()
    test_newaxis_trailing_with_ellipsis()
    test_newaxis_alone()
    test_newaxis_survives_into_an_operand()
    test_newaxis_survives_arithmetic()
    test_newaxis_between_index_array_and_integer()
    test_newaxis_after_index_array()
    test_newaxis_before_index_array()
    test_newaxis_before_index_array_and_integer()
    test_integer_does_not_separate_the_advanced_chunk()
    test_slice_does_separate_the_advanced_chunk()
    test_index_array_beside_a_size_one_slice()
    test_separated_index_arrays_move_chunk_to_front()
    test_two_dimensional_index_array()
    test_ellipsis_with_index_array()
    test_nested_basic_then_newaxis_then_ellipsis()
    test_nested_through_an_index_array()
    test_basic_slice_then_index_array()
    test_index_array_then_basic()
    test_view_of_a_computed_transient()
    test_index_by_a_slice_of_an_index_array()
    test_index_by_arithmetic_on_an_index_array()
    test_reversed_slice()
    test_reversed_slice_with_stride()
    test_reversed_slice_with_bounds()
    test_reversed_slice_on_one_axis_of_several()
    test_reversed_slice_on_both_axes()
    test_reversed_slice_beside_an_integer_index()
    test_reversed_slice_in_an_expression()
    test_reversed_slice_as_a_write_target()
    test_reversed_slice_combined_with_an_index_array()
    test_negative_index_array_wraps_when_enabled()
    test_negative_index_array_wrap_is_off_by_default()
    test_negative_index_array_wraps_on_a_scatter()
    test_gather_and_scatter_in_one_statement()
    test_gather_and_scatter_with_overlapping_indices()
    test_gather_and_scatter_across_two_containers()
    test_gather_and_scatter_on_a_two_dimensional_target()
