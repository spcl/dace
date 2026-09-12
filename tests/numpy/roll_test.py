# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``numpy.roll``, which lowers onto the ``CShift`` library node.

``CShift`` is Fortran ``CSHIFT`` and rotates the OTHER WAY, so the node carries ``-shift``. That
sign is the whole risk in this replacement: getting it backwards yields an array of the right
shape, dtype and even the right multiset of values, so only a value comparison against numpy --
and the structural assertion on the node's own ``shift`` property -- can catch it.
"""
import numpy as np
import pytest

import dace
from dace.libraries.standard.nodes.cshift import CShift, ShiftDirection
from common import compare_numpy_output


@compare_numpy_output()
def test_roll_1d_forward(A: dace.float64[8]):
    return np.roll(A, 3)


@compare_numpy_output()
def test_roll_1d_backward(A: dace.float64[8]):
    return np.roll(A, -3)


@compare_numpy_output()
def test_roll_1d_zero_is_a_copy(A: dace.float64[8]):
    return np.roll(A, 0)


@compare_numpy_output()
def test_roll_1d_shift_exceeds_the_extent(A: dace.float64[5]):
    """numpy wraps a shift larger than the axis; the floored modulus has to as well."""
    return np.roll(A, 13)


@compare_numpy_output()
def test_roll_2d_first_axis(A: dace.float64[4, 6]):
    return np.roll(A, 1, axis=0)


@compare_numpy_output()
def test_roll_2d_last_axis(A: dace.float64[4, 6]):
    return np.roll(A, 2, axis=1)


@compare_numpy_output()
def test_roll_2d_negative_axis(A: dace.float64[4, 6]):
    return np.roll(A, -2, axis=-1)


@compare_numpy_output()
def test_roll_3d_middle_axis(A: dace.float64[3, 5, 4]):
    return np.roll(A, 2, axis=1)


@compare_numpy_output()
def test_roll_tuple_of_shifts_and_axes(A: dace.float64[4, 6]):
    """numpy applies the pairs in order; each becomes its own node in the chain."""
    return np.roll(A, (1, -2), axis=(0, 1))


@compare_numpy_output()
def test_roll_one_shift_over_several_axes(A: dace.float64[4, 6]):
    return np.roll(A, 2, axis=(0, 1))


@compare_numpy_output()
def test_roll_int32_keeps_its_dtype(A: dace.int32[8]):
    return np.roll(A, 2)


def cshift_nodes(program):
    """Every ``CShift`` library node in ``program``'s parsed SDFG, recursively."""
    sdfg = program.to_sdfg(simplify=False)
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CShift)]


def test_roll_lowers_to_a_cshift_in_the_numpy_direction():
    """The node must say NUMPY and keep the shift as written. A FORTRAN node with the same shift
    rotates the other way and still type-checks, which is why the direction is asserted here."""

    @dace.program
    def prog(a: dace.float64[8], out: dace.float64[8]):
        out[:] = np.roll(a, 3)

    nodes = cshift_nodes(prog)
    assert len(nodes) == 1
    assert nodes[0].dim == 1  # CShift's dim is FORTRAN 1-based; axis 0 is dim 1
    assert nodes[0].shift == 3
    assert nodes[0].direction is ShiftDirection.NUMPY


def test_roll_over_two_axes_chains_two_nodes():
    """One node per (shift, axis) pair, in numpy's order -- not one node that tries to do both."""

    @dace.program
    def prog(a: dace.float64[4, 6], out: dace.float64[4, 6]):
        out[:] = np.roll(a, (1, -2), axis=(0, 1))

    nodes = cshift_nodes(prog)
    assert len(nodes) == 2
    assert sorted((n.dim, n.shift) for n in nodes) == [(1, 1), (2, -2)]
    assert all(n.direction is ShiftDirection.NUMPY for n in nodes)


def test_an_axis_less_roll_over_a_matrix_is_refused():
    """numpy FLATTENS here, which is a reshape only when the operand is contiguous."""

    @dace.program
    def prog(a: dace.float64[4, 6], out: dace.float64[4, 6]):
        out[:] = np.roll(a, 2)

    with pytest.raises(Exception, match="flattens"):
        prog.to_sdfg()


def test_mismatched_shift_and_axis_counts_are_refused():

    @dace.program
    def prog(a: dace.float64[4, 6], out: dace.float64[4, 6]):
        out[:] = np.roll(a, (1, 2, 3), axis=(0, 1))

    with pytest.raises(Exception, match="must agree"):
        prog.to_sdfg()


def test_roll_of_a_strided_view_keeps_the_view_strides():
    """The expanded node must index the operand the way the OPERAND is laid out.

    A view like ``a[:, 0:8:2]`` has strides ``(8, 2)``. Declaring the ``_x`` connector compact
    makes the expansion read it as ``d0*4 + d1``, which lands on entirely different elements --
    and still produces an array of the right shape and dtype, so only the strides or the values
    can catch it.
    """
    n = 8

    @dace.program
    def prog(a: dace.float64[n, n], out: dace.float64[n, n // 2]):
        out[:] = np.roll(a[:, 0:n:2], -1, axis=1)

    sdfg = prog.to_sdfg(simplify=False)
    assert any(isinstance(v, CShift) for v, _ in sdfg.all_nodes_recursive()), "roll must lower to CShift"
    sdfg.expand_library_nodes()
    inner = next(v.sdfg for v, _ in sdfg.all_nodes_recursive()
                 if isinstance(v, dace.sdfg.nodes.NestedSDFG) and "_x" in v.sdfg.arrays)
    assert list(inner.arrays["_x"].strides) == [n, 2], \
        f'_x must carry the view strides, got {list(inner.arrays["_x"].strides)}'


def test_roll_of_a_strided_view_computes_the_right_values():
    """The numbers, not just the strides: the shape is right either way."""
    n = 8

    @dace.program
    def prog(a: dace.float64[n, n], out: dace.float64[n, n // 2]):
        out[:] = np.roll(a[:, 0:n:2], -1, axis=1)

    a = np.arange(n * n, dtype=np.float64).reshape(n, n)
    out = np.zeros((n, n // 2))
    prog(a=a.copy(), out=out)
    assert np.allclose(out, np.roll(a[:, 0:n:2], -1, axis=1)), f"{out[0]} != {np.roll(a[:, 0:n:2], -1, axis=1)[0]}"


if __name__ == '__main__':
    test_roll_1d_forward()
    test_roll_1d_backward()
    test_roll_1d_zero_is_a_copy()
    test_roll_1d_shift_exceeds_the_extent()
    test_roll_2d_first_axis()
    test_roll_2d_last_axis()
    test_roll_2d_negative_axis()
    test_roll_3d_middle_axis()
    test_roll_tuple_of_shifts_and_axes()
    test_roll_one_shift_over_several_axes()
    test_roll_int32_keeps_its_dtype()
    test_roll_lowers_to_a_cshift_in_the_numpy_direction()
    test_roll_over_two_axes_chains_two_nodes()
    test_an_axis_less_roll_over_a_matrix_is_refused()
    test_mismatched_shift_and_axis_counts_are_refused()
    test_roll_of_a_strided_view_keeps_the_view_strides()
    test_roll_of_a_strided_view_computes_the_right_values()
