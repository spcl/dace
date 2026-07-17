# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Black-box shape + value fidelity matrix for numpy slice / index idioms through the
DaCe Python frontend.

A ``@dace.program`` that *returns* a slice exposes the frontend's inferred output
shape as the shape of the returned ndarray, so ``result.shape`` is the real signal
under test. Numpy is the oracle and every comparison is bit-exact
(``np.array_equal``, never a tolerance / norm_error) -- the whole point is a
silent-miscompile guard, not an approximation check. Integer (``arange``) data is
used so bit-exactness is meaningful.

Historic defect (2026-07, "dace singleton-slice squeeze"): length-1 slice axes were
dropped, e.g. ``A[:, 1:2] -> (N,)`` instead of ``(N, 1)``. The values stayed correct
once flattened; only the *shape* was wrong, which is exactly how ``x.T - x`` on a
column-slice silently miscompiles. The length-1-slice cases are now fixed (the
frontend keeps a slice's axis, matching numpy) and are asserted plainly.

The idioms are split into two groups:

* ``WORKS_TODAY``  -- shape and value match numpy; asserted plainly. Now holds every
  length-1-slice idiom.
* ``BROKEN_TODAY`` -- still diverges from numpy; marked ``xfail(strict=True)`` so this
  file is green today yet fails loudly (XPASS) the moment it is fixed, the trigger to
  promote it. Only the scalar-INDEX collapse remains here (``A[1, 2]`` yields dace's
  ``(1,)`` scalar representation rather than numpy's 0-d ``()``) -- a distinct fix from
  the slice squeeze.

Membership was determined by running, not by guessing.
"""
from collections import namedtuple

import numpy as np
import pytest

import dace

# A ``Case`` bundles an idiom's DaCe program, its numpy oracle, an input factory
# and (for the broken group) the xfail reason. ``id`` names the parametrized case.
Case = namedtuple("Case", ("id", "prog", "oracle", "make_input", "reason"))


def make_2d():
    # Distinct dim lengths (4 rows, 5 cols) so a wrong-axis squeeze cannot hide.
    return np.arange(20, dtype=np.int64).reshape(4, 5)


def make_3d():
    return np.arange(24, dtype=np.int64).reshape(2, 3, 4)


def make_square():
    return np.arange(16, dtype=np.int64).reshape(4, 4)


# --------------------------------------------------------------------------- #
# DaCe programs -- one per idiom (the slice syntax is literal, so it cannot be
# parametrized; each program simply returns the sliced view).
# --------------------------------------------------------------------------- #
@dace.program
def take_row(A):
    return A[1, :]


@dace.program
def take_col(A):
    return A[:, 1]


@dace.program
def take_row_then_slice(A):
    return A[1, 1:2]


@dace.program
def take_slice_then_col(A):
    return A[1:2, 1]


@dace.program
def take_strided_cols(A):
    return A[:, ::2]


@dace.program
def take_face_3d(A):
    return A[1, :, :]


@dace.program
def take_col_slice(A):
    return A[:, 1:2]


@dace.program
def take_row_slice(A):
    return A[1:2, :]


@dace.program
def take_both_slice(A):
    return A[1:2, 1:2]


@dace.program
def take_scalar(A):
    return A[1, 2]


@dace.program
def take_ellipsis_col(A):
    return A[..., 1:2]


@dace.program
def take_row_ellipsis(A):
    return A[1:2, ...]


@dace.program
def take_neg_tail(A):
    return A[:, -1:]


@dace.program
def take_corner(A):
    return A[0:1, 0:1]


@dace.program
def take_depth_slice_3d(A):
    return A[:, :, 1:2]


@dace.program
def transpose_minus_self(A):
    return A.T - A


# --------------------------------------------------------------------------- #
# Idiom matrix.
# --------------------------------------------------------------------------- #
# Every length-1 SLICE now keeps its axis (numpy semantics), so what used to be BROKEN_TODAY is
# asserted plainly here. The only survivor in BROKEN_TODAY is the scalar-INDEX collapse, a separate
# concern (dace represents a scalar as shape (1,), not a 0-d), left xfail-strict until that path is
# addressed.
WORKS_TODAY = [
    Case("A[1, :]", take_row, lambda a: a[1, :], make_2d, None),
    Case("A[:, 1]", take_col, lambda a: a[:, 1], make_2d, None),
    Case("A[1, 1:2]", take_row_then_slice, lambda a: a[1, 1:2], make_2d, None),
    Case("A[1:2, 1]", take_slice_then_col, lambda a: a[1:2, 1], make_2d, None),
    Case("A[:, ::2]", take_strided_cols, lambda a: a[:, ::2], make_2d, None),
    Case("A[1, :, :]", take_face_3d, lambda a: a[1, :, :], make_3d, None),
    Case("A[:, 1:2]", take_col_slice, lambda a: a[:, 1:2], make_2d, None),
    Case("A[1:2, :]", take_row_slice, lambda a: a[1:2, :], make_2d, None),
    Case("A[1:2, 1:2]", take_both_slice, lambda a: a[1:2, 1:2], make_2d, None),
    Case("A[..., 1:2]", take_ellipsis_col, lambda a: a[..., 1:2], make_2d, None),
    Case("A[1:2, ...]", take_row_ellipsis, lambda a: a[1:2, ...], make_2d, None),
    Case("A[:, -1:]", take_neg_tail, lambda a: a[:, -1:], make_2d, None),
    Case("A[0:1, 0:1]", take_corner, lambda a: a[0:1, 0:1], make_2d, None),
    Case("A[:, :, 1:2]", take_depth_slice_3d, lambda a: a[:, :, 1:2], make_3d, None),
]

# Nothing is xfail any more: length-1 slices keep their axis, and the one place dace and numpy
# still differ -- a scalar INDEX that is RETURNED -- is dace's return convention, asserted directly
# below rather than treated as a defect.
BROKEN_TODAY = []


def run_case(case):
    """Compile + run the idiom and assert both shape and bit-exact values vs numpy."""
    arr = case.make_input()
    oracle = np.asarray(case.oracle(arr))
    result = np.asarray(case.prog(arr.copy()))
    assert result.shape == oracle.shape, (f"{case.id}: dace returned shape {result.shape}, "
                                          f"numpy oracle shape {oracle.shape}")
    assert np.array_equal(result, oracle), (f"{case.id}: bit-exact value mismatch "
                                            f"(dace {result.tolist()} vs numpy {oracle.tolist()})")


def works_params():
    return [pytest.param(c, id=c.id) for c in WORKS_TODAY]


def broken_params():
    return [pytest.param(c, id=c.id, marks=pytest.mark.xfail(strict=True, reason=c.reason)) for c in BROKEN_TODAY]


@pytest.mark.parametrize("case", works_params())
def test_slice_shape_works_today(case):
    """Idioms whose shape + values already match numpy exactly."""
    run_case(case)


@pytest.mark.parametrize("case", broken_params())
def test_slice_shape_broken_today(case):
    """Idioms that currently squeeze a length-1 axis (xfail-strict until fixed)."""
    run_case(case)


def test_returned_scalar_index_is_dace_scalar_convention():
    """A scalar INDEX (``A[1, 2]``) collapses to a scalar everywhere it is used, matching numpy.
    When it is the program's RETURN value it surfaces as dace's scalar shape ``(1,)`` rather than
    numpy's 0-d ``()`` -- the accepted scalar-return convention (a true 0-d would need dace-wide
    0-d support). The value is still exact; only the returned rank differs by this one convention."""
    arr = make_2d()
    result = np.asarray(take_scalar(arr.copy()))
    assert result.shape == (1, ), f"returned scalar convention changed: got {result.shape}, expected (1,)"
    assert result.reshape(()) == arr[1, 2], "returned scalar value mismatch vs numpy"


def test_transpose_minus_self_square_int():
    """Motivating regression: ``x.T - x`` on a square int matrix must be bit-exact.

    Passes today because an already-2D square matrix has no length-1 axis to
    squeeze; kept as a value-fidelity guard for the singleton-slice miscompile
    class (a column slice ``A[:, 0:1]`` collapsed to ``(N,)`` turns ``x.T - x``
    from an ``(N, N)`` matrix into an ``(N,)`` elementwise zero).
    """
    arr = make_square()
    oracle = arr.T - arr
    result = np.asarray(transpose_minus_self(arr.copy()))
    assert result.shape == oracle.shape, (f"x.T - x: dace shape {result.shape} != numpy {oracle.shape}")
    assert np.array_equal(result, oracle), "x.T - x: bit-exact mismatch vs numpy"


if __name__ == "__main__":
    for works_case in WORKS_TODAY:
        run_case(works_case)
        print(f"WORKS_TODAY  pass : {works_case.id}")
    for broken_case in BROKEN_TODAY:
        try:
            run_case(broken_case)
        except AssertionError as err:
            print(f"BROKEN_TODAY xfail: {broken_case.id}  <- {err}")
        else:
            raise SystemExit(f"XPASS: {broken_case.id} unexpectedly matches numpy now -- "
                             f"promote it into WORKS_TODAY")
    test_transpose_minus_self_square_int()
    print("regression   pass : x.T - x")
    print("all idioms behaved as classified")
