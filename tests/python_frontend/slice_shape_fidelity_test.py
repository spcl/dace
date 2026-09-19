# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shape/value fidelity matrix for slice and index idioms through the DaCe Python frontend.
Guards against the length-1-slice squeeze bug: e.g. A[:, 1:2] used to drop to (N,) instead of
numpy's (N, 1). Numpy is the bit-exact oracle; BROKEN_TODAY (empty) holds any idiom that
regresses, marked xfail(strict=True) so a fix shows up as XPASS.
"""
from collections import namedtuple

import numpy as np
import pytest

import dace

Case = namedtuple("Case", ("id", "prog", "oracle", "make_input", "reason"))


def make_2d():
    # Distinct dim lengths (4 rows, 5 cols) so a wrong-axis squeeze cannot hide.
    return np.arange(20, dtype=np.int64).reshape(4, 5)


def make_3d():
    return np.arange(24, dtype=np.int64).reshape(2, 3, 4)


def make_square():
    return np.arange(16, dtype=np.int64).reshape(4, 4)


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
    run_case(case)


@pytest.mark.parametrize("case", broken_params())
def test_slice_shape_broken_today(case):
    run_case(case)


def test_returned_scalar_index_is_dace_scalar_convention():
    """A[1, 2] returned collapses to dace's scalar shape (1,), not numpy's 0-d () -- accepted convention."""
    arr = make_2d()
    result = np.asarray(take_scalar(arr.copy()))
    assert result.shape == (1, ), f"returned scalar convention changed: got {result.shape}, expected (1,)"
    assert result.reshape(()) == arr[1, 2], "returned scalar value mismatch vs numpy"


def test_transpose_minus_self_square_int():
    """x.T - x on a square int matrix: regression guard for the singleton-slice squeeze class."""
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
