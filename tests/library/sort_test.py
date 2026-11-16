# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``numpy.sort`` and ``numpy.argsort`` along an axis, through the Sort library node, against NumPy's stable sort:
float keys with ties and a NaN (NumPy orders NaN last), both axes of a 2-D array, and the descending top-k
spellings HPCAgent-Bench kernels use (``-np.sort(-x)``, ``np.argsort(-p)[:, :2]``)."""

import numpy as np
import pytest

import dace

N = dace.symbol("N", dtype=dace.int64, positive=True)
F = dace.symbol("F", dtype=dace.int64, positive=True)


@dace.program
def sort_rows(x: dace.float64[N, F], out: dace.float64[N, F]):
    out[:] = np.sort(x, axis=-1)


@dace.program
def sort_columns(x: dace.float64[N, F], out: dace.float64[N, F]):
    out[:] = np.sort(x, axis=0)


@dace.program
def argsort_rows(x: dace.float64[N, F], out: dace.int64[N, F]):
    out[:] = np.argsort(x, axis=1)


@dace.program
def argsort_columns(x: dace.float32[N, F], out: dace.int64[N, F]):
    out[:] = np.argsort(x, axis=0)


@dace.program
def descending_top_two(x: dace.float32[N, F], out: dace.float32[N, F], top: dace.int64[N, 2]):
    out[:] = -np.sort(-x, axis=1)
    top[:] = np.argsort(-x, axis=1)[:, :2]


def keys(dtype) -> np.ndarray:
    """A 5 x 6 array with ties along both axes and one NaN."""
    x = np.array(
        [
            [3.0, 1.0, 2.0, 1.0, 0.5, 9.0],
            [2.0, 2.0, 2.0, -1.0, 7.0, 0.0],
            [np.nan, 4.0, 1.0, 1.0, 3.0, 3.0],
            [0.0, -2.0, 5.0, 5.0, 1.0, 2.0],
            [3.0, 1.0, 2.0, 8.0, 0.5, -9.0],
        ]
    )
    return x.astype(dtype)


@pytest.mark.parametrize(
    "program, axis, out_dtype",
    [(sort_rows, -1, np.float64), (sort_columns, 0, np.float64)],
    ids=["rows", "columns"],
)
def test_sort_matches_numpy(program, axis, out_dtype):
    x = keys(np.float64)
    out = np.zeros(x.shape, out_dtype)
    program(x=x, out=out, N=x.shape[0], F=x.shape[1])
    assert np.array_equal(out, np.sort(x, axis=axis, kind="stable"), equal_nan=True), out


@pytest.mark.parametrize(
    "program, axis, key_dtype",
    [(argsort_rows, 1, np.float64), (argsort_columns, 0, np.float32)],
    ids=["rows", "columns"],
)
def test_argsort_is_numpys_stable_order(program, axis, key_dtype):
    x = keys(key_dtype)
    out = np.zeros(x.shape, np.int64)
    program(x=x, out=out, N=x.shape[0], F=x.shape[1])
    assert np.array_equal(out, np.argsort(x, axis=axis, kind="stable")), out


def test_descending_top_two():
    x = np.nan_to_num(keys(np.float32))
    out = np.zeros(x.shape, np.float32)
    top = np.zeros((x.shape[0], 2), np.int64)
    descending_top_two(x=x, out=out, top=top, N=x.shape[0], F=x.shape[1])
    assert np.array_equal(out, -np.sort(-x, axis=1, kind="stable")), out
    assert np.array_equal(top, np.argsort(-x, axis=1, kind="stable")[:, :2]), top


if __name__ == "__main__":
    test_sort_matches_numpy(sort_rows, -1, np.float64)
    test_sort_matches_numpy(sort_columns, 0, np.float64)
    test_argsort_is_numpys_stable_order(argsort_rows, 1, np.float64)
    test_argsort_is_numpys_stable_order(argsort_columns, 0, np.float32)
    test_descending_top_two()
