# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Reproducers: a Python loop over an int64 size symbol whose iterator bounds a slice of an array operation. """
from typing import List

import numpy as np

import dace

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def lu_column_update(A: dace.float64[N, N]):
    for k in range(N):
        A[k + 1:, k] = A[k + 1:, k] / A[k, k]
        A[k + 1:, k + 1:] = A[k + 1:, k + 1:] - A[k + 1:, k][:, None] * A[k, k + 1:][None, :]


@dace.program
def scaled_row_prefix(C: dace.float64[N, N], beta: dace.float64):
    for i in range(N):
        C[i, :i + 1] = C[i, :i + 1] * beta


@dace.program
def forward_substitution(A: dace.float64[N, N], b: dace.float64[N], y: dace.float64[N]):
    for i in range(N):
        y[i] = b[i] - A[i, :i] @ y[:i]


@dace.program
def backward_row_suffix(A: dace.float64[N, N], x: dace.float64[N]):
    for i in range(N - 1, -1, -1):
        A[i, i + 1:] = A[i, i + 1:] + x[i + 1:]


def iterator_dtypes(sdfg: dace.SDFG, name: str) -> List[dace.typeclass]:
    found: List[dace.typeclass] = []
    for sub in sdfg.all_sdfgs_recursive():
        for state in sub.states():
            for edge in state.edges():
                for subset in (edge.data.subset, edge.data.other_subset):
                    if subset is None:
                        continue
                    for bound in (b for rng in subset.ndrange() for b in rng):
                        found.extend(s.dtype for s in getattr(bound, 'free_symbols', ()) if s.name == name)
    return found


def well_conditioned(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, n)) + n * np.eye(n)


def test_lu_column_update_matches_numpy():
    A = well_conditioned(12)
    expected = A.copy()
    for k in range(12):
        expected[k + 1:, k] = expected[k + 1:, k] / expected[k, k]
        expected[k + 1:,
                 k + 1:] = expected[k + 1:, k + 1:] - expected[k + 1:, k][:, None] * expected[k, k + 1:][None, :]
    lu_column_update(A)
    assert np.allclose(A, expected)


def test_scaled_row_prefix_matches_numpy():
    C = well_conditioned(9)
    expected = C.copy()
    for i in range(9):
        expected[i, :i + 1] = expected[i, :i + 1] * 0.5
    scaled_row_prefix(C, 0.5)
    assert np.allclose(C, expected)


def test_forward_substitution_matches_numpy():
    A, b = well_conditioned(10), np.arange(10, dtype=np.float64)
    y, expected = np.zeros(10), np.zeros(10)
    for i in range(10):
        expected[i] = b[i] - A[i, :i] @ expected[:i]
    forward_substitution(A, b, y)
    assert np.allclose(y, expected)


def test_backward_row_suffix_matches_numpy():
    A, x = well_conditioned(8), np.arange(8, dtype=np.float64)
    expected = A.copy()
    for i in range(7, -1, -1):
        expected[i, i + 1:] = expected[i, i + 1:] + x[i + 1:]
    backward_row_suffix(A, x)
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_lu_column_update_matches_numpy()
    test_scaled_row_prefix_matches_numpy()
    test_forward_substitution_matches_numpy()
    test_backward_row_suffix_matches_numpy()
