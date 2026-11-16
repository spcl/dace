# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Slices whose bounds are arithmetic on data-dependent scalars, e.g., ``A[r0:r0 + 4]``."""

import numpy as np

import dace

N = dace.symbol("N")
M = dace.symbol("M")


def test_slice_of_scalar_offset():
    """Both bounds use the same symbol, so the slice has the size of the target."""

    @dace.program
    def gather_rows(A: dace.float64[N], idx: dace.int64[M], out: dace.float64[M, 4]):
        for i in dace.map[0:M]:
            r0 = idx[i] * 4
            out[i, :] = A[r0 : r0 + 4]

    A = np.random.rand(40)
    idx = np.array([3, 0, 9, 5], dtype=np.int64)
    out = np.zeros((4, 4))
    gather_rows(A=A, idx=idx, out=out)
    assert np.allclose(out, A.reshape(10, 4)[idx])


def test_slice_in_nested_scope():
    """A scalar promoted on one branch of the enclosing scope, and used in a nested scope on the other: the nested
    scope's symbol must not be mistaken for the enclosing scope's."""

    @dace.program
    def branches(A: dace.float64[N], idx: dace.int64[M], flag: dace.int64[M], out: dace.float64[M, 4]):
        for i in dace.map[0:M]:
            r0 = idx[i] * 4
            if flag[i] == 0:
                first = A[r0]
                for j in dace.map[0:4]:
                    out[i, j] = first
            else:
                for _ in dace.map[0:1]:
                    out[i, :] = A[r0 : r0 + 4]

    A = np.random.rand(40)
    idx = np.array([3, 0, 9, 5], dtype=np.int64)
    flag = np.array([0, 1, 1, 0], dtype=np.int64)
    out = np.zeros((4, 4))
    branches(A=A, idx=idx, flag=flag, out=out)
    rows = A.reshape(10, 4)[idx]
    expected = np.where(flag[:, None] == 0, rows[:, :1], rows)
    assert np.allclose(out, expected)


if __name__ == "__main__":
    test_slice_of_scalar_offset()
    test_slice_in_nested_scope()
