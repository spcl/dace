# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dc
import numpy as np


@dc.program
def toplevel_scalar_indirection(A: dc.float32[2, 3, 4, 5],
                                B: dc.float32[4]):
    i = 0
    j = 0
    k = 0
    B[:] = A[:, i, :, j][k, :]


def test_toplevel_scalar_indirection():
    A = np.random.rand(2, 3, 4, 5).astype(np.float32)
    B = np.random.rand(4).astype(np.float32)
    toplevel_scalar_indirection(A, B)
    ref = A[0, 0, :, 0]
    assert(np.array_equal(B, ref))


@dc.program
def nested_scalar_indirection(A: dc.float32[2, 3, 4, 5],
                                   B: dc.float32[2, 4]):
    for l in dc.map[0:2]:
        i = 0
        j = 0
        k = l
        B[k] = A[:, i, :, j][k, :]


def test_nested_scalar_indirection():
    A = np.random.rand(2, 3, 4, 5).astype(np.float32)
    B = np.random.rand(2, 4).astype(np.float32)
    nested_scalar_indirection(A, B)
    ref = A[:, 0, :, 0]
    assert(np.array_equal(B, ref))


if __name__ == "__main__":
    test_toplevel_scalar_indirection()
    test_nested_scalar_indirection()
