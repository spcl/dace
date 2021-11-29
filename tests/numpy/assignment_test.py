# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from common import compare_numpy_output


def test_multiassign():
    @dace.program
    def multiassign(A: dace.float64[20], B: dace.float64[1],
                    C: dace.float64[2]):
        tmp = C[0] = A[5]
        B[0] = tmp

    A = np.random.rand(20)
    B = np.random.rand(1)
    C = np.random.rand(2)
    multiassign(A, B, C)
    assert B == C[0] and C[0] == A[5]


def test_multiassign_mutable():
    @dace.program
    def mutable(D: dace.float64[2]):
        D[0] += 1
        return D[0]

    @dace.program
    def multiassign(B: dace.float64[1], C: dace.float64[2]):
        tmp = C[1] = mutable(C)
        B[0] = tmp

    B = np.random.rand(1)
    C = np.random.rand(2)
    expected = C[0] + 1
    multiassign(B, C)
    assert B[0] == expected and C[1] == expected


@compare_numpy_output(positive=True)
def test_assign(A: dace.float32[3, 5], B: dace.float32[2, 2]):
    A[1:3, 2:4] = B
    return A


@compare_numpy_output(positive=True)
def test_scalar(A: dace.float32[3, 5], B: dace.float32):
    A[:] = B
    return A


@compare_numpy_output(positive=True)
def test_simple(A: dace.float32[3, 5], B: dace.float32[3, 5]):
    A[:] = B
    return A


@compare_numpy_output(positive=True)
def test_broadcast(A: dace.float32[3, 5], B: dace.float32[3, 1]):
    A[:] = B
    return A


@compare_numpy_output(positive=True)
def test_broadcast2(A: dace.float32[3, 5], B: dace.float32[5]):
    A[:] = B
    return A


@compare_numpy_output(positive=True)
def test_broadcast3(A: dace.float32[3, 5], B: dace.float32[1]):
    A[:] = B
    return A


@compare_numpy_output(positive=True)
def test_broadcast4(A: dace.float32[3, 5], B: dace.float32[2, 1]):
    A[1:3, :] = B
    return A


@compare_numpy_output(positive=True)
def test_broadcast5(A: dace.float32[3, 5], B: dace.float32[2]):
    A[1:3, 2:4] = B
    return A


@compare_numpy_output(positive=True)
def test_assign_wild(A: dace.float32[3, 5, 10, 13], B: dace.float32[2, 1, 4]):
    A[2, 2:4, :, 8:12] = B
    return A


@compare_numpy_output(positive=True)
def test_assign_squeezed(A: dace.float32[3, 5, 10, 20, 13],
                         B: dace.float32[2, 1, 4]):
    A[2, 2:4, :, 1, 8:12] = B
    return A


def test_annotated_assign_type():
    @dace.program
    def annassign(a: dace.float64[20], t: dace.int64):
        b: dace.float64
        for i in dace.map[0:t]:
            b = t
            a[i] = b

    # Test types
    sdfg = annassign.to_sdfg()
    assert 't' not in sdfg.symbols or sdfg.symbols['t'] == dace.int64
    b = next(arr for _, name, arr in sdfg.arrays_recursive() if name == 'b')
    assert b.dtype == dace.float64

    # Test program correctness
    a = np.random.rand(20)
    t = 5
    sdfg(a, t)
    assert np.allclose(a[0:5], t)
    assert not np.allclose(a[5:], t)


if __name__ == '__main__':
    test_multiassign()
    test_multiassign_mutable()
    test_assign()
    test_multiassign()
    test_multiassign_mutable()
    test_scalar()
    test_simple()
    test_broadcast()
    test_broadcast2()
    test_broadcast3()
    test_broadcast4()
    test_broadcast5()
    test_assign_wild()
    test_annotated_assign_type()