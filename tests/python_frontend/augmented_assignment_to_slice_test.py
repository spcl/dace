# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np


def test_augmented_assignment_to_indirect_access():

    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def _test_prog(A: dace.int32[M], ind: dace.int32[N], B: dace.int32[N]):
        return A[ind] + B

    sdfg = _test_prog.to_sdfg()
    assert sdfg.is_valid()


def test_augmented_assignment_to_indirect_access_regression():

    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def _test_prog(A: dace.int32[M], ind: dace.int32[N], B: dace.int32[N]):
        A[ind] += B

    sdfg = _test_prog.to_sdfg()
    assert sdfg.is_valid()

    # Test correctness
    A = np.random.randint(0, 100, 10).astype(np.int32)
    # Create a random permutation so that we don't create duplicate indices (as per NumPy)
    ind = np.copy(np.random.permutation(10).astype(np.int32)[:5])
    B = np.random.randint(0, 100, 5).astype(np.int32)
    A_copy = A.copy()
    A_copy[ind] += B
    _test_prog(A, ind, B)
    assert np.allclose(A, A_copy)


def test_augmented_multidim_indirect_assignment():
    """
    Tests multi-dimensional indirect augmented assignment for broadcasting
    and other potential issues.
    """
    N = dace.symbol('N')
    M = dace.symbol('M')

    @dace.program
    def _test_prog(A: dace.int32[20, M, 6], ind: dace.int32[M, N], B: dace.int32[N]):
        A[2:4, ind[1], 3] += B

    sdfg = _test_prog.to_sdfg()
    assert sdfg.is_valid()

    # Test correctness
    A = np.random.randint(0, 100, size=(20, 10, 6)).astype(np.int32)
    # Create a random permutation so that we don't create duplicate indices (as per NumPy)
    ind = np.zeros((10, 5), dtype=np.int32)
    for i in range(10):
        ind[i] = np.random.permutation(10)[:5]
    B = np.random.randint(0, 100, 5).astype(np.int32)
    A_copy = A.copy()
    A_copy[2:4, ind[1], 3] += B
    _test_prog(A, ind, B)
    assert np.allclose(A, A_copy)


if __name__ == '__main__':
    test_augmented_assignment_to_indirect_access()
    test_augmented_assignment_to_indirect_access_regression()
    test_augmented_multidim_indirect_assignment()
