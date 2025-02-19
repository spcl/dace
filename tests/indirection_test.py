# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace as dp
import numpy as np


def test():
    W = dp.symbol('W')

    @dp.program
    def indirection(A, x, B):

        @dp.map(_[0:W])
        def ind(i):
            bla << A[x[i]]
            out >> B[i]
            out = bla

    w = 5

    A = np.ndarray([w * w])
    B = np.ndarray([w])
    x = np.ndarray([w], dtype=np.uint32)

    A[:] = np.arange(10, 10 + w * w)
    B[:] = 0
    x[:] = np.random.randint(0, w * w, w)

    indirection(A, x, B, W=w)

    B_ref = np.array([A[x[i]] for i in range(w)], dtype=B.dtype)
    diff = np.linalg.norm(B - B_ref)
    assert diff == 0


def test_two_nested_levels_indirection():
    W = dp.symbol('W')
    H = dp.symbol('H')

    @dp.program
    def indirection(A, x, B):
        for j in dp.map[0:H]:

            @dp.map(_[0:W])
            def ind(i):
                bla << A[x[i]]
                out >> B[i]
                out = bla

    w = h = 5

    A = np.arange(10, 10 + w * w, dtype=np.float64)
    B = np.zeros((w, ), dtype=np.float64)
    x = np.random.randint(0, w * w, w, dtype=np.uint32)

    indirection(A, x, B, W=w, H=h)

    B_ref = np.array([A[x[i]] for i in range(w)], dtype=B.dtype)
    diff = np.linalg.norm(B - B_ref)
    assert diff == 0


def test_multi_dimensional_indirection():
    W = dp.symbol('W')
    H = dp.symbol('H')

    @dp.program
    def indirection(A, x, B):
        for j in dp.map[0:H]:

            @dp.map(_[0:W])
            def ind(i):
                bla << A[x[i, j]]
                out >> B[i, j]
                out = bla

    w = h = 5
                
    A = np.ndarray([w * w])
    B = np.ndarray([w, h])
    x = np.ndarray([w, h], dtype=np.uint32)

    A[:] = np.arange(10, 10 + w * w)
    B[:] = 0
    x[:] = np.random.randint(0, w * w, size=(w, h))

    indirection(A, x, B, W=w, H=h)

    B_ref = np.ndarray((w, h), dtype=B.dtype)
    for i in range(w):
        for j in range(h):
            B_ref[i, j] = A[x[i, j]]
    diff = np.linalg.norm(B - B_ref)
    assert diff == 0


if __name__ == "__main__":
    test()
    test_two_nested_levels_indirection()
    test_multi_dimensional_indirection()
