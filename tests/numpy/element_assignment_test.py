# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace


@dace.program
def foo123(a: dace.float32[2], b: dace.float32[2]):
    b[0] = a[0]


def test_elem_assignment():
    A = np.array([1, 2], dtype=np.float32)
    B = np.array([3, 4], dtype=np.float32)

    foo123(A, B)

    assert A[0] == B[0]


@dace.program
def optest(A: dace.float64[5, 5], B: dace.float64[5, 5], C: dace.float64[5, 5]):
    tmp = (-A) * B
    for i, j in dace.map[0:5, 0:5]:
        with dace.tasklet:
            t << tmp[i, j]
            c >> C[i, j]
            c = t


def test_elementwise():
    A = np.random.rand(5, 5)
    B = np.random.rand(5, 5)
    C = np.random.rand(5, 5)

    optest(A, B, C)
    diff = np.linalg.norm(C - ((-A) * B))
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == '__main__':
    test_elem_assignment()
    test_elementwise()
