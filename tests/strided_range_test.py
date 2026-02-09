# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import numpy as np


def test_strided_range():
    sr = dace.SDFG('strided_range_test')
    sr.add_array('A', [2, 16, 4], dace.float32)
    sr.add_array('B', [16], dace.float32)

    s0 = sr.add_state('s0')
    A = s0.add_access('A')
    B = s0.add_access('B')
    tasklet = s0.add_tasklet('srtest', {'a'}, {'b'}, """
b[0] = a[0,0] * 2
b[1] = a[0,1] * 2
b[2] = a[1,0] * 2
b[3] = a[1,1] * 2
""")
    me, mx = s0.add_map('srmap', dict(i='0:4'))

    # Reading A at [1,    2i:2i+8:8:2,    3]
    s0.add_memlet_path(A, me, tasklet, dst_conn='a', memlet=Memlet.simple(A, '1, 2*i:2*i+10:8:2, 3'))

    # Writing B at [4*i:4*i+4]
    s0.add_memlet_path(tasklet, mx, B, src_conn='b', memlet=Memlet.simple(B, '4*i:4*i+4'))

    A = np.random.rand(2, 16, 4).astype(np.float32)
    B = np.random.rand(16).astype(np.float32)

    sr(A=A, B=B)

    diffs = [
        B[0:2] - 2 * A[1, 0:2, 3], B[2:4] - 2 * A[1, 8:10, 3], B[4:6] - 2 * A[1, 2:4, 3], B[6:8] - 2 * A[1, 10:12, 3],
        B[8:10] - 2 * A[1, 4:6, 3], B[10:12] - 2 * A[1, 12:14, 3], B[12:14] - 2 * A[1, 6:8, 3],
        B[14:16] - 2 * A[1, 14:16, 3]
    ]
    diff = np.linalg.norm(np.array(diffs))
    assert diff <= 1e-5


def test_strided_view():

    @dace.program
    def padding(a, c):
        b = np.zeros_like(a)
        b[:, 1:-1, 1:-1, :] = c
        return b

    a = np.random.rand(2, 3, 4, 5)
    c = np.random.rand(2, 1, 2, 5)
    result = padding(a, c)
    expected = padding.f(a, c)

    assert np.allclose(result, expected)


def test_strided_view_retval():
    S0, S1, S2, S3 = (dace.symbol(s) for s in ('S0', 'S1', 'S2', 'S3'))

    @dace.program
    def fancy_copy(input: dace.float64[S0, S1, S2, S3]):
        output = np.ndarray((S0, S1, S2, S3), dtype=np.float64)

        for i in range(S1):
            for j in range(S2):
                output[:, i, j, :] = input[:, i, j, :]

        return output

    @dace.program
    def padding(a):
        b = np.zeros((a.shape[0], a.shape[1] + 2, a.shape[2] + 2, a.shape[3]))
        b[:, 1:-1, 1:-1, :] = fancy_copy(a)
        return b

    a = np.random.rand(2, 3, 4, 5)
    result = padding(a)

    expected = np.zeros((2, 5, 6, 5))
    expected[:, 1:-1, 1:-1, :] = a

    assert np.allclose(result, expected)


if __name__ == "__main__":
    test_strided_range()
    test_strided_view()
    test_strided_view_retval()
