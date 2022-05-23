# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.memlet import Memlet
import numpy as np

sr = dace.SDFG('strided_range_test')
s0 = sr.add_state('s0')

A = s0.add_array('A', [2, 16, 4], dace.float32)
B = s0.add_array('B', [16], dace.float32)
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


def test():
    print('Strided range tasklet test')
    A = np.random.rand(2, 16, 4).astype(np.float32)
    B = np.random.rand(16).astype(np.float32)

    sr(A=A, B=B)

    diffs = [
        B[0:2] - 2 * A[1, 0:2, 3], B[2:4] - 2 * A[1, 8:10, 3], B[4:6] - 2 * A[1, 2:4, 3], B[6:8] - 2 * A[1, 10:12, 3],
        B[8:10] - 2 * A[1, 4:6, 3], B[10:12] - 2 * A[1, 12:14, 3], B[12:14] - 2 * A[1, 6:8, 3],
        B[14:16] - 2 * A[1, 14:16, 3]
    ]
    diff = np.linalg.norm(np.array(diffs))
    print('Differences:', [np.linalg.norm(d) for d in diffs])
    assert diff <= 1e-5


if __name__ == "__main__":
    test()
