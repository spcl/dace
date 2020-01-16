import dace
import numpy as np


@dace.program
def nested_subrange_test(A: dace.float32[2, 3, 4], B: dace.float32[3, 4]):
    for i, j in dace.map[0:3, 0:4]:
        tmp = A[:, i, j]
        B[i, j] = tmp[0] + tmp[1]


@dace.program
def subrange1_test(A: dace.float32[2, 3, 4], B: dace.float32[3, 4]):
    B[:] = A[0, :, :] + A[1, :, :]


@dace.program
def subrange2_test(A: dace.float32[2, 3, 4], B: dace.float32[3, 4]):
    tmp0 = A[0, :, :]
    tmp1 = A[1, :, :]
    for i, j in dace.map[0:3, 0:4]:
        B[i, j] = tmp0[i, j] + tmp1[i, j]


if __name__ == '__main__':
    A = np.random.rand(2, 3, 4).astype(np.float32)
    expected = A[0, :, :] + A[1, :, :]

    B1 = np.random.rand(3, 4).astype(np.float32)
    B2 = np.random.rand(3, 4).astype(np.float32)
    B3 = np.random.rand(3, 4).astype(np.float32)

    sdfg = nested_subrange_test.to_sdfg()
    sdfg(A=A, B=B1)

    sdfg = subrange1_test.to_sdfg()
    sdfg(A=A, B=B2)

    sdfg = subrange2_test.to_sdfg()
    sdfg(A=A, B=B3)

    diff1 = np.linalg.norm(expected - B1)
    diff2 = np.linalg.norm(expected - B2)
    diff3 = np.linalg.norm(expected - B3)
    diffs = [diff1, diff2, diff3]
    print('Difference:', diffs)
    exit(1 if any(d > 1e-5 for d in diffs) else 0)
