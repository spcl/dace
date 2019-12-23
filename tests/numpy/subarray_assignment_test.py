import numpy as np
import dace


@dace.program
def foo123(a: dace.float32[2, 3], b: dace.float32[2, 3]):
    b[0, :] = a[0, :]


if __name__ == '__main__':

    A = np.full((2, 3), 3, dtype=np.float32)
    B = np.full((2, 3), 4, dtype=np.float32)

    foo123(A, B)

    if not np.allclose(B[0, :], A[0, :]):
        exit(1)
