import dace
import numpy as np


@dace.program
def augoptest(A: dace.float64[5, 5], B: dace.float64[5, 5]):
    B += A


if __name__ == '__main__':
    A = np.random.rand(5, 5)
    B = np.random.rand(5, 5)
    origB = B.copy()

    augoptest(A, B)
    diff = np.linalg.norm(B - (A + B))
    print('Difference:', diff)
    if diff > 1e-5:
        exit(1)
