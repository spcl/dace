import numpy as np

import dace


@dace
def cast_to_long(A: dace.int32[1], B: dace.int64[1]):
    B[:] = dace.elementwise(lambda x: dace.int64(x), A)


def sympy_test():
    A = np.ones((1, )).astype(np.int32)
    B = np.zeros((1, )).astype(np.int64)
    print(B)
    cast_to_long.to_sdfg()(A=A, B=B)


if __name__ == "__main__":
    sympy_test()
