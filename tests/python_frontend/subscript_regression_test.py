# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import dace


@dace.program
def dace_func(
    X: dace.float32[4, 5],
    Y: dace.float32[4, 3],
    W: dace.float32[4, 3],
    S: dace.float32[1],
):
    Xt = np.transpose(X)
    YW = W * Y
    Z = Xt @ YW

    @dace.map(_[0:5, 0:3])
    def summap(i, j):
        s >> S(1, lambda x, y: x + y)[0]
        z << Z[i, j]
        s = log(z + 1)


def test_regression():
    dace_func.to_sdfg()


if __name__ == "__main__":
    test_regression()
