# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math as mt
import numpy as np


@dace.program
def myprint(input, N, M):

    @dace.tasklet
    def myprint():
        a << input
        for i in range(0, N):
            for j in range(0, M):
                mt.sin(a[i, j])


def test():
    input = dace.ndarray([10, 10], dtype=dace.float32)
    input[:] = np.random.rand(10, 10).astype(dace.float32.type)

    myprint(input, 10, 10)


if __name__ == "__main__":
    test()
