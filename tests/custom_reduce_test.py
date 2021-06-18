# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def customreduction(A: dace.float32[20], out: dace.float32[1]):
    dace.reduce(lambda a, b: a if a < b else b, A, out, identity=9999999)


def test_custom_reduce():
    A = np.random.rand(20).astype(np.float32)
    B = np.zeros([1], dtype=np.float32)
    customreduction(A, B)
    diff = (B - np.min(A))
    print('Difference:', diff)
    assert diff <= 1e-5


if __name__ == "__main__":
    test_custom_reduce()
