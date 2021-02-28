# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
from common import compare_numpy_output


M = 10
N = 20


@compare_numpy_output()
def test_reshape(A: dace.float32[N, N]):
    return A.reshape([1, N*N])


if __name__ == "__main__":
    test_reshape()