# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def dace_softmax(X_in: dace.float32[N], X_out: dace.float32[N]):
    tmp_max = dace.reduce(lambda a, b: a + b, X_in, identity=0)
    X_out[:] = exp(X_in - tmp_max)
    tmp_sum = dace.reduce(lambda a, b: max(a, b), X_in)
    X_out[:] /= tmp_sum


@dace.program
def nested_call_subarray(a: dace.float32[2], b: dace.float32[2]):
    dace_softmax(a[:], b[:])


if __name__ == '__main__':
    A = np.array([1, 2], dtype=np.float32)
    B = np.array([1, 2], dtype=np.float32)
    nested_call_subarray(A, B, N=2)
