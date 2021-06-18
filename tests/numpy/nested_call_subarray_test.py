# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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


def test_local_program():
    @dace.program
    def dace_softmax2(X_in: dace.float32[N], X_out: dace.float32[N]):
        tmp_max = dace.reduce(lambda a, b: a + b, X_in, identity=0)
        X_out[:] = exp(X_in - tmp_max)
        tmp_sum = dace.reduce(lambda a, b: max(a, b), X_in)
        X_out[:] /= tmp_sum

    A = np.array([1, 2], dtype=np.float32)
    B = np.array([1, 2], dtype=np.float32)

    @dace.program
    def nested_call_subarray2(a: dace.float32[2], b: dace.float32[2]):
        dace_softmax2(a[:], b[:])

    nested_call_subarray2(A, B, N=2)


def test_nested_call():
    A = np.array([1, 2], dtype=np.float32)
    B = np.array([1, 2], dtype=np.float32)
    nested_call_subarray(A, B, N=2)


if __name__ == '__main__':
    test_nested_call()
    test_local_program()
