# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

N = dace.symbol('N')


@dace.program
def dace_softmax_ncs(X_in: dace.float32[N], X_out: dace.float32[N]):
    tmp_max = dace.reduce(lambda a, b: a + b, X_in, identity=0)
    X_out[:] = exp(X_in - tmp_max)
    tmp_sum = dace.reduce(lambda a, b: max(a, b), X_in)
    X_out[:] /= tmp_sum


@dace.program
def nested_call_subarray_prog(a: dace.float32[2], b: dace.float32[2]):
    dace_softmax_ncs(a[:], b[:])


def test_ncs_local_program():
    @dace.program
    def dace_softmax_localprog(X_in: dace.float32[N], X_out: dace.float32[N]):
        tmp_max = dace.reduce(lambda a, b: a + b, X_in, identity=0)
        X_out[:] = exp(X_in - tmp_max)
        tmp_sum = dace.reduce(lambda a, b: max(a, b), X_in)
        X_out[:] /= tmp_sum

    A = np.array([1, 2], dtype=np.float32)
    B = np.array([1, 2], dtype=np.float32)

    @dace.program
    def nested_call_subarray_localprog(a: dace.float32[2], b: dace.float32[2]):
        dace_softmax_localprog(a[:], b[:])

    nested_call_subarray_localprog(A, B, N=2)


def test_nested_sa_call():
    A = np.array([1, 2], dtype=np.float32)
    B = np.array([1, 2], dtype=np.float32)
    nested_call_subarray_prog(A, B, N=2)


def test_nested_selfcopy_slice():
    @dace.program
    def nested(q, i, j):
        q[3 + j, 4 + i, 0:3] = q[3 - i + 1, 4 + j, 0:3]

    @dace.program
    def selfcopy(q: dace.float64[128, 128, 80]):
        for i in range(1, 4):
            for j in range(1, 4):
                nested(q, i, j)

    q = np.random.rand(128, 128, 80)
    ref = np.copy(q)
    for i in range(1, 4):
        for j in range(1, 4):
            ref[3 + j, 4 + i, 0:3] = ref[3 - i + 1, 4 + j, 0:3]
    selfcopy(q)
    assert np.allclose(q, ref)


if __name__ == '__main__':
    test_nested_sa_call()
    test_ncs_local_program()
    test_nested_selfcopy_slice()
