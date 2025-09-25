# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np

import dace

N = dace.symbol('N')


@dace.program
def dace_sum(X_in: dace.float32[N], X_out: dace.float32[1]):
    dace.reduce(lambda a, b: a + b, X_in, X_out, identity=0)


@dace.program
def dace_max(X_in: dace.float32[N], X_out: dace.float32[1]):
    dace.reduce(lambda a, b: max(a, b), X_in, X_out, identity=-9999999)


@dace.program
def dace_softmax(X_in: dace.float32[N], X_out: dace.float32[N]):
    tmp_max = dace.define_local([1], dtype=dace.float32)
    tmp_sum = dace.define_local([1], dtype=dace.float32)

    dace_max(X_in, tmp_max)

    @dace.map
    def softmax_tasklet_sub(i: _[0:N]):
        x_in << X_in[i]
        x_max << tmp_max
        x_out >> X_out[i]

        x_out = exp(x_in - x_max)

    dace_sum(X_out, tmp_sum)

    @dace.map
    def softmax_tasklet_div(i: _[0:N]):
        x_in << X_out[i]
        x_sum << tmp_sum
        x_out >> X_out[i]

        x_out = x_in / x_sum


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


def test_softmax():
    X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    Y = np.zeros(X.shape, dtype=np.float32)

    dace_softmax.compile()
    dace_softmax(X_in=X, X_out=Y, N=X.shape[0])

    assert np.allclose(softmax(X), Y)


def test_softmax_ufunc():

    @dace.program
    def softmax_ufunc(input: dace.float64[1, 2, 8, 8], output: dace.float64[1, 2, 8, 8]):
        maximum = np.maximum.reduce(input, axis=3, keepdims=True)
        exponent = np.exp(input - maximum)
        sum = np.add.reduce(exponent, axis=3, keepdims=True)
        output[:] = exponent / sum

    a = np.random.rand(1, 2, 8, 8)
    expected = np.random.rand(1, 2, 8, 8)
    b = np.random.rand(1, 2, 8, 8)
    softmax_ufunc.f(a, expected)

    softmax_ufunc(a, b)
    assert np.allclose(b, expected)


if __name__ == "__main__":
    test_softmax()
    test_softmax_ufunc()
