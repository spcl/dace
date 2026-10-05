# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reverse-mode differentiation of tasklets that read the same array through several connectors (e.g., ``x * x``)."""
import numpy as np

import dace
from dace.autodiff import add_backward_pass

N = 8


def test_square_through_two_connectors():

    @dace.program
    def square_sum(x: dace.float64[N], loss: dace.float64[1]):
        y = np.ndarray([N], dace.float64)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                b << x[i]
                c >> y[i]
                c = a * b
        loss[0] = np.sum(y)

    sdfg = square_sum.to_sdfg(simplify=True)
    add_backward_pass(sdfg, outputs=['loss'], inputs=['x'])
    x = np.random.rand(N)
    gradient = np.zeros(N)
    sdfg(x=x.copy(), loss=np.zeros(1), gradient_x=gradient, gradient_loss=np.ones(1))
    np.testing.assert_allclose(gradient, 2 * x)


def test_intermediate_squared():
    """The backward pass of ``y * y`` needs the forward value of ``y`` twice in the same map."""

    @dace.program
    def squared_intermediate(x: dace.float64[N], loss: dace.float64[1]):
        y = np.sin(x)
        loss[0] = np.sum(y * y)

    sdfg = squared_intermediate.to_sdfg(simplify=True)
    add_backward_pass(sdfg, outputs=['loss'], inputs=['x'])
    x = np.random.rand(N)
    gradient = np.zeros(N)
    sdfg(x=x.copy(), loss=np.zeros(1), gradient_x=gradient, gradient_loss=np.ones(1))
    np.testing.assert_allclose(gradient, 2 * np.sin(x) * np.cos(x))


if __name__ == '__main__':
    test_square_through_two_connectors()
    test_intermediate_squared()
