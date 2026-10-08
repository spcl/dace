# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reverse-mode differentiation of tasklets whose expressions are wrapped in type casts (``dace.float64(...)``)."""

import numpy as np
import pytest

import dace
from dace.autodiff import add_backward_pass

N = 8


@pytest.mark.parametrize("cast", ("float64", "float32"))
def test_cast_around_compound_expression(cast):
    """The cast must not change the grouping of the expression it wraps (``(a - 1) * (a - 1)``)."""
    dtype = getattr(dace, cast)
    nptype = np.float64 if cast == "float64" else np.float32

    @dace.program
    def squared_shift(x: dtype[N], loss: dtype[1]):
        y = np.ndarray([N], dtype)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                b >> y[i]
                b = dtype(((a - 1.0) * (a - 1.0)))
        loss[0] = np.sum(y)

    sdfg = squared_shift.to_sdfg(simplify=True)
    add_backward_pass(sdfg, outputs=["loss"], inputs=["x"])
    x = np.random.rand(N).astype(nptype)
    gradient = np.zeros(N, nptype)
    sdfg(x=x.copy(), loss=np.zeros(1, nptype), gradient_x=gradient, gradient_loss=np.ones(1, nptype))
    np.testing.assert_allclose(gradient, 2 * (x - 1), rtol=1e-5)


def test_nested_casts():

    @dace.program
    def nested(x: dace.float64[N], loss: dace.float64[1]):
        y = np.ndarray([N], dace.float64)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                b >> y[i]
                b = dace.float64(dace.float64(a * 3.0) * dace.float64((a + 2.0)))
        loss[0] = np.sum(y)

    sdfg = nested.to_sdfg(simplify=True)
    add_backward_pass(sdfg, outputs=["loss"], inputs=["x"])
    x = np.random.rand(N)
    gradient = np.zeros(N)
    sdfg(x=x.copy(), loss=np.zeros(1), gradient_x=gradient, gradient_loss=np.ones(1))
    np.testing.assert_allclose(gradient, 6 * x + 6.0)


if __name__ == "__main__":
    test_cast_around_compound_expression("float64")
    test_cast_around_compound_expression("float32")
    test_nested_casts()
