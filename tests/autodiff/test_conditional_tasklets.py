# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reverse-mode differentiation of tasklets with conditional expressions (``a if condition else b``)."""
import numpy as np

import dace
from dace.autodiff import add_backward_pass

N = 8


def _gradients(program, inputs, outputs='loss'):
    """Adds a backward pass for ``loss`` w.r.t. ``inputs`` and returns the gradients for random data."""
    sdfg = program.to_sdfg(simplify=True)
    add_backward_pass(sdfg, outputs=[outputs], inputs=list(inputs))
    arrays = {name: value.copy() for name, value in inputs.items()}
    gradients = {f'gradient_{name}': np.zeros_like(value) for name, value in inputs.items()}
    sdfg(**arrays, **gradients, loss=np.zeros(1), gradient_loss=np.ones(1))
    return {name: gradients[f'gradient_{name}'] for name in inputs}


def test_condition_on_expression():
    """A NaN-propagating ReLU: the condition is an expression of the input, not a connector."""

    @dace.program
    def relu(x: dace.float64[N], loss: dace.float64[1]):
        y = np.ndarray([N], dace.float64)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                b >> y[i]
                b = (a if (a != a) or (a > 0) else 0.0)
        loss[0] = np.sum(y)

    x = np.random.rand(N) - 0.5
    np.testing.assert_allclose(_gradients(relu, {'x': x})['x'], (x > 0).astype(np.float64))


def test_condition_on_two_inputs():
    """``maximum``: the condition reads both inputs, and the gradient goes to the larger one."""

    @dace.program
    def maximum(x: dace.float64[N], z: dace.float64[N], loss: dace.float64[1]):
        y = np.ndarray([N], dace.float64)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                c << z[i]
                b >> y[i]
                b = (a if a > c else c)
        loss[0] = np.sum(y * y)

    x, z = np.random.rand(N), np.random.rand(N)
    grads = _gradients(maximum, {'x': x, 'z': z})
    larger = np.maximum(x, z)
    np.testing.assert_allclose(grads['x'], np.where(x > z, 2 * larger, 0.0))
    np.testing.assert_allclose(grads['z'], np.where(x > z, 0.0, 2 * larger))


def test_cast_around_conditional():
    """A conditional expression wrapped in a cast, as the TorchDynamo frontend emits it."""

    @dace.program
    def scaled(x: dace.float64[N], loss: dace.float64[1]):
        y = np.ndarray([N], dace.float64)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                b >> y[i]
                b = dace.float64((a * 3.0 if a > 0.5 else a * a))
        loss[0] = np.sum(y)

    x = np.random.rand(N)
    np.testing.assert_allclose(_gradients(scaled, {'x': x})['x'], np.where(x > 0.5, 3.0, 2 * x))


def test_if_statement_with_expression():
    """The statement form ``if condition: out = expression`` with a condition that is an expression."""

    @dace.program
    def clipped(x: dace.float64[N], loss: dace.float64[1]):
        y = np.zeros([N], dace.float64)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                b >> y[i]
                if a < 0.5:
                    b = 4.0 * a
        loss[0] = np.sum(y)

    x = np.random.rand(N)
    np.testing.assert_allclose(_gradients(clipped, {'x': x})['x'], np.where(x < 0.5, 4.0, 0.0))


def test_condition_on_connector():
    """The previously supported form, in which the condition is a boolean input connector."""

    @dace.program
    def select(x: dace.float64[N], mask: dace.bool_[N], loss: dace.float64[1]):
        y = np.ndarray([N], dace.float64)
        for i in dace.map[0:N]:
            with dace.tasklet:
                a << x[i]
                m << mask[i]
                b >> y[i]
                b = (a * a if m else 0.0)
        loss[0] = np.sum(y)

    x = np.random.rand(N)
    mask = np.random.rand(N) > 0.5
    sdfg = select.to_sdfg(simplify=True)
    add_backward_pass(sdfg, outputs=['loss'], inputs=['x'])
    gradient = np.zeros(N)
    sdfg(x=x.copy(), mask=mask, loss=np.zeros(1), gradient_x=gradient, gradient_loss=np.ones(1))
    np.testing.assert_allclose(gradient, np.where(mask, 2 * x, 0.0))


if __name__ == '__main__':
    test_condition_on_expression()
    test_condition_on_two_inputs()
    test_cast_around_conditional()
    test_if_statement_with_expression()
    test_condition_on_connector()
