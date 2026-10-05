# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Separate forward and backward SDFGs (``make_backward_pass``): the forward pass returns what the backward reads."""
import numpy as np
import pytest

import dace
from dace.autodiff import add_backward_pass, make_backward_pass

N = dace.symbol('N')


def _run(backward_pass, arguments, output_gradients, symbols):
    """Runs the forward SDFG, then the backward SDFG on what the forward SDFG forwarded; returns input gradients."""

    def allocate(sdfg, names):
        result = {}
        for name in names:
            desc = sdfg.arrays[name]
            shape = tuple(int(dace.symbolic.evaluate(s, symbols)) for s in desc.shape)
            strides = tuple(int(dace.symbolic.evaluate(s, symbols)) * desc.dtype.bytes for s in desc.strides)
            buffer = np.zeros(int(dace.symbolic.evaluate(desc.total_size, symbols)), desc.dtype.as_numpy_dtype())
            result[name] = np.lib.stride_tricks.as_strided(buffer, shape, strides)
        return result

    # Inputs the backward pass reads are forwarded as themselves; other forwarded data is written by the forward pass
    forwarded = allocate(backward_pass.forward, [name for name in backward_pass.forwarded if name not in arguments])
    forwarded.update({name: arguments[name] for name in backward_pass.forwarded if name in arguments})
    forward_args = {name: value for name, value in arguments.items() if name in backward_pass.forward.arglist()}
    with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
        backward_pass.forward(**{**forward_args, **forwarded}, **symbols)
        gradients = allocate(backward_pass.backward, backward_pass.input_gradients.values())
        backward_args = {backward_pass.forwarded[name]: value for name, value in forwarded.items()}
        backward_args.update({backward_pass.output_gradients[name]: value for name, value in output_gradients.items()})
        backward_args.update({
            name: value
            for name, value in arguments.items()
            if name in backward_pass.backward.arglist() and name not in backward_args
        })
        backward_pass.backward(**backward_args, **gradients, **symbols)
    return {name: gradients[gradient] for name, gradient in backward_pass.input_gradients.items()}


def _numerical_gradient(f, x, eps=1e-6):
    gradient = np.zeros_like(x)
    for index in np.ndindex(*x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        gradient[index] = (f(plus) - f(minus)) / (2 * eps)
    return gradient


@pytest.mark.autodiff
def test_forwarded_transient_view_and_scalar():
    """
    The backward pass reads an intermediate array, a view (a slice of an input), and a scalar of the forward pass;
    all three become outputs of the forward SDFG.
    """

    @dace.program
    def program(A: dace.float64[N, 4], B: dace.float64[4]):
        S = A[1:, :]  # A view the backward pass reads
        T = np.tanh(S)  # An intermediate array
        s = np.sum(B)  # A scalar
        C = T * B * s
        return np.sum(C * C)

    rows = 5
    A = np.random.rand(rows, 4)
    B = np.random.rand(4)

    def reference(a, b):
        c = np.tanh(a[1:, :]) * b * np.sum(b)
        return np.sum(c * c)

    backward_pass = make_backward_pass(program.to_sdfg(simplify=True), outputs=['__return'], inputs=['A', 'B'])
    assert set(backward_pass.forwarded) <= set(backward_pass.forward.arglist())
    # Every argument of the backward SDFG is forwarded, a gradient, or an input of the program
    provided = set(backward_pass.forwarded.values()) | set(backward_pass.input_gradients.values()) | set(
        backward_pass.output_gradients.values()) | {'A', 'B', 'N'}
    assert set(backward_pass.backward.arglist()) <= provided

    gradients = _run(backward_pass, {'A': A.copy(), 'B': B.copy()}, {'__return': np.ones(1)}, {'N': rows})
    np.testing.assert_allclose(gradients['A'], _numerical_gradient(lambda a: reference(a, B), A), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(gradients['B'], _numerical_gradient(lambda b: reference(A, b), B), rtol=1e-5, atol=1e-7)


@pytest.mark.autodiff
def test_add_backward_pass_separate_sdfgs():
    """``add_backward_pass(separate_sdfgs=True)`` makes the intermediates the backward SDFG reads non-transient."""

    @dace.program
    def program(A: dace.float64[N]):
        T = np.exp(A)
        return np.sum(T * T)

    sdfg = program.to_sdfg(simplify=True)
    backward = add_backward_pass(sdfg, outputs=['__return'], inputs=['A'], separate_sdfgs=True)
    for name, desc in backward.arglist().items():
        if not name.startswith('gradient_') and name in sdfg.arrays:
            assert not sdfg.arrays[name].transient, f'the backward pass reads {name}, which the forward pass hides'


@pytest.mark.autodiff
def test_recompute_forward_with_two_outputs():
    """
    With ``recompute_forward``, nothing is forwarded: the backward SDFG differentiates the forward pass again, through
    the vector-Jacobian product with the given output gradients (cotangents).
    """

    @dace.program
    def program(A: dace.float64[N, 3]):
        T = np.tanh(A)
        return T * 2, np.exp(A) + T

    rows = 4
    A = np.random.rand(rows, 3)
    G0, G1 = np.random.rand(rows, 3), np.random.rand(rows, 3)

    def reference(a):
        t = np.tanh(a)
        return np.sum(t * 2 * G0) + np.sum((np.exp(a) + t) * G1)

    backward_pass = make_backward_pass(program.to_sdfg(simplify=True),
                                       outputs=['__return_0', '__return_1'],
                                       inputs=['A'],
                                       recompute_forward=True)
    assert backward_pass.forwarded == {}
    assert set(backward_pass.output_gradients) == {'__return_0', '__return_1'}
    gradients = {name: np.zeros((rows, 3)) for name in backward_pass.input_gradients.values()}
    arguments = {
        'A': A.copy(),
        backward_pass.output_gradients['__return_0']: G0.copy(),
        backward_pass.output_gradients['__return_1']: G1.copy()
    }
    for name, desc in backward_pass.backward.arglist().items():  # Outputs of the recomputed forward pass
        if name not in arguments and name not in gradients and name != 'N':
            arguments[name] = np.zeros(tuple(int(dace.symbolic.evaluate(s, {'N': rows})) for s in desc.shape))
    backward_pass.backward(**arguments, **gradients, N=rows)
    np.testing.assert_allclose(gradients[backward_pass.input_gradients['A']],
                               _numerical_gradient(reference, A),
                               rtol=1e-5,
                               atol=1e-7)


if __name__ == '__main__':
    test_forwarded_transient_view_and_scalar()
    test_add_backward_pass_separate_sdfgs()
    test_recompute_forward_with_two_outputs()
