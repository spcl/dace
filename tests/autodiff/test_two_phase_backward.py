# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
One SDFG with a forward and a backward phase (``make_two_phase_backward_pass``): the forward phase records a tape of
what the backward phase reads, which the caller keeps between the two calls.
"""
import numpy as np
import pytest

import dace
from dace.autodiff import BACKWARD_PHASE, FORWARD_PHASE, make_two_phase_backward_pass

N = dace.symbol('N')


def _numerical_gradient(f, x, eps=1e-6):
    gradient = np.zeros_like(x)
    for index in np.ndindex(*x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        gradient[index] = (f(plus) - f(minus)) / (2 * eps)
    return gradient


def _check(program, reference, rows: int, sign: float = 1):
    """Runs both phases on ``A`` (``rows`` x 3), and compares the gradient of ``A`` with numerical gradients."""
    a = np.random.rand(rows, 3) * sign
    return _check_gradients(program, reference, {'A': a}, {'N': rows})


def _check_gradients(program, reference, inputs, symbols):
    """
    Runs both phases, giving each only the arguments it uses, and compares the output (``out``) with
    ``reference(**inputs)`` and the gradients of all inputs with numerical gradients.
    """
    two_phase = make_two_phase_backward_pass(program.to_sdfg(simplify=True), outputs=['out'], inputs=list(inputs))
    sdfg = two_phase.sdfg

    def allocate(name):
        desc = sdfg.arrays[name]
        return np.zeros(tuple(int(dace.symbolic.evaluate(s, symbols)) for s in desc.shape), desc.dtype.as_numpy_dtype())

    def call(phase, used, given):
        arguments = {name: None for name, desc in sdfg.arglist().items() if isinstance(desc, dace.data.Array)}
        arguments.update({name: value for name, value in given.items() if name in used})
        compiled(**arguments, **{two_phase.phase: phase})

    compiled = sdfg.compile()
    expected = reference(**inputs)
    cotangent = np.random.rand(*expected.shape)
    given = {**{name: value.copy() for name, value in inputs.items()}, 'out': np.zeros_like(expected), **symbols}
    given.update({name: allocate(name) for name in two_phase.tape})
    call(FORWARD_PHASE, two_phase.forward_arguments, given)
    np.testing.assert_allclose(given['out'], expected)

    gradients = {name: np.zeros_like(value) for name, value in inputs.items()}
    given.update({two_phase.output_gradients['out']: cotangent})
    given.update({two_phase.input_gradients[name]: gradient for name, gradient in gradients.items()})
    call(BACKWARD_PHASE, two_phase.backward_arguments, given)
    for name, value in inputs.items():

        def f(x):
            return np.sum(reference(**{**inputs, name: x}) * cotangent)

        np.testing.assert_allclose(gradients[name], _numerical_gradient(f, value), rtol=1e-5, atol=1e-7, err_msg=name)
    return two_phase


@pytest.mark.autodiff
def test_loop():
    """Values that the loop overwrites are stored per iteration in the forward phase."""

    @dace.program
    def loop(A: dace.float64[N, 3], out: dace.float64[3]):
        y = np.zeros(3)
        for i in range(N):
            y[:] = np.tanh(y) + A[i]
        out[:] = y * y

    def reference(A):
        y = np.zeros(3)
        for i in range(A.shape[0]):
            y = np.tanh(y) + A[i]
        return y * y

    two_phase = _check(loop, reference, rows=6)
    assert 'A' not in two_phase.backward_arguments  # The backward phase reads the stored values instead
    assert all(name in two_phase.forward_arguments and name in two_phase.backward_arguments for name in two_phase.tape)


@pytest.mark.autodiff
def test_branch_on_data():
    """The scalar that the branch decides on is on the tape: the backward phase takes the branch the forward took."""

    @dace.program
    def branch(A: dace.float64[N, 3], out: dace.float64[3]):
        s = np.sum(A)
        if s > 0:
            out[:] = np.tanh(A[0]) * s
        else:
            out[:] = A[1] * A[1]

    def reference(A):
        s = np.sum(A)
        return np.tanh(A[0]) * s if s > 0 else A[1] * A[1]

    for sign in (1, -1):  # Both branches
        two_phase = _check(branch, reference, rows=4, sign=sign)
        assert any(name.startswith('tape_') for name in two_phase.tape)


@pytest.mark.autodiff
@pytest.mark.parametrize('scale, expected_trips', [(0.1, 0), (0.9, None), (3.0, 6)])
def test_loop_with_data_dependent_exit(scale, expected_trips):
    """
    ``while k < N and s > 1``: a for loop bounded by ``N`` that may exit earlier, depending on data. The forward phase
    counts the iterations, the values it stores per iteration are sized by the bound, and the backward phase reverses
    the iterations that ran.
    """

    @dace.program
    def damped(A: dace.float64[3], B: dace.float64[N, 3], out: dace.float64[3]):
        h = np.copy(A)
        s = np.sum(h * h)
        k = 0
        while k < N and s > 1.0:
            h[:] = np.tanh(h) * 0.9 + B[k] * 0.1
            s = np.sum(h * h)
            k += 1
        out[:] = h * h

    def run(A, B):
        h, k = A.copy(), 0
        while k < B.shape[0] and np.sum(h * h) > 1.0:
            h = np.tanh(h) * 0.9 + B[k] * 0.1
            k += 1
        return h * h, k

    rows = 6
    a = np.array([1.0, 0.7, 1.1]) * scale
    b = np.full((rows, 3), 3.0) if expected_trips == rows else np.random.rand(rows, 3) * 0.1
    trips = run(a, b)[1]
    assert trips == expected_trips if expected_trips is not None else 0 < trips < rows
    _check_gradients(damped, lambda A, B: run(A, B)[0], {'A': a, 'B': b}, {'N': rows})


if __name__ == '__main__':
    test_loop()
    test_branch_on_data()
    test_loop_with_data_dependent_exit(0.9, None)
