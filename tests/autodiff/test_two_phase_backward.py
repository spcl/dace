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
    """Runs both phases, giving each only the arguments it uses, and compares with numerical gradients."""
    two_phase = make_two_phase_backward_pass(program.to_sdfg(simplify=True), outputs=['out'], inputs=['A'])
    sdfg = two_phase.sdfg

    def allocate(name):
        desc = sdfg.arrays[name]
        return np.zeros(tuple(int(dace.symbolic.evaluate(s, {'N': rows})) for s in desc.shape),
                        desc.dtype.as_numpy_dtype())

    def call(phase, used, given):
        arguments = {name: None for name, desc in sdfg.arglist().items() if isinstance(desc, dace.data.Array)}
        arguments.update({name: value for name, value in given.items() if name in used})
        compiled(**arguments, **{two_phase.phase: phase})

    compiled = sdfg.compile()
    a = np.random.rand(rows, 3) * sign
    cotangent = np.random.rand(3)
    given = {'A': a.copy(), 'out': np.zeros(3), 'N': rows, **{name: allocate(name) for name in two_phase.tape}}
    call(FORWARD_PHASE, two_phase.forward_arguments, given)
    np.testing.assert_allclose(given['out'], reference(a))

    gradient = np.zeros_like(a)
    given.update({two_phase.output_gradients['out']: cotangent, two_phase.input_gradients['A']: gradient})
    call(BACKWARD_PHASE, two_phase.backward_arguments, given)
    np.testing.assert_allclose(gradient,
                               _numerical_gradient(lambda x: np.sum(reference(x) * cotangent), a),
                               rtol=1e-5,
                               atol=1e-7)
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

    def reference(a):
        y = np.zeros(3)
        for i in range(a.shape[0]):
            y = np.tanh(y) + a[i]
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

    def reference(a):
        s = np.sum(a)
        return np.tanh(a[0]) * s if s > 0 else a[1] * a[1]

    for sign in (1, -1):  # Both branches
        two_phase = _check(branch, reference, rows=4, sign=sign)
        assert any(name.startswith('tape_') for name in two_phase.tape)


if __name__ == '__main__':
    test_loop()
    test_branch_on_data()
