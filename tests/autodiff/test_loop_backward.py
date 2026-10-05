# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Reverse-mode differentiation of loops: while loops that count, loop-carried gradients, and loops under branches."""
import numpy as np
import pytest

import dace
from dace.autodiff import add_backward_pass
from dace.autodiff.base_abc import AutoDiffException

N = dace.symbol('N')
ROWS = 5


def _recurrence(a: np.ndarray, start: int = 0) -> np.ndarray:
    y = np.zeros(3)
    for i in range(start, a.shape[0]):
        y = np.tanh(y) + a[i]
    return y


def _numerical_gradient(f, x, eps=1e-6):
    gradient = np.zeros_like(x)
    for index in np.ndindex(*x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        gradient[index] = (f(plus) - f(minus)) / (2 * eps)
    return gradient


def _check(program, reference, rows: int = ROWS):
    """Differentiates ``out[0]`` of ``program`` with respect to ``A`` and compares with numerical gradients."""
    sdfg = program.to_sdfg(simplify=True)
    add_backward_pass(sdfg, outputs=['out'], inputs=['A'])
    a = np.random.rand(rows, 3)
    gradient = np.zeros_like(a)
    sdfg(A=a.copy(), out=np.zeros(1), gradient_A=gradient, gradient_out=np.ones(1), N=rows)
    np.testing.assert_allclose(gradient, _numerical_gradient(reference, a), rtol=1e-5, atol=1e-7)


@pytest.mark.autodiff
def test_counting_while_loop():
    """A while loop that counts is differentiated as a for loop."""

    @dace.program
    def counting_while(A: dace.float64[N, 3], out: dace.float64[1]):
        y = np.zeros(3)
        i = 0
        while i < N:
            y[:] = np.tanh(y) + A[i]
            i += 1
        out[0] = np.sum(y * y)

    _check(counting_while, lambda a: np.sum(_recurrence(a)**2))


@pytest.mark.autodiff
def test_array_overwritten_in_every_iteration():
    """
    ``last`` is written in every iteration but read after the loop: its gradient is carried between iterations of the
    reversed loop, in a transient that only one state uses.
    """

    @dace.program
    def overwritten(A: dace.float64[N, 3], out: dace.float64[1]):
        y = np.zeros(3)
        last = np.zeros(3)
        for i in range(N):
            y[:] = np.tanh(y) + A[i]
            last[:] = y
        out[0] = np.sum(last * last)

    _check(overwritten, lambda a: np.sum(_recurrence(a)**2))


@pytest.mark.autodiff
def test_loop_carried_copies_in_separate_states():
    """
    A later state of the loop body copies the new value into the array that the next iteration reads, and into the
    array read after the loop. In reverse order, that state is visited before the state that reads the array, so the
    gradient of the carried copy is only found by visiting the loop's states again.
    """

    @dace.program
    def carried(A: dace.float64[N, 3], out: dace.float64[1]):
        current = np.zeros(3)
        following = np.zeros(3)
        last = np.zeros(3)
        for i in range(N):
            following[:] = np.tanh(current) + A[i]
            if i >= 0:  # A separate state
                current[:] = following
                last[:] = following
        out[0] = np.sum(last * last)

    _check(carried, lambda a: np.sum(_recurrence(a)**2))


@pytest.mark.autodiff
def test_loop_under_a_branch():
    """The state after a branch is reached unconditionally, through either the loop or the other branch."""

    @dace.program
    def branch(A: dace.float64[N, 3], out: dace.float64[1]):
        y = np.zeros(3)
        if N > 2:
            for i in range(1, N):
                y[:] = np.tanh(y) + A[i]
        else:
            y[:] = A[0] * 3
        out[0] = np.sum(y * y)

    _check(branch, lambda a: np.sum(_recurrence(a, start=1)**2))
    _check(branch, lambda a: np.sum((a[0] * 3)**2), rows=2)


@pytest.mark.autodiff
def test_loop_that_is_not_a_loop_region_is_rejected():
    """A cycle with two exits (a ``break``) stays unstructured, which the backward pass cannot reverse."""
    sdfg = dace.SDFG('two_exits')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    guard = sdfg.add_state('guard')
    body = sdfg.add_state('body')
    tasklet = body.add_tasklet('accumulate', {'a', 'o'}, {'r'}, 'r = o + a * a')
    body.add_edge(body.add_read('A'), None, tasklet, 'a', dace.Memlet('A[i]'))
    body.add_edge(body.add_read('out'), None, tasklet, 'o', dace.Memlet('out[0]'))
    body.add_edge(tasklet, 'r', body.add_write('out'), None, dace.Memlet('out[0]'))
    after = sdfg.add_state('after')
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={'i': '0'}))
    sdfg.add_edge(guard, body, dace.InterstateEdge('i < N'))
    sdfg.add_edge(guard, after, dace.InterstateEdge('i >= N'))
    sdfg.add_edge(body, guard, dace.InterstateEdge('i < 3', assignments={'i': 'i + 1'}))
    sdfg.add_edge(body, after, dace.InterstateEdge('i >= 3'))
    with pytest.raises(AutoDiffException, match='not a loop region'):
        add_backward_pass(sdfg, outputs=['out'], inputs=['A'])


if __name__ == '__main__':
    test_counting_while_loop()
    test_array_overwritten_in_every_iteration()
    test_loop_carried_copies_in_separate_states()
    test_loop_under_a_branch()
    test_loop_that_is_not_a_loop_region_is_rejected()
