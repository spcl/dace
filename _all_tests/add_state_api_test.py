# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


@dace.program
def addstate(A: dace.float64[10]):
    if A[0] < 0.5:
        for i in range(5):
            A[i] *= 2
    else:
        for i in range(5, 10):
            A[i] *= 2


def _configure():
    A = np.random.rand(10)
    expected = A.copy()
    if A[0] < 0.5:
        expected[0:5] *= 2
    else:
        expected[5:10] *= 2
    sdfg = addstate.to_sdfg()
    return sdfg, A, expected


def test_state_before():
    sdfg, A, expected = _configure()
    old_states = list(sdfg.nodes())
    for state in old_states:
        sdfg.add_state_before(state)

    assert sdfg.number_of_nodes() == 2 * len(old_states)
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_state_after():
    sdfg, A, expected = _configure()
    old_states = list(sdfg.nodes())
    for state in old_states:
        sdfg.add_state_after(state)

    assert sdfg.number_of_nodes() == 2 * len(old_states)
    sdfg(A=A)
    assert np.allclose(A, expected)


if __name__ == '__main__':
    test_state_before()
    test_state_after()
