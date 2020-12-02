# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np

W = dace.symbol('W')
H = dace.symbol('H')


@dace.program
def control_flow_test(A, B, tol):
    if tol[0] < 4:
        while tol[0] < 4:

            @dace.map(_[0:W])
            def something(i):
                a << A[0, i]
                b >> B[0, i]
                t >> tol(1, lambda x, y: x + y)
                b = a
                t = a * a
    elif tol[0] <= 5:

        @dace.map(_[0:W])
        def something(i):
            a << A[0, i]
            b >> B[0, i]
            b = a
    elif tol[0] <= 6:

        @dace.map(_[0:W])
        def something(i):
            a << A[0, i]
            b >> B[0, i]
            b = a
    else:
        for i in range(W):

            @dace.map(_[0:W])
            def something(j):
                a << A[0, j]
                b >> B[0, j]
                b = a


@dace.program
def fictest(A: dace.int32[4]):
    for a in range(min(A[0], A[1])):
        with dace.tasklet:
            inp << A[2]
            out >> A[3]
            out = inp + a


@dace.program
def arr2dtest(A: dace.float64[4, 2]):
    if A[1, 1] < 0.5:
        with dace.tasklet:
            out >> A[0, 0]
            out = 100.0
    else:
        with dace.tasklet:
            out >> A[0, 0]
            out = -100.0


def test_control_flow_basic():
    control_flow_test.compile(dace.float32[W, H], dace.float32[H, W],
                              dace.float32[1])


def test_function_in_condition():
    A = np.random.randint(0, 10, 4, dtype=np.int32)
    expected = A.copy()
    for a in range(min(A[0], A[1])):
        expected[3] = expected[2] + a

    fictest(A)
    assert np.allclose(A, expected)


def test_2d_access():
    print("Running without strict transformations ...")
    A = np.random.rand(4, 2)
    expected = A.copy()
    expected[0, 0] = 100.0 if expected[1, 1] < 0.5 else -100.0

    # arr2dtest(A)
    sdfg = arr2dtest.to_sdfg(strict=False)
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_2d_access_sdfgapi():
    sdfg = dace.SDFG('access2d_sdfg')
    sdfg.add_array('A', [4, 2], dace.float64)
    begin_state = sdfg.add_state()
    state_true = sdfg.add_state()
    state_false = sdfg.add_state()
    state_true.add_edge(
        state_true.add_tasklet('assign', {}, {'a'}, 'a = 100.0'), 'a',
        state_true.add_write('A'), None, dace.Memlet('A[0, 0]'))
    state_false.add_edge(
        state_false.add_tasklet('assign', {}, {'a'}, 'a = -100.0'), 'a',
        state_false.add_write('A'), None, dace.Memlet('A[0, 0]'))

    sdfg.add_edge(begin_state, state_true, dace.InterstateEdge('A[1,1] < 0.5'))
    sdfg.add_edge(begin_state, state_false,
                  dace.InterstateEdge('A[1,1] >= 0.5'))

    # Prepare inputs
    A = np.random.rand(4, 2)
    expected = A.copy()
    expected[0, 0] = 100.0 if expected[1, 1] < 0.5 else -100.0

    # Without control-flow detection
    A1 = A.copy()
    csdfg = sdfg.compile()
    csdfg(A=A1)
    assert np.allclose(A1, expected)
    del csdfg

    # With control-flow detection
    end_state = sdfg.add_state()
    sdfg.add_edge(state_true, end_state, dace.InterstateEdge())
    sdfg.add_edge(state_false, end_state, dace.InterstateEdge())
    assert 'else' in sdfg.generate_code()[0].code

    csdfg = sdfg.compile()
    csdfg(A=A)
    assert np.allclose(A, expected)


def test_2d_assignment():
    sdfg = dace.SDFG('assign2d')
    sdfg.add_array('A', [4, 2], dace.float64)
    state = sdfg.add_state()
    state2 = sdfg.add_state()
    state2.add_edge(state2.add_tasklet('assign', {}, {'a'}, 'a = i'), 'a',
                    state2.add_write('A'), None, dace.Memlet('A[0, 0]'))
    sdfg.add_edge(state, state2,
                  dace.InterstateEdge(assignments=dict(i='A[1, 1]')))

    A = np.random.rand(4, 2)
    sdfg(A=A)
    assert np.allclose(A[0, 0], A[1, 1])


@dace.program
def whiletest(A: dace.int32[1]):
    while A[0] > 0:
        with dace.tasklet:
            a << A[0]
            b >> A[0]
            b = a - 1


def test_while():
    A = dace.ndarray([1], dace.int32)
    A[0] = 5

    whiletest(A)

    assert A[0] == 0


if __name__ == '__main__':
    test_control_flow_basic()
    test_function_in_condition()
    test_2d_access()
    test_2d_access_sdfgapi()
    test_2d_assignment()
    test_while()
