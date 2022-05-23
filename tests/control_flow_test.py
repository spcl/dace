# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
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
    control_flow_test.compile(dace.float32[W, H], dace.float32[H, W], dace.float32[1])


def test_function_in_condition():
    A = np.random.randint(0, 10, 4, dtype=np.int32)
    expected = A.copy()
    for a in range(min(A[0], A[1])):
        expected[3] = expected[2] + a

    fictest(A)
    assert np.allclose(A, expected)


def test_2d_access():
    print("Running without simplification...")
    A = np.random.rand(4, 2)
    expected = A.copy()
    expected[0, 0] = 100.0 if expected[1, 1] < 0.5 else -100.0

    # arr2dtest(A)
    sdfg = arr2dtest.to_sdfg(simplify=False)
    sdfg(A=A)
    assert np.allclose(A, expected)


def test_2d_access_sdfgapi():
    sdfg = dace.SDFG('access2d_sdfg')
    sdfg.add_array('A', [4, 2], dace.float64)
    begin_state = sdfg.add_state()
    state_true = sdfg.add_state()
    state_false = sdfg.add_state()
    state_true.add_edge(state_true.add_tasklet('assign', {}, {'a'}, 'a = 100.0'), 'a', state_true.add_write('A'), None,
                        dace.Memlet('A[0, 0]'))
    state_false.add_edge(state_false.add_tasklet('assign', {}, {'a'}, 'a = -100.0'), 'a', state_false.add_write('A'),
                         None, dace.Memlet('A[0, 0]'))

    sdfg.add_edge(begin_state, state_true, dace.InterstateEdge('A[1,1] < 0.5'))
    sdfg.add_edge(begin_state, state_false, dace.InterstateEdge('A[1,1] >= 0.5'))

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
    state2.add_edge(state2.add_tasklet('assign', {}, {'a'}, 'a = i'), 'a', state2.add_write('A'), None,
                    dace.Memlet('A[0, 0]'))
    sdfg.add_edge(state, state2, dace.InterstateEdge(assignments=dict(i='A[1, 1]')))

    A = np.random.rand(4, 2)
    sdfg(A=A)
    assert np.allclose(A[0, 0], A[1, 1])


def test_while_symbol():
    @dace.program
    def whiletest_symbol(A: dace.int32[1]):
        i = 6
        while i > 0:
            A[0] -= 1
            i -= i

    A = dace.ndarray([1], dace.int32)
    A[0] = 5

    whiletest_symbol(A)

    assert A[0] == 4

    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        code = whiletest_symbol.to_sdfg().generate_code()[0].clean_code
        assert 'while ' in code
        assert 'goto ' not in code


def test_while_data():
    @dace.program
    def whiletest_data(A: dace.int32[1]):
        while A[0] > 0:
            with dace.tasklet:
                a << A[0]
                b >> A[0]
                b = a - 1

    A = dace.ndarray([1], dace.int32)
    A[0] = 5

    whiletest_data(A)

    assert A[0] == 0

    # Disable check due to CFG generation in Python frontend
    # if dace.Config.get_bool('optimizer', 'detect_control_flow'):
    #     code = whiletest_data.to_sdfg().generate_code()[0].clean_code
    #     assert 'while ' in code


def test_dowhile():
    sdfg = dace.SDFG('dowhiletest')
    sdfg.add_array('A', [1], dace.int32)
    init = sdfg.add_state()
    state1 = sdfg.add_state()
    sdfg.add_edge(init, state1, dace.InterstateEdge(assignments={'cond': '1'}))
    state2 = sdfg.add_state()
    sdfg.add_edge(state1, state2, dace.InterstateEdge(assignments={'cond': 'cond + 1'}))
    guard = sdfg.add_state_after(state2)
    after = sdfg.add_state()
    sdfg.add_edge(guard, state1, dace.InterstateEdge('cond < 5'))
    sdfg.add_edge(guard, after, dace.InterstateEdge('cond >= 5'))

    t = state1.add_tasklet('something', {'a'}, {'o'}, 'o = a + 1')
    r = state1.add_read('A')
    w = state1.add_write('A')
    state1.add_edge(r, None, t, 'a', dace.Memlet('A'))
    state1.add_edge(t, 'o', w, None, dace.Memlet('A'))

    A = np.zeros([1], dtype=np.int32)
    sdfg(A=A)
    assert A[0] == 4

    # TODO: Not yet available
    # if dace.Config.get_bool('optimizer', 'detect_control_flow'):
    #     code = sdfg.generate_code()[0].clean_code
    #     assert 'do {' in code and '} while' in code


def test_ifchain():
    @dace.program
    def ifchain_program(A: dace.int32[2]):
        if A[0] == 0:
            A[1] = 5
        elif A[0] == 1:
            A[1] = 3
        elif A[0] == 3:
            A[1] = 1
        elif A[0] == 5:
            A[1] = 0

    sdfg: dace.SDFG = ifchain_program.to_sdfg()
    A = np.array([3, 0], dtype=np.int32)
    sdfg(A=A)
    assert A[1] == 1

    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        code = sdfg.generate_code()[0].clean_code
        assert 'else ' in code


def test_ifchain_manual():
    sdfg = dace.SDFG('ifchain')
    sdfg.add_array('A', [2], dace.int32)
    init = sdfg.add_state()
    case0 = sdfg.add_state()
    case1 = sdfg.add_state()
    case3 = sdfg.add_state()
    case5 = sdfg.add_state()
    end = sdfg.add_state()
    for case, state in [(0, case0), (1, case1), (3, case3), (5, case5)]:
        if case == 5:
            sdfg.add_edge(init, state, dace.InterstateEdge(f'A[0] >= {case}'))
        else:
            sdfg.add_edge(init, state, dace.InterstateEdge(f'A[0] == {case}'))
        t = state.add_tasklet('update', {}, {'a'}, f'a = {case}')
        w = state.add_write('A')
        state.add_edge(t, 'a', w, None, dace.Memlet('A[1]'))
        sdfg.add_edge(state, end, dace.InterstateEdge())

    A = np.array([6, 0], dtype=np.int32)
    sdfg(A=A)
    assert A[1] == 5

    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        code = sdfg.generate_code()[0].clean_code
        assert 'else if' in code


def test_switchcase():
    sdfg = dace.SDFG('switchcase')
    sdfg.add_array('A', [2], dace.int32)
    init = sdfg.add_state()
    case0 = sdfg.add_state()
    case1 = sdfg.add_state()
    case3 = sdfg.add_state()
    case5 = sdfg.add_state()
    end = sdfg.add_state()
    for case, state in [(0, case0), (1, case1), (3, case3), (5, case5)]:
        if case == 3:
            sdfg.add_edge(init, state, dace.InterstateEdge(f'{case} == A[0]'))
        else:
            sdfg.add_edge(init, state, dace.InterstateEdge(f'A[0] == {case}'))
        t = state.add_tasklet('update', {}, {'a'}, f'a = {case}')
        w = state.add_write('A')
        state.add_edge(t, 'a', w, None, dace.Memlet('A[1]'))
        sdfg.add_edge(state, end, dace.InterstateEdge())

    A = np.array([3, 0], dtype=np.int32)
    sdfg(A=A)
    assert A[1] == 3

    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        code = sdfg.generate_code()[0].clean_code
        assert 'switch ' in code


def test_fsm():
    # Could be interpreted as a while loop of a switch-case
    sdfg = dace.SDFG('fsmtest')
    sdfg.add_scalar('nextstate', dace.int32)
    sdfg.add_array('A', [1], dace.int32)
    start = sdfg.add_state()
    init = sdfg.add_state_after(start)
    case0 = sdfg.add_state()
    case1 = sdfg.add_state()
    case3 = sdfg.add_state()
    case5 = sdfg.add_state()
    estate = sdfg.add_state()

    # State transitions
    fsm = {0: 3, 3: 1, 1: 5, 5: 7}

    for case, state in [(0, case0), (1, case1), (3, case3), (5, case5)]:
        sdfg.add_edge(init, state, dace.InterstateEdge(f'nextstate == {case}'))

        r = state.add_read('A')
        t = state.add_tasklet('update', {'ain'}, {'a', 'nstate'}, f'a = ain + {case}; nstate = {fsm[case]}')
        w = state.add_write('A')
        ws = state.add_write('nextstate')
        state.add_edge(r, None, t, 'ain', dace.Memlet('A'))
        state.add_edge(t, 'a', w, None, dace.Memlet('A'))
        state.add_edge(t, 'nstate', ws, None, dace.Memlet('nextstate'))

        sdfg.add_edge(state, estate, dace.InterstateEdge())
    sdfg.add_edge(estate, init, dace.InterstateEdge())

    A = np.array([1], dtype=np.int32)
    sdfg(A=A, nextstate=0)
    assert A[0] == 1 + 3 + 1 + 5

    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        code = sdfg.generate_code()[0].clean_code
        assert 'switch ' in code


def test_nested_loop_detection():
    @dace.program
    def nestedloop(A: dace.float64[1]):
        for i in range(5):
            for j in range(5):
                A[0] += i + j

    if dace.Config.get_bool('optimizer', 'detect_control_flow'):
        code = nestedloop.to_sdfg().generate_code()[0].clean_code
        assert code.count('for ') == 2

    a = np.random.rand(1)
    expected = np.copy(a)
    nestedloop.f(expected)

    nestedloop(a)
    assert np.allclose(a, expected)


if __name__ == '__main__':
    test_control_flow_basic()
    test_function_in_condition()
    test_2d_access()
    test_2d_access_sdfgapi()
    test_2d_assignment()
    test_while_symbol()
    test_while_data()
    test_dowhile()
    test_ifchain()
    test_ifchain_manual()
    test_switchcase()
    test_fsm()
    test_nested_loop_detection()
