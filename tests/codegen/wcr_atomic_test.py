# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests atomic WCR detection in code generation. """
import dace

N = dace.symbol('N')


def test_wcr_overlapping_atomic():

    @dace.program
    def tester(A: dace.float32[2 * N + 3]):
        for i in dace.map[0:N]:
            A[2 * i:2 * i + 3] += 1

    sdfg = tester.to_sdfg()
    code: str = sdfg.generate_code()[0].code
    assert code.count('atomic(') == 1


def test_wcr_strided_atomic():

    @dace.program
    def tester(A: dace.float32[2 * N]):
        for i in dace.map[1:N - 1]:
            A[2 * i - 1:2 * i + 2] += 1

    sdfg = tester.to_sdfg()
    code: str = sdfg.generate_code()[0].code
    assert code.count('atomic(') == 1


def test_wcr_strided_nonatomic():

    @dace.program
    def tester(A: dace.float32[2 * N + 3]):
        for i in dace.map[0:N]:
            A[2 * i:2 * i + 2] += 1

    sdfg = tester.to_sdfg()
    code: str = sdfg.generate_code()[0].code
    assert code.count('atomic(') == 0


def test_wcr_strided_nonatomic_offset():

    @dace.program
    def tester(A: dace.float32[2 * N]):
        for i in dace.map[1:N - 1]:
            A[2 * i - 1:2 * i + 1] += 1

    sdfg = tester.to_sdfg()
    code: str = sdfg.generate_code()[0].code
    assert code.count('atomic(') == 0


def _nested_wcr_sdfg(through_view: bool) -> dace.SDFG:
    """A parallel map calls a nested SDFG that accumulates into the same element of a shared container."""
    size = 1000
    inner = dace.SDFG('inner')
    inner.add_array('A', [size], dace.float64)
    state = inner.add_state()
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    if through_view:
        inner.add_view('V', [5], dace.float64)
        view = state.add_access('V')
        state.add_edge(tasklet, 'o', view, None, dace.Memlet('V[0]', wcr='lambda a, b: a + b'))
        state.add_edge(view, 'views', state.add_write('A'), None, dace.Memlet('A[5:10]'))
    else:
        state.add_edge(tasklet, 'o', state.add_write('A'), None, dace.Memlet('A[5]', wcr='lambda a, b: a + b'))

    sdfg = dace.SDFG('wcr_nested_view' if through_view else 'wcr_nested_direct')
    sdfg.add_array('A', [size], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{size}'), schedule=dace.ScheduleType.CPU_Multicore)
    node = state.add_nested_sdfg(inner, {}, {'A'})
    state.add_edge(me, None, node, None, dace.Memlet())
    state.add_memlet_path(node, mx, state.add_write('A'), src_conn='A', memlet=dace.Memlet(f'A[0:{size}]'))
    sdfg.validate()
    return sdfg


def test_wcr_nested_atomic():
    code: str = _nested_wcr_sdfg(through_view=False).generate_code()[0].code
    assert code.count('atomic(') == 1


def test_wcr_through_view_atomic():
    code: str = _nested_wcr_sdfg(through_view=True).generate_code()[0].code
    assert code.count('atomic(') == 1


if __name__ == '__main__':
    test_wcr_overlapping_atomic()
    test_wcr_strided_atomic()
    test_wcr_strided_nonatomic()
    test_wcr_strided_nonatomic_offset()
    test_wcr_nested_atomic()
    test_wcr_through_view_atomic()
