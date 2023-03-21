# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.sdfg import InvalidSDFGError


def test_multiple_in_connectors():
    sdfg = dace.SDFG('mctest')
    sdfg.add_array('A', [1], dace.float64)
    state = sdfg.add_state()
    a = state.add_read('A')
    b = state.add_read('A')
    tasklet = state.add_tasklet('dosomething', {'a'}, {}, 'a * a')
    state.add_edge(a, None, tasklet, 'a', dace.Memlet('A[0]'))
    state.add_edge(b, None, tasklet, 'a', dace.Memlet('A[0]'))

    try:
        sdfg.validate()
        raise AssertionError('SDFG validates successfully, test failed!')
    except InvalidSDFGError as ex:
        print('Exception caught, test passed')


def test_multiple_out_connectors():
    sdfg = dace.SDFG('mctest')
    sdfg.add_array('A', [1], dace.float64)
    state = sdfg.add_state()
    a = state.add_write('A')
    b = state.add_write('A')
    tasklet = state.add_tasklet('dosomething', {}, {'a'}, 'a  = 5')
    state.add_edge(tasklet, 'a', a, None, dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'a', b, None, dace.Memlet('A[0]'))

    try:
        sdfg.validate()
        raise AssertionError('SDFG validates successfully, test failed!')
    except InvalidSDFGError:
        print('Exception caught, test passed')


if __name__ == '__main__':
    test_multiple_in_connectors()
    test_multiple_out_connectors()
