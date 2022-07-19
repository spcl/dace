# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace.transformation.dataflow import RedundantArray, RedundantSecondArray


def _make_sdfg_1(succeed: bool = True):

    name = 'success' if succeed else 'failure'
    sdfg = dace.SDFG(f'redundant_array_{name}')
    sdfg.add_array('A', [20], dace.int32)
    sdfg.add_transient('tmp', [7], dace.int32)
    sdfg.add_view('A_0', [8], dace.int32)
    sdfg.add_view('A_1', [7], dace.int32)

    state = sdfg.add_state()

    first_A = state.add_read('A')
    second_A = state.add_write('A')
    first_A_0 = state.add_access('A_0')
    second_A_0 = state.add_access('A_0')

    _, me, mx = state.add_mapped_tasklet('MyMap', {'i': '0:7'}, {'inp': dace.Memlet('A_1[i]')},
                                         'out = 2 * inp', {'out': dace.Memlet('tmp[i]')},
                                         external_edges=True)
    A_1 = state.in_edges(me)[0].src
    tmp = state.out_edges(mx)[0].dst
    # NOTE: View edges must point to the viewed Data, not the Views.
    iset = '11:19' if succeed else '1:9'
    state.add_nedge(first_A, first_A_0, dace.Memlet(data='A', subset=iset, other_subset='0:8'))
    state.add_nedge(first_A_0, A_1, dace.Memlet(data='A_0', subset='1:8', other_subset='0:7'))
    state.add_nedge(tmp, second_A_0, dace.Memlet(data='A_0', subset='0:7', other_subset='0:7'))
    state.add_nedge(second_A_0, second_A, dace.Memlet(data='A', subset='1:9', other_subset='0:8'))

    return sdfg


def test_redundant_array_success():
    sdfg = _make_sdfg_1(succeed=True)
    sdfg.save('test2.sdfg')
    num = sdfg.apply_transformations(RedundantArray)
    assert (num == 1)


def test_redundant_array_failure():
    sdfg = _make_sdfg_1(succeed=False)
    sdfg.save('test2.sdfg')
    num = sdfg.apply_transformations(RedundantArray)
    assert (num == 0)


def _make_sdfg_2(succeed: bool = True):

    name = 'success' if succeed else 'failure'
    sdfg = dace.SDFG(f'redundant_second_array_{name}')
    sdfg.add_array('A', [20], dace.int32)
    sdfg.add_transient('tmp', [7], dace.int32)
    sdfg.add_view('A_0', [8], dace.int32)
    sdfg.add_view('A_1', [7], dace.int32)

    state = sdfg.add_state()

    first_A = state.add_read('A')
    second_A = state.add_write('A')
    first_A_0 = state.add_access('A_0')
    second_A_0 = state.add_access('A_0')

    _, me, mx = state.add_mapped_tasklet('MyMap', {'i': '0:7'}, {'inp': dace.Memlet('tmp[i]')},
                                         'out = 2 * inp', {'out': dace.Memlet('A_1[i]')},
                                         external_edges=True)
    tmp = state.in_edges(me)[0].src
    A_1 = state.out_edges(mx)[0].dst
    # NOTE: View edges must point to the viewed Data, not the Views.
    iset = '11:19' if succeed else '1:9'
    state.add_nedge(first_A, first_A_0, dace.Memlet(data='A', subset=iset, other_subset='0:8'))
    state.add_nedge(first_A_0, tmp, dace.Memlet(data='A_0', subset='1:8', other_subset='0:7'))
    state.add_nedge(A_1, second_A_0, dace.Memlet(data='A_0', subset='0:7', other_subset='0:7'))
    state.add_nedge(second_A_0, second_A, dace.Memlet(data='A', subset='1:9', other_subset='0:8'))

    return sdfg


def test_redundant_second_array_success():
    sdfg = _make_sdfg_2(succeed=True)
    sdfg.save('test2.sdfg')
    num = sdfg.apply_transformations(RedundantSecondArray)
    assert (num == 1)


def test_redundant_second_array_failure():
    sdfg = _make_sdfg_2(succeed=False)
    sdfg.save('test2.sdfg')
    num = sdfg.apply_transformations(RedundantSecondArray)
    assert (num == 0)


if __name__ == '__main__':
    test_redundant_array_success()
    test_redundant_array_failure()
    test_redundant_second_array_success()
    test_redundant_second_array_failure()
