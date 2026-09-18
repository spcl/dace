# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import dace

# Try to detect invalid names in SDFG


# SDFG label
def test_sdfg_name1():
    with pytest.raises(dace.sdfg.InvalidSDFGError):
        dace.SDFG(' ')


def test_sdfg_name2():
    with pytest.raises(dace.sdfg.InvalidSDFGError):
        dace.SDFG('3sat')


# State
def test_state_duplication():
    sdfg = dace.SDFG('ok')
    s1 = sdfg.add_state('also_ok')
    s2 = sdfg.add_state('also_ok')
    s2.label = 'also_ok'
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


def test_state_name1():
    sdfg = dace.SDFG('ok')
    sdfg.add_state('not ok')

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


def test_state_name2():
    sdfg = dace.SDFG('ok')
    sdfg.add_state('$5')

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


# Array
def test_array():
    sdfg = dace.SDFG('ok')
    sdfg.add_array('8', [1], dace.float32)

    state = sdfg.add_state('also_ok')
    _8 = state.add_access('8')
    t = state.add_tasklet('tasklet', {'a'}, {}, 'print(a)')
    state.add_edge(_8, None, t, 'a', dace.Memlet.from_array(_8.data, _8.desc(sdfg)))

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


# Tasklet
def test_tasklet():
    sdfg = dace.SDFG('ok')
    sdfg.add_array('A', [1], dace.float32)
    sdfg.add_array('B', [1], dace.float32)

    state = sdfg.add_state('also_ok')
    A = state.add_access('A')
    B = state.add_access('B')
    t = state.add_tasklet(' tasklet', {'a'}, {'b'}, 'b = a')
    state.add_edge(A, None, t, 'a', dace.Memlet.from_array(A.data, A.desc(sdfg)))
    state.add_edge(t, 'b', B, None, dace.Memlet.from_array(B.data, B.desc(sdfg)))

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


# Connector
def test_connector():
    sdfg = dace.SDFG('ok')
    sdfg.add_array('A', [1], dace.float32)
    sdfg.add_array('B', [1], dace.float32)

    state = sdfg.add_state('also_ok')
    A = state.add_access('A')
    B = state.add_access('B')
    t = state.add_tasklet('tasklet', {'$a'}, {' b'}, '')
    state.add_edge(A, None, t, '$a', dace.Memlet.from_array(A.data, A.desc(sdfg)))
    state.add_edge(t, ' b', B, None, dace.Memlet.from_array(B.data, B.desc(sdfg)))

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


# Interstate edge
def test_interstate_edge():
    sdfg = dace.SDFG('ok')
    sdfg.add_array('A', [1], dace.float32)
    sdfg.add_array('B', [1], dace.float32)

    state = sdfg.add_state('also_ok', is_start_block=True)
    A = state.add_access('A')
    B = state.add_access('B')
    t = state.add_tasklet('tasklet', {'a'}, {'b'}, 'b = a')
    state.add_edge(A, None, t, 'a', dace.Memlet.from_array(A.data, A.desc(sdfg)))
    state.add_edge(t, 'b', B, None, dace.Memlet.from_array(B.data, B.desc(sdfg)))
    sdfg.add_edge(state, state, dace.InterstateEdge(assignments={'%5': '1'}))

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


def test_only_arrays_can_be_returned():
    # Only arrays can be returned from a top-level function; a non-array return
    # descriptor (here a Structure) is rejected by validation.
    sdfg = dace.SDFG('ok')
    st = dace.data.Structure(dict(a=dace.float64[1]), name='S')
    sdfg.add_datadesc('__return', st)
    sdfg.add_state()

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


def test_return_naming_single_and_tuple_conflict():
    # A single return value is named `__return`; a tuple (including a one-element
    # tuple) uses `__return_<i>`. Both present at once is ambiguous (single vs
    # tuple) and must be rejected.
    sdfg = dace.SDFG('ok')
    sdfg.add_array('__return', [1], dace.float64)
    sdfg.add_array('__return_0', [1], dace.float64)
    sdfg.add_state()

    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


if __name__ == '__main__':
    test_sdfg_name1()
    test_sdfg_name2()
    test_state_duplication()
    test_state_name1()
    test_state_name2()
    test_array()
    test_tasklet()
    test_connector()
    test_interstate_edge()
    test_only_arrays_can_be_returned()
    test_return_naming_single_and_tuple_conflict()
