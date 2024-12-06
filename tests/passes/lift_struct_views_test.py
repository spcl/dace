# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the LiftStructViews pass. """

import dace
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.lift_struct_views import LiftStructViews


def test_simple_tasklet_access():
    sdfg = dace.SDFG('simple_tasklet_access')

    a_struct = dace.data.Structure({
        'b': dace.data.Scalar(dace.int32)
    }, 't_A')
    z_struct = dace.data.Structure({
        'b': dace.data.Scalar(dace.int32)
    }, 't_Z')
    sdfg.add_datadesc('A', a_struct)
    sdfg.add_datadesc('Z', z_struct)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b = i1.b')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 2

    res = LiftStructViews().apply_pass(sdfg, {})

    assert len(res['A']) == 1
    assert len(res['Z']) == 1
    assert len(state.nodes()) == 5
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 4
    assert sdfg.is_valid()


def test_sliced_tasklet_access():
    sdfg = dace.SDFG('sliced_tasklet_access')

    a_struct = dace.data.Structure({
        'b': dace.data.Array(dace.int32, (20,))
    }, 't_A')
    z_struct = dace.data.Structure({
        'b': dace.data.Array(dace.int32, (20,))
    }, 't_Z')
    sdfg.add_datadesc('A', a_struct)
    sdfg.add_datadesc('Z', z_struct)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b[0] = i1.b[0]')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 2

    res = LiftStructViews().apply_pass(sdfg, {})

    assert len(res['A']) == 1
    assert len(res['Z']) == 1
    assert len(state.nodes()) == 5
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 4
    assert sdfg.is_valid()


def test_sliced_multi_tasklet_access():
    sdfg = dace.SDFG('sliced_multi_tasklet_access')

    b_struct = dace.data.Structure({
        'c': dace.data.Scalar(dace.int32)
    }, 't_B')
    a_struct = dace.data.Structure({
        'b': dace.data.ContainerArray(b_struct, (20,))
    }, 't_A')
    y_struct = dace.data.Structure({
        'c': dace.data.Scalar(dace.int32)
    }, 't_Y')
    z_struct = dace.data.Structure({
        'b': dace.data.ContainerArray(y_struct, (20,))
    }, 't_Z')
    sdfg.add_datadesc('A', a_struct)
    sdfg.add_datadesc('Z', z_struct)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b[0].c = i1.b[0].c')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 2

    FixedPointPipeline([LiftStructViews()]).apply_pass(sdfg, {})

    assert len(state.nodes()) == 9
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 8
    assert sdfg.is_valid()


def test_tasklet_access_to_cont_array():
    sdfg = dace.SDFG('sliced_multi_tasklet_access_to_cont_array')

    a_struct = dace.data.Structure({
        'b': dace.data.Scalar(dace.int32)
    }, 't_A')
    z_struct = dace.data.Structure({
        'b': dace.data.Scalar(dace.int32)
    }, 't_Z')
    a_arr = dace.data.ContainerArray(a_struct, (20,))
    z_arr = dace.data.ContainerArray(z_struct, (20,))
    sdfg.add_datadesc('A', a_arr)
    sdfg.add_datadesc('Z', z_arr)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b = i1.b')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A[0]'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z[0]'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 2

    FixedPointPipeline([LiftStructViews()]).apply_pass(sdfg, {})

    assert len(state.nodes()) == 7
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 6
    assert sdfg.is_valid()


def test_sliced_multi_tasklet_access_to_cont_array():
    sdfg = dace.SDFG('sliced_multi_tasklet_access_to_cont_array')

    b_struct = dace.data.Structure({
        'c': dace.data.Scalar(dace.int32)
    }, 't_B')
    a_struct = dace.data.Structure({
        'b': dace.data.ContainerArray(b_struct, (20,))
    }, 't_A')
    y_struct = dace.data.Structure({
        'c': dace.data.Scalar(dace.int32)
    }, 't_Y')
    z_struct = dace.data.Structure({
        'b': dace.data.ContainerArray(b_struct, (20,))
    }, 't_Z')
    a_arr = dace.data.ContainerArray(a_struct, (20,))
    z_arr = dace.data.ContainerArray(z_struct, (20,))
    sdfg.add_datadesc('A', a_arr)
    sdfg.add_datadesc('Z', z_arr)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b[0].c = i1.b[0].c')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A[0]'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z[0]'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 2

    FixedPointPipeline([LiftStructViews()]).apply_pass(sdfg, {})

    assert len(state.nodes()) == 11
    assert len(sdfg.arrays) - len(sdfg.size_arrays()) == 10
    assert sdfg.is_valid()


if __name__ == '__main__':
    test_simple_tasklet_access()
    test_sliced_tasklet_access()
    test_sliced_multi_tasklet_access()
    test_tasklet_access_to_cont_array()
    test_sliced_multi_tasklet_access_to_cont_array()
