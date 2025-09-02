# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
""" Tests the LiftStructViews pass. """

import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.pass_pipeline import FixedPointPipeline
from dace.transformation.passes.lift_struct_views import LiftStructViews


def test_simple_tasklet_access():
    sdfg = dace.SDFG('simple_tasklet_access')

    a_struct = dace.data.Structure({'b': dace.data.Scalar(dace.int32)}, 't_A')
    z_struct = dace.data.Structure({'b': dace.data.Scalar(dace.int32)}, 't_Z')
    sdfg.add_datadesc('A', a_struct)
    sdfg.add_datadesc('Z', z_struct)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b = i1.b')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) == 2

    res = LiftStructViews().apply_pass(sdfg, {})

    assert len(res['A']) == 1
    assert len(res['Z']) == 1
    assert len(state.nodes()) == 5
    assert len(sdfg.arrays) == 4
    assert sdfg.is_valid()


def test_sliced_tasklet_access():
    sdfg = dace.SDFG('sliced_tasklet_access')

    a_struct = dace.data.Structure({'b': dace.data.Array(dace.int32, (20, ))}, 't_A')
    z_struct = dace.data.Structure({'b': dace.data.Array(dace.int32, (20, ))}, 't_Z')
    sdfg.add_datadesc('A', a_struct)
    sdfg.add_datadesc('Z', z_struct)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b[0] = i1.b[0]')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) == 2

    res = LiftStructViews().apply_pass(sdfg, {})

    assert len(res['A']) == 1
    assert len(res['Z']) == 1
    assert len(state.nodes()) == 5
    assert len(sdfg.arrays) == 4
    assert sdfg.is_valid()


def test_sliced_multi_tasklet_access():
    sdfg = dace.SDFG('sliced_multi_tasklet_access')

    b_struct = dace.data.Structure({'c': dace.data.Scalar(dace.int32)}, 't_B')
    a_struct = dace.data.Structure({'b': dace.data.ContainerArray(b_struct, (20, ))}, 't_A')
    y_struct = dace.data.Structure({'c': dace.data.Scalar(dace.int32)}, 't_Y')
    z_struct = dace.data.Structure({'b': dace.data.ContainerArray(y_struct, (20, ))}, 't_Z')
    sdfg.add_datadesc('A', a_struct)
    sdfg.add_datadesc('Z', z_struct)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b[0].c = i1.b[0].c')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) == 2

    FixedPointPipeline([LiftStructViews()]).apply_pass(sdfg, {})

    assert len(state.nodes()) == 9
    assert len(sdfg.arrays) == 8
    assert sdfg.is_valid()


def test_tasklet_access_to_cont_array():
    sdfg = dace.SDFG('sliced_multi_tasklet_access_to_cont_array')

    a_struct = dace.data.Structure({'b': dace.data.Scalar(dace.int32)}, 't_A')
    z_struct = dace.data.Structure({'b': dace.data.Scalar(dace.int32)}, 't_Z')
    a_arr = dace.data.ContainerArray(a_struct, (20, ))
    z_arr = dace.data.ContainerArray(z_struct, (20, ))
    sdfg.add_datadesc('A', a_arr)
    sdfg.add_datadesc('Z', z_arr)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b = i1.b')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A[0]'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z[0]'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) == 2

    FixedPointPipeline([LiftStructViews()]).apply_pass(sdfg, {})

    assert len(state.nodes()) == 7
    assert len(sdfg.arrays) == 6
    assert sdfg.is_valid()


def test_sliced_multi_tasklet_access_to_cont_array():
    sdfg = dace.SDFG('sliced_multi_tasklet_access_to_cont_array')

    b_struct = dace.data.Structure({'c': dace.data.Scalar(dace.int32)}, 't_B')
    a_struct = dace.data.Structure({'b': dace.data.ContainerArray(b_struct, (20, ))}, 't_A')
    y_struct = dace.data.Structure({'c': dace.data.Scalar(dace.int32)}, 't_Y')
    z_struct = dace.data.Structure({'b': dace.data.ContainerArray(b_struct, (20, ))}, 't_Z')
    a_arr = dace.data.ContainerArray(a_struct, (20, ))
    z_arr = dace.data.ContainerArray(z_struct, (20, ))
    sdfg.add_datadesc('A', a_arr)
    sdfg.add_datadesc('Z', z_arr)
    state = sdfg.add_state('state', is_start_block=True)
    t1 = state.add_tasklet('t1', {'i1'}, {'o1'}, 'o1.b[0].c = i1.b[0].c')
    read = state.add_access('A')
    write = state.add_access('Z')
    state.add_edge(read, None, t1, 'i1', dace.Memlet('A[0]'))
    state.add_edge(t1, 'o1', write, None, dace.Memlet('Z[0]'))

    assert len(state.nodes()) == 3
    assert len(sdfg.arrays) == 2

    FixedPointPipeline([LiftStructViews()]).apply_pass(sdfg, {})

    assert len(state.nodes()) == 11
    assert len(sdfg.arrays) == 10
    assert sdfg.is_valid()


def test_lift_in_loop_meta_code():
    sdfg = dace.SDFG('lift_in_loop_meta_code')
    bounds_struct = dace.data.Structure({
        'start': dace.data.Scalar(dace.int32),
        'end': dace.data.Scalar(dace.int32),
        'step': dace.data.Scalar(dace.int32),
    })
    sdfg.add_datadesc('bounds', bounds_struct)
    sdfg.add_array('A', (20, ), dace.int32)
    loop = LoopRegion('loop', 'i < bounds.end', 'i', 'i = bounds.start', 'i = i + bounds.step')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('state', is_start_block=True)
    a_write = state.add_access('A')
    t1 = state.add_tasklet('t1', {}, {'o1'}, 'o1 = 1')
    state.add_edge(t1, 'o1', a_write, None, dace.Memlet('A[i]'))

    assert len(sdfg.nodes()) == 1
    assert len(sdfg.arrays) == 2
    assert sdfg.is_valid()

    FixedPointPipeline([LiftStructViews()]).apply_pass(sdfg, {})

    assert len(sdfg.nodes()) == 2
    assert len(sdfg.arrays) == 5
    assert sdfg.is_valid()

    a = np.zeros((20, ), np.int32)
    valid = np.full((20, ), 1, np.int32)
    inpBounds = bounds_struct.dtype._typeclass.as_ctypes()(start=0, end=20, step=1)
    func = sdfg.compile()
    func(A=a, bounds=inpBounds)

    assert np.allclose(a, valid)


if __name__ == '__main__':
    test_simple_tasklet_access()
    test_sliced_tasklet_access()
    test_sliced_multi_tasklet_access()
    test_tasklet_access_to_cont_array()
    test_sliced_multi_tasklet_access_to_cont_array()
    test_lift_in_loop_meta_code()
