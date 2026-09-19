# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map-body-local transient must shrink to the box its accesses name."""
import numpy as np

import dace
from dace.transformation.passes.canonicalize.finalize import finalize_transient_storage
from dace.transformation.passes.canonicalize.shrink_map_local_transients import ShrinkMapLocalTransients

N = dace.symbol('N', dace.int64)


def scratch_inside_a_map() -> dace.SDFG:
    """A map whose body writes and reads one element of a FULL-extent transient."""
    sdfg = dace.SDFG('map_local_scratch')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N, N], dace.float64)
    sdfg.add_transient('scratch', [N, N], dace.float64)

    state = sdfg.add_state()
    entry, exit_node = state.add_map('body', dict(i='0:N', j='0:N'))
    produce = state.add_tasklet('produce', {'a'}, {'s'}, 's = a * 2.0')
    consume = state.add_tasklet('consume', {'s'}, {'b'}, 'b = s + 1.0')
    scratch = state.add_access('scratch')

    state.add_memlet_path(state.add_read('A'), entry, produce, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_edge(produce, 's', scratch, None, dace.Memlet('scratch[i, j]'))
    state.add_edge(scratch, None, consume, 's', dace.Memlet('scratch[i, j]'))
    state.add_memlet_path(consume, exit_node, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i, j]'))
    return sdfg


def test_map_body_local_transient_shrinks_to_the_box_it_names():
    sdfg = scratch_inside_a_map()
    assert tuple(str(s) for s in sdfg.arrays['scratch'].shape) == ('N', 'N')

    assert ShrinkMapLocalTransients().apply_pass(sdfg, {}) == 1
    assert tuple(int(s) for s in sdfg.arrays['scratch'].shape) == (1, 1)
    for edge in sdfg.states()[0].edges():
        if edge.data.data == 'scratch':
            assert str(edge.data.subset) == '0, 0'
    sdfg.validate()

    size = 64
    a = np.random.rand(size, size)
    b = np.zeros((size, size))
    sdfg(A=a, B=b, N=size)
    assert np.allclose(b, a * 2.0 + 1.0)


def test_finalize_leaves_no_symbolically_sized_stack_array():
    """``Register`` storage is the stack; a symbolic extent there overflows it at run time."""
    sdfg = scratch_inside_a_map()
    finalize_transient_storage(sdfg, dace.DeviceType.CPU)

    for sd, name, desc in sdfg.arrays_recursive():
        if not desc.transient or desc.storage != dace.StorageType.Register:
            continue
        assert not dace.symbolic.issymbolic(desc.total_size), f'{name} is a symbolically sized stack array'

    # Big enough that the unshrunk extent (size*size doubles) cannot live on a thread stack.
    size = 4096
    a = np.random.rand(size, size)
    b = np.zeros((size, size))
    sdfg(A=a, B=b, N=size)
    assert np.allclose(b, a * 2.0 + 1.0)


if __name__ == '__main__':
    test_map_body_local_transient_shrinks_to_the_box_it_names()
    test_finalize_leaves_no_symbolically_sized_stack_array()
