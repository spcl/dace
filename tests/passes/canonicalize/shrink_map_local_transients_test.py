# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A map-body-local transient must shrink to the box its accesses name."""
import numpy as np
import pytest

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


def scratch_inside_a_kernel(schedule: dace.ScheduleType) -> dace.SDFG:
    """``out[i] = (a[i] * 2) + 1`` through a FULL-extent ``GPU_Global`` scratch, under ``schedule``.

    The shape the GPU offload leaves after a fusion pulls a producer into its consumer: npbench
    warpx_boris_push carries three such ``(np_particles,)`` buffers per kernel.
    """
    sdfg = dace.SDFG(f'kernel_local_scratch_{schedule.name}')
    sdfg.add_array('A', [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [N], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_transient('scratch', [N], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('body', dict(i='0:N'), schedule=schedule)
    produce = state.add_tasklet('produce', {'a'}, {'s'}, 's = a * 2.0')
    consume = state.add_tasklet('consume', {'s'}, {'b'}, 'b = s + 1.0')
    scratch = state.add_access('scratch')
    state.add_memlet_path(state.add_read('A'), entry, produce, dst_conn='a', memlet=dace.Memlet('A[i]'))
    state.add_edge(produce, 's', scratch, None, dace.Memlet('scratch[i]'))
    state.add_edge(scratch, None, consume, 's', dace.Memlet('scratch[i]'))
    state.add_memlet_path(consume, exit_node, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    return sdfg


def test_a_kernel_local_device_scratch_shrinks_to_a_register():
    """Skipped as not resizable, the scratch kept its full extent, and the codegen hoist then gave
    every kernel iteration a whole slice: an ``np_particles * np_particles`` device buffer that
    faulted warpx_boris_push on the GPU canonicalize column."""
    sdfg = scratch_inside_a_kernel(dace.ScheduleType.GPU_Device)
    assert ShrinkMapLocalTransients().apply_pass(sdfg, {}) == 1
    desc = sdfg.arrays['scratch']
    assert tuple(int(s) for s in desc.shape) == (1, )
    assert desc.storage == dace.StorageType.Register, 'a shrunk device scratch shared by every thread'
    sdfg.validate()


def test_a_host_map_leaves_its_device_scratch_alone():
    """Under a host map the buffer is not thread-private, so it keeps its extent and storage."""
    sdfg = scratch_inside_a_kernel(dace.ScheduleType.Sequential)
    assert ShrinkMapLocalTransients().apply_pass(sdfg, {}) is None
    assert sdfg.arrays['scratch'].storage == dace.StorageType.GPU_Global


@pytest.mark.gpu
def test_a_shrunk_kernel_scratch_computes_the_values():
    import cupy  # GPU-only dependency; a CPU collection of this file must not need it
    sdfg = scratch_inside_a_kernel(dace.ScheduleType.GPU_Device)
    ShrinkMapLocalTransients().apply_pass(sdfg, {})
    size = 1 << 16
    a = np.random.default_rng(2).random(size)
    arrays = {'A': cupy.asarray(a), 'B': cupy.zeros(size)}
    sdfg(**arrays, N=size)
    assert np.allclose(arrays['B'].get(), a * 2.0 + 1.0)


def scratch_inside_a_nested_body(reassign: bool = False) -> dace.SDFG:
    """``B[i] = A[i] * 2 + 1`` with the map body nested into its own SDFG, which owns a FULL-extent
    scratch indexed by the symbol ``k`` the map binds to ``i``.

    :param reassign: Add a second state after an interstate edge that assigns ``k``, so the box the
        accesses name is not one element for the whole nest.
    """
    body = dace.SDFG('nested_body')
    body.add_array('A', [N], dace.float64)
    body.add_array('B', [N], dace.float64)
    body.add_transient('scratch', [N], dace.float64)
    body.add_symbol('k', dace.int64)
    state = body.add_state('compute', is_start_block=True)
    produce = state.add_tasklet('produce', {'a'}, {'s'}, 's = a * 2.0')
    consume = state.add_tasklet('consume', {'s'}, {'b'}, 'b = s + 1.0')
    scratch = state.add_access('scratch')
    state.add_edge(state.add_read('A'), None, produce, 'a', dace.Memlet('A[k]'))
    state.add_edge(produce, 's', scratch, None, dace.Memlet('scratch[k]'))
    state.add_edge(scratch, None, consume, 's', dace.Memlet('scratch[k]'))
    state.add_edge(consume, 'b', state.add_write('B'), None, dace.Memlet('B[k]'))
    if reassign:
        again = body.add_state_after(state, 'again', assignments={'k': 'N - 1 - k'})
        rewrite = again.add_tasklet('rewrite', {'s'}, {'b'}, 'b = s')
        again.add_edge(again.add_read('scratch'), None, rewrite, 's', dace.Memlet('scratch[k]'))
        again.add_edge(rewrite, 'b', again.add_write('B'), None, dace.Memlet('B[k]'))

    sdfg = dace.SDFG(f'nested_scratch_{reassign}')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    outer = sdfg.add_state()
    entry, exit_node = outer.add_map('rows', dict(i='0:N'))
    nest = outer.add_nested_sdfg(body, {'A'}, {'B'}, symbol_mapping={'k': 'i', 'N': 'N'})
    outer.add_memlet_path(outer.add_read('A'), entry, nest, dst_conn='A', memlet=dace.Memlet('A[0:N]'))
    outer.add_memlet_path(nest, exit_node, outer.add_write('B'), src_conn='B', memlet=dace.Memlet('B[0:N]'))
    sdfg.validate()
    return sdfg


def test_a_nested_body_scratch_shrinks_to_the_box_its_symbol_names():
    """npbench warpx_boris_push nests each kernel body, and four ``(np_particles,)`` scratch buffers
    of those nests were left full-size: every thread then got a whole copy, an ``np_particles**2``
    device allocation that faulted on the GPU canonicalize column."""
    sdfg = scratch_inside_a_nested_body()
    assert ShrinkMapLocalTransients().apply_pass(sdfg, {}) == 1
    nest = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert tuple(int(s) for s in nest.sdfg.arrays['scratch'].shape) == (1, )
    sdfg.validate()

    size = 64
    a = np.random.rand(size)
    b = np.zeros(size)
    sdfg(A=a, B=b, N=size)
    assert np.allclose(b, a * 2.0 + 1.0)


def test_a_nested_scratch_whose_symbol_is_reassigned_keeps_its_extent():
    """After ``k = N - 1 - k`` the same text names another element, so one box is not enough."""
    sdfg = scratch_inside_a_nested_body(reassign=True)
    assert ShrinkMapLocalTransients().apply_pass(sdfg, {}) is None
    nest = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert tuple(str(s) for s in nest.sdfg.arrays['scratch'].shape) == ('N', )


if __name__ == '__main__':
    test_map_body_local_transient_shrinks_to_the_box_it_names()
    test_finalize_leaves_no_symbolically_sized_stack_array()
    test_a_kernel_local_device_scratch_shrinks_to_a_register()
    test_a_host_map_leaves_its_device_scratch_alone()
    test_a_nested_body_scratch_shrinks_to_the_box_its_symbol_names()
    test_a_nested_scratch_whose_symbol_is_reassigned_keeps_its_extent()
