# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``is_warp_tile`` tag must survive the device offload and be redeemed exactly once.

The tag exists because the offload refuses a GPU schedule set before it runs and assigns every
nested scope ``Sequential`` -- correctly, since a kernel launch inside a kernel is not expressible.
So a producer that has PROVEN a nested map data-parallel has no way to say so at the time it knows
it. The tag carries the request across, and :class:`PromoteWarpTiles` redeems it afterwards.

Two failure directions, both checked below: losing the tag (the request never arrives) and
redeeming it where a thread block has no meaning (nested inside another one, or with no kernel
around it at all).
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import finalize
from dace.transformation.passes.gpu_specialization.promote_warp_tiles import PromoteWarpTiles

N = dace.symbol('N', dtype=dace.int64)
TILE = 64


def two_level(outer_schedule=dtypes.ScheduleType.Default, inner_schedule=dtypes.ScheduleType.Default, tag=True):
    """A device map over a tile map, in the shape the offload produces: outer kernel, inner tile."""
    sdfg = dace.SDFG('warp_tile')
    sdfg.add_array('a', [N, TILE], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    outer_e, outer_x = state.add_map('outer', {'ti': '0:N'}, schedule=outer_schedule)
    inner_e, inner_x = state.add_map('tile', {'tj': f'0:{TILE}'}, schedule=inner_schedule)
    inner_e.map.is_warp_tile = tag
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in * 2.0')
    rd, wr = state.add_read('a'), state.add_write('a')
    state.add_memlet_path(rd, outer_e, inner_e, tasklet, dst_conn='__in', memlet=dace.Memlet('a[ti, tj]'))
    state.add_memlet_path(tasklet, inner_x, outer_x, wr, src_conn='__out', memlet=dace.Memlet('a[ti, tj]'))
    sdfg.validate()
    return sdfg


def tile_map(sdfg):
    return next(n for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
                if isinstance(n, nodes.MapEntry) and n.map.label == 'tile')


def test_the_tag_is_off_by_default():
    """Nothing gets a thread block by accident: every map ever built starts untagged."""
    sdfg = two_level(tag=False)
    assert tile_map(sdfg).map.is_warp_tile is False
    plain = dace.SDFG('plain')
    plain.add_array('a', [N], dace.float64)
    st = plain.add_state('s', is_start_block=True)
    entry, _ = st.add_map('m', {'i': '0:N'})
    assert entry.map.is_warp_tile is False


def test_the_device_offload_sequentializes_the_map_and_keeps_the_tag():
    """The whole reason the tag exists: the offload is right to flatten, and must not clear it.

    Driven through the raw device offload rather than :func:`finalize.offload_to_gpu`, which now
    redeems the tag itself -- this is the state the promotion has to start from.
    """
    sdfg = two_level()
    sdfg.apply_gpu_transformations()
    tile = tile_map(sdfg)
    assert tile.map.schedule == dtypes.ScheduleType.Sequential, 'the offload stopped sequentializing'
    assert tile.map.is_warp_tile is True, 'the tag did not survive the offload'
    assert PromoteWarpTiles().apply_pass(sdfg, {}) == 1
    assert tile_map(sdfg).map.schedule == dtypes.ScheduleType.GPU_ThreadBlock


def test_the_full_offload_redeems_the_tag_itself():
    """``offload_to_gpu`` promotes between the device move and the block-size choice, because the
    selector skips a kernel that already has a thread-block map -- promoting after it would leave
    the kernel carrying both a declared size and a block map, which codegen refuses."""
    sdfg = two_level()
    finalize.offload_to_gpu(sdfg)
    assert tile_map(sdfg).map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    kernel = next(n for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
                  if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device)
    assert kernel.map.gpu_block_size is None, 'the thread-block map is the block spec; nothing may declare another'


def test_an_untagged_map_stays_sequential():
    sdfg = two_level(tag=False)
    finalize.offload_to_gpu(sdfg)
    assert PromoteWarpTiles().apply_pass(sdfg, {}) is None
    assert tile_map(sdfg).map.schedule == dtypes.ScheduleType.Sequential


def test_a_map_someone_else_already_scheduled_is_not_overruled():
    """Only a map the offload left ``Sequential`` is a pending request; anything else is a decision."""
    sdfg = two_level()
    finalize.offload_to_gpu(sdfg)
    tile_map(sdfg).map.schedule = dtypes.ScheduleType.GPU_ThreadBlock_Dynamic
    assert PromoteWarpTiles().apply_pass(sdfg, {}) is None
    assert tile_map(sdfg).map.schedule == dtypes.ScheduleType.GPU_ThreadBlock_Dynamic


def test_a_tag_with_no_kernel_around_it_is_refused():
    """A thread block outside a kernel is meaningless; the tag alone must not conjure one."""
    sdfg = dace.SDFG('hostside')
    sdfg.add_array('a', [N], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_ = state.add_map('tile', {'i': '0:N'}, schedule=dtypes.ScheduleType.Sequential)
    entry.map.is_warp_tile = True
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in + 1.0')
    rd, wr = state.add_read('a'), state.add_write('a')
    state.add_memlet_path(rd, entry, tasklet, dst_conn='__in', memlet=dace.Memlet('a[i]'))
    state.add_memlet_path(tasklet, exit_, wr, src_conn='__out', memlet=dace.Memlet('a[i]'))
    sdfg.validate()
    assert PromoteWarpTiles().apply_pass(sdfg, {}) is None
    assert entry.map.schedule == dtypes.ScheduleType.Sequential


def test_a_tag_inside_a_thread_block_is_refused():
    """A thread block nested in a thread block has no meaning; the enclosing scope decides."""
    sdfg = two_level(outer_schedule=dtypes.ScheduleType.GPU_ThreadBlock, inner_schedule=dtypes.ScheduleType.Sequential)
    assert PromoteWarpTiles().apply_pass(sdfg, {}) is None
    assert tile_map(sdfg).map.schedule == dtypes.ScheduleType.Sequential


def test_a_tag_wrapping_a_thread_block_is_refused():
    """Same rule from the other side: a promotion may not put a block scope inside a block scope."""
    sdfg = dace.SDFG('threedeep')
    sdfg.add_array('a', [N, 8, 8], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    dev_e, dev_x = state.add_map('outer', {'ti': '0:N'}, schedule=dtypes.ScheduleType.GPU_Device)
    mid_e, mid_x = state.add_map('tile', {'tj': '0:8'}, schedule=dtypes.ScheduleType.Sequential)
    mid_e.map.is_warp_tile = True
    in_e, in_x = state.add_map('block', {'tk': '0:8'}, schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in * 2.0')
    rd, wr = state.add_read('a'), state.add_write('a')
    state.add_memlet_path(rd, dev_e, mid_e, in_e, tasklet, dst_conn='__in', memlet=dace.Memlet('a[ti, tj, tk]'))
    state.add_memlet_path(tasklet, in_x, mid_x, dev_x, wr, src_conn='__out', memlet=dace.Memlet('a[ti, tj, tk]'))
    sdfg.validate()
    assert PromoteWarpTiles().apply_pass(sdfg, {}) is None
    assert mid_e.map.schedule == dtypes.ScheduleType.Sequential


def test_the_tag_survives_a_json_round_trip():
    """A tag that a save/load drops is a request lost between the passes that set and read it."""
    sdfg = two_level()
    restored = dace.SDFG.from_json(sdfg.to_json())
    assert tile_map(restored).map.is_warp_tile is True
    untagged = dace.SDFG.from_json(two_level(tag=False).to_json())
    assert tile_map(untagged).map.is_warp_tile is False


@pytest.mark.gpu
def test_a_promoted_tile_still_computes():
    """The point is a kernel that is both faster and RIGHT; this checks the second half."""
    import cupy
    sdfg = two_level()
    finalize.offload_to_gpu(sdfg)
    finalize.finalize_for_target(sdfg, 'gpu')
    n = 40
    a = np.random.default_rng(7).random((n, TILE))
    device = cupy.asarray(a)
    sdfg(a=device, N=n)
    assert np.allclose(cupy.asnumpy(device), a * 2.0)


def stepped_two_level(tag=True, tile=128):
    """A thread-block map re-entered by a sequential loop -- the shape that needs a barrier."""
    from dace.sdfg.state import LoopRegion
    sdfg = dace.SDFG('stepped')
    sdfg.add_array('a', [N, tile], dace.float64)
    outer = LoopRegion('step', 'step < N', 'step', 'step = 0', 'step = step + 1')
    sdfg.add_node(outer, is_start_block=True)
    state = outer.add_state('body', is_start_block=True)
    dev_e, dev_x = state.add_map('outer', {'ti': '0:N'}, schedule=dtypes.ScheduleType.Default)
    tile_e, tile_x = state.add_map('tile', {'tj': f'0:{tile}'}, schedule=dtypes.ScheduleType.Default)
    tile_e.map.is_warp_tile = tag
    tasklet = state.add_tasklet('t', {'__in'}, {'__out'}, '__out = __in * 2.0')
    rd, wr = state.add_read('a'), state.add_write('a')
    state.add_memlet_path(rd, dev_e, tile_e, tasklet, dst_conn='__in', memlet=dace.Memlet('a[ti, tj]'))
    state.add_memlet_path(tasklet, tile_x, dev_x, wr, src_conn='__out', memlet=dace.Memlet('a[ti, tj]'))
    sdfg.validate()
    return sdfg


def barrier_tasklets(sdfg):
    return [
        n for g in sdfg.all_sdfgs_recursive() for st in g.all_states() for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and '__syncthreads' in n.code.as_string
    ]


def test_a_promoted_tile_inside_a_loop_gets_a_barrier():
    """The miscompile this exists to stop.

    Once the map is a thread block, nothing separates one execution of it from the next. A block
    wider than one hardware wavefront runs them out of step, so the next execution reads what the
    previous has not finished writing. Measured on the wavefront tile: right at 32 and 64 threads
    (one CDNA wavefront, lockstep by accident), WRONG at 128, 256 and 512.
    """
    sdfg = stepped_two_level()
    finalize.offload_to_gpu(sdfg)
    assert barrier_tasklets(sdfg), 'a thread block re-entered by a loop must be separated by a barrier'


def test_a_promoted_tile_with_no_loop_around_it_gets_no_barrier():
    """A barrier costs the whole block; only a loop can carry a dependence across executions."""
    sdfg = two_level()
    finalize.offload_to_gpu(sdfg)
    assert tile_map(sdfg).map.schedule == dtypes.ScheduleType.GPU_ThreadBlock
    assert not barrier_tasklets(sdfg), 'nothing re-enters this map; the barrier is pure cost'


def test_an_untagged_map_inside_a_loop_gets_no_barrier():
    """Non-vacuity: the barrier follows the PROMOTION, not the loop."""
    sdfg = stepped_two_level(tag=False)
    finalize.offload_to_gpu(sdfg)
    assert not barrier_tasklets(sdfg)


def test_the_barrier_sits_outside_the_lane_mask_in_the_emitted_code():
    """Inside the map the body is lane-masked, and a divergent ``__syncthreads()`` DEADLOCKS --
    strictly worse than the race. It has to land after the map exit, where the codegen has closed
    the thread-block guard and every thread of the block reaches it."""
    sdfg = stepped_two_level()
    finalize.offload_to_gpu(sdfg)
    finalize.finalize_for_target(sdfg, 'gpu')
    device = [c for c in sdfg.generate_code() if c.title == 'CUDA']
    assert device, 'no device translation unit was generated'
    code = '\n'.join(c.clean_code for c in device)
    assert '__syncthreads();' in code, 'the barrier never reached the generated kernel'


if __name__ == '__main__':
    test_the_tag_is_off_by_default()
    test_the_device_offload_sequentializes_the_map_and_keeps_the_tag()
    test_the_full_offload_redeems_the_tag_itself()
    test_an_untagged_map_stays_sequential()
    test_a_map_someone_else_already_scheduled_is_not_overruled()
    test_a_tag_with_no_kernel_around_it_is_refused()
    test_a_tag_inside_a_thread_block_is_refused()
    test_a_tag_wrapping_a_thread_block_is_refused()
    test_the_tag_survives_a_json_round_trip()
    test_a_promoted_tile_inside_a_loop_gets_a_barrier()
    test_a_promoted_tile_with_no_loop_around_it_gets_no_barrier()
    test_an_untagged_map_inside_a_loop_gets_no_barrier()
