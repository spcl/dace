# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural tests for :mod:`tests.corpus.cloudsc.offload_cloudsc_to_gpu`.

All in-process and tiny -- no CUDA, no compile, no CloudSC build. Each test pins one post-condition of
the offload: where the kernel boundary lands relative to the block map, which arrays are mirrored, and
which of those are dual-resident constants. The block-map fixture is built twice, hand-rolled and via
an equivalent ``@dace.program``, so a frontend shape the hand-built graph does not reproduce cannot
hide a bug.

    pytest tests/corpus/cloudsc/cloudsc_offload_to_gpu_test.py -v
"""
import pytest

import dace
from dace import dtypes
from dace.memlet import Memlet
from dace.sdfg import nodes

from tests.corpus.cloudsc.offload_cloudsc_to_gpu import constant_offload_data, offload_cloudsc_to_gpu

nblocks = dace.symbol('nblocks')
klev = dace.symbol('klev')
klon = dace.symbol('klon')


@dace.program
def cloudsc_like(pin: dace.float64[klev, klon, nblocks], pout: dace.float64[klev, klon, nblocks]):
    """CloudSC's shape in miniature: a per-block outer map wrapping the horizontal/vertical work."""
    for ibl in dace.map[0:nblocks]:
        for jk, jl in dace.map[0:klev, 0:klon]:
            pout[jk, jl, ibl] = 2.0 * pin[jk, jl, ibl]


def blocked_sdfg(inner: bool = True) -> dace.SDFG:
    """Hand-built twin of :func:`cloudsc_like`. ``inner=False`` drops the inner map, leaving a leaf
    map whose range still mentions ``nblocks``."""
    sdfg = dace.SDFG('blocked')
    for name in ('pin', 'pout'):
        sdfg.add_array(name, [klev, klon, nblocks], dace.float64)
    state = sdfg.add_state()
    read, write = state.add_read('pin'), state.add_write('pout')
    entry, exit_ = state.add_map('blocks', {'ibl': '0:nblocks'})
    state.add_edge(read, None, entry, 'IN_pin', Memlet.from_array('pin', sdfg.arrays['pin']))
    if inner:
        inner_entry, inner_exit = state.add_map('work', {'jk': '0:klev', 'jl': '0:klon'})
        tasklet = state.add_tasklet('double', {'a'}, {'o'}, 'o = 2.0 * a')
        state.add_edge(entry, 'OUT_pin', inner_entry, 'IN_pin', Memlet(data='pin', subset='0:klev, 0:klon, ibl'))
        state.add_edge(inner_entry, 'OUT_pin', tasklet, 'a', Memlet(data='pin', subset='jk, jl, ibl'))
        state.add_edge(tasklet, 'o', inner_exit, 'IN_pout', Memlet(data='pout', subset='jk, jl, ibl'))
        state.add_edge(inner_exit, 'OUT_pout', exit_, 'IN_pout', Memlet(data='pout', subset='0:klev, 0:klon, ibl'))
        for node, conns in ((inner_entry, ('IN_pin', 'OUT_pin')), (inner_exit, ('IN_pout', 'OUT_pout'))):
            node.add_in_connector(conns[0])
            node.add_out_connector(conns[1])
    else:
        tasklet = state.add_tasklet('double', {'a'}, {'o'}, 'o = 2.0 * a')
        state.add_edge(entry, 'OUT_pin', tasklet, 'a', Memlet(data='pin', subset='0, 0, ibl'))
        state.add_edge(tasklet, 'o', exit_, 'IN_pout', Memlet(data='pout', subset='0, 0, ibl'))
    state.add_edge(exit_, 'OUT_pout', write, None, Memlet.from_array('pout', sdfg.arrays['pout']))
    entry.add_in_connector('IN_pin')
    entry.add_out_connector('OUT_pin')
    exit_.add_in_connector('IN_pout')
    exit_.add_out_connector('OUT_pout')
    return sdfg


def map_schedules(sdfg: dace.SDFG):
    """``{map label: (schedule, is_top_level_in_its_own_state)}`` over the whole tree."""
    out = {}
    for graph in sdfg.all_sdfgs_recursive():
        for state in graph.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    out[node.map.label] = (node.map.schedule, state.entry_node(node) is None)
    return out


@pytest.mark.parametrize('build', [blocked_sdfg, lambda: cloudsc_like.to_sdfg(simplify=True)],
                         ids=['handbuilt', 'dace_program'])
def test_block_map_stays_host_and_inner_map_is_offloaded(build):
    """The nblocks map orchestrates on the host; the map strictly inside it becomes the kernel."""
    sdfg = build()
    offload_cloudsc_to_gpu(sdfg)
    schedules = map_schedules(sdfg)
    top = [s for s, is_top in schedules.values() if is_top]
    inner = [s for s, is_top in schedules.values() if not is_top]
    assert top == [dtypes.ScheduleType.Sequential], schedules
    assert inner and all(s == dtypes.ScheduleType.GPU_Device for s in inner), schedules


def test_leaf_map_over_blocks_is_still_offloaded():
    """The block-symbol name signal alone is not enough: a map with nothing inside it to offload is
    the kernel itself, so it must not be demoted to the host."""
    sdfg = blocked_sdfg(inner=False)
    offload_cloudsc_to_gpu(sdfg)
    assert map_schedules(sdfg)['blocks'][0] == dtypes.ScheduleType.GPU_Device


def test_maps_below_the_kernel_are_sequential():
    """Three levels: nblocks -> kernel -> device-level. Only the middle one launches."""
    sdfg = dace.SDFG('three_level')
    sdfg.add_array('a', [4, 4, 4], dace.float64)
    sdfg.add_symbol('nblocks', dace.int32)
    state = sdfg.add_state()
    write = state.add_write('a')
    outer_entry, outer_exit = state.add_map('blocks', {'ibl': '0:nblocks'})
    mid_entry, mid_exit = state.add_map('kernel', {'j': '0:4'})
    deep_entry, deep_exit = state.add_map('deep', {'k': '0:4'})
    tasklet = state.add_tasklet('set', {}, {'o'}, 'o = 1.0')
    state.add_edge(outer_entry, None, mid_entry, None, Memlet())
    state.add_edge(mid_entry, None, deep_entry, None, Memlet())
    state.add_edge(deep_entry, None, tasklet, None, Memlet())
    state.add_edge(tasklet, 'o', deep_exit, 'IN_a', Memlet(data='a', subset='ibl, j, k'))
    state.add_edge(deep_exit, 'OUT_a', mid_exit, 'IN_a', Memlet(data='a', subset='ibl, j, 0:4'))
    state.add_edge(mid_exit, 'OUT_a', outer_exit, 'IN_a', Memlet(data='a', subset='ibl, 0:4, 0:4'))
    state.add_edge(outer_exit, 'OUT_a', write, None, Memlet.from_array('a', sdfg.arrays['a']))
    for node in (deep_exit, mid_exit, outer_exit):
        node.add_in_connector('IN_a')
        node.add_out_connector('OUT_a')

    offload_cloudsc_to_gpu(sdfg)
    schedules = map_schedules(sdfg)
    assert schedules['blocks'][0] == dtypes.ScheduleType.Sequential
    assert schedules['kernel'][0] == dtypes.ScheduleType.GPU_Device
    assert schedules['deep'][0] == dtypes.ScheduleType.Sequential


def test_nested_sdfg_under_block_map_is_offloaded():
    """The CloudSC Fortran frontend wraps the per-block body in a NestedSDFG. Its top map is still the
    outermost map strictly inside the block map, so it is the kernel -- and its own nested maps are
    device-level."""
    inner = dace.SDFG('body')
    for name in ('a', 'b'):
        inner.add_array(name, [4, 4], dace.float64)
    istate = inner.add_state()
    entry, exit_ = istate.add_map('body_work', {'j': '0:4'})
    deep_entry, deep_exit = istate.add_map('body_deep', {'k': '0:4'})
    tasklet = istate.add_tasklet('double', {'x'}, {'o'}, 'o = 2.0 * x')
    istate.add_edge(istate.add_read('a'), None, entry, 'IN_a', Memlet.from_array('a', inner.arrays['a']))
    istate.add_edge(entry, 'OUT_a', deep_entry, 'IN_a', Memlet(data='a', subset='j, 0:4'))
    istate.add_edge(deep_entry, 'OUT_a', tasklet, 'x', Memlet(data='a', subset='j, k'))
    istate.add_edge(tasklet, 'o', deep_exit, 'IN_b', Memlet(data='b', subset='j, k'))
    istate.add_edge(deep_exit, 'OUT_b', exit_, 'IN_b', Memlet(data='b', subset='j, 0:4'))
    istate.add_edge(exit_, 'OUT_b', istate.add_write('b'), None, Memlet.from_array('b', inner.arrays['b']))
    for node, conns in ((entry, ('IN_a', 'OUT_a')), (deep_entry, ('IN_a', 'OUT_a')), (deep_exit, ('IN_b', 'OUT_b')),
                        (exit_, ('IN_b', 'OUT_b'))):
        node.add_in_connector(conns[0])
        node.add_out_connector(conns[1])

    sdfg = dace.SDFG('blocked_nsdfg')
    for name in ('a', 'b'):
        sdfg.add_array(name, [4, 4, 4], dace.float64)
    sdfg.add_symbol('nblocks', dace.int32)
    state = sdfg.add_state()
    bentry, bexit = state.add_map('blocks', {'ibl': '0:nblocks'})
    nsdfg = state.add_nested_sdfg(inner, {'a'}, {'b'})
    state.add_edge(state.add_read('a'), None, bentry, 'IN_a', Memlet.from_array('a', sdfg.arrays['a']))
    state.add_edge(bentry, 'OUT_a', nsdfg, 'a', Memlet(data='a', subset='0:4, 0:4, ibl'))
    state.add_edge(nsdfg, 'b', bexit, 'IN_b', Memlet(data='b', subset='0:4, 0:4, ibl'))
    state.add_edge(bexit, 'OUT_b', state.add_write('b'), None, Memlet.from_array('b', sdfg.arrays['b']))
    bentry.add_in_connector('IN_a')
    bentry.add_out_connector('OUT_a')
    bexit.add_in_connector('IN_b')
    bexit.add_out_connector('OUT_b')

    offload_cloudsc_to_gpu(sdfg)
    schedules = map_schedules(sdfg)
    assert schedules['blocks'][0] == dtypes.ScheduleType.Sequential
    assert schedules['body_work'][0] == dtypes.ScheduleType.GPU_Device
    assert schedules['body_deep'][0] == dtypes.ScheduleType.Sequential
    # The NSDFG's inner descriptors follow the outer GPU_Global bindings.
    assert inner.arrays['a'].storage == dtypes.StorageType.GPU_Global
    assert inner.arrays['b'].storage == dtypes.StorageType.GPU_Global


def test_read_only_input_is_dual_resident():
    """``pin`` is never written: mirror it once in the head state, keep the host original, emit no
    copy-out. ``pout`` is written, so it round-trips."""
    sdfg = blocked_sdfg()
    assert constant_offload_data(sdfg, {'pin', 'pout'}) == {'pin': None}
    offload_cloudsc_to_gpu(sdfg)

    assert sdfg.arrays['gpu_pin'].storage == dtypes.StorageType.GPU_Global
    assert sdfg.arrays['gpu_pin'].transient and not sdfg.arrays['pin'].transient
    states = {s.label: s for s in sdfg.states()}
    copied_in = {e.dst.data for e in states['gpu_copy_in'].edges()}
    copied_out = {e.src.data for e in states['gpu_copy_out'].edges()}
    assert copied_in == {'gpu_pin', 'gpu_pout'}
    assert copied_out == {'gpu_pout'}, 'a read-only input must not be copied back'


def test_write_once_full_copy_is_constant():
    """A top-level full-array copy (what canon's ``lift_copy`` leaves behind) is a host-side single
    writer that fully covers the array -- constant, so no copy-out, and its mirror is filled after the
    producing state rather than in the head state."""
    sdfg = dace.SDFG('write_once')
    for name in ('src', 'tab', 'out'):
        sdfg.add_array(name, [8], dace.float64)
    produce = sdfg.add_state('produce', is_start_block=True)
    produce.add_edge(produce.add_read('src'), None, produce.add_write('tab'), None,
                     Memlet.from_array('src', sdfg.arrays['src']))
    use = sdfg.add_state_after(produce, 'use')
    entry, exit_ = use.add_map('kernel', {'i': '0:8'})
    tasklet = use.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    use.add_edge(use.add_read('tab'), None, entry, 'IN_tab', Memlet.from_array('tab', sdfg.arrays['tab']))
    use.add_edge(entry, 'OUT_tab', tasklet, 'a', Memlet(data='tab', subset='i'))
    use.add_edge(tasklet, 'o', exit_, 'IN_out', Memlet(data='out', subset='i'))
    use.add_edge(exit_, 'OUT_out', use.add_write('out'), None, Memlet.from_array('out', sdfg.arrays['out']))
    entry.add_in_connector('IN_tab')
    entry.add_out_connector('OUT_tab')
    exit_.add_in_connector('IN_out')
    exit_.add_out_connector('OUT_out')

    assert constant_offload_data(sdfg, {'tab', 'out'}) == {'tab': produce}
    offload_cloudsc_to_gpu(sdfg)
    states = {s.label: s for s in sdfg.states()}
    assert {e.dst.data for e in states['gpu_const_copy_in'].edges()} == {'gpu_tab'}
    assert {e.dst.data for e in states['gpu_copy_in'].edges()} == {'gpu_out'}
    assert {e.src.data for e in states['gpu_copy_out'].edges()} == {'gpu_out'}
    # Dual residency: the host descriptor survives untouched next to its device mirror.
    assert not sdfg.arrays['tab'].transient
    assert sdfg.arrays['tab'].storage in (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)


def test_partial_write_is_not_constant():
    """Under-approximation is the safe direction: a write that does not provably cover the array
    leaves it out of the constant set, so it round-trips."""
    sdfg = dace.SDFG('partial')
    for name in ('src', 'tab', 'out'):
        sdfg.add_array(name, [8], dace.float64)
    produce = sdfg.add_state('produce', is_start_block=True)
    produce.add_edge(produce.add_read('src'), None, produce.add_write('tab'), None, Memlet(data='tab', subset='0:4'))
    use = sdfg.add_state_after(produce, 'use')
    entry, exit_ = use.add_map('kernel', {'i': '0:8'})
    tasklet = use.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    use.add_edge(use.add_read('tab'), None, entry, 'IN_tab', Memlet.from_array('tab', sdfg.arrays['tab']))
    use.add_edge(entry, 'OUT_tab', tasklet, 'a', Memlet(data='tab', subset='i'))
    use.add_edge(tasklet, 'o', exit_, 'IN_out', Memlet(data='out', subset='i'))
    use.add_edge(exit_, 'OUT_out', use.add_write('out'), None, Memlet.from_array('out', sdfg.arrays['out']))
    entry.add_in_connector('IN_tab')
    entry.add_out_connector('OUT_tab')
    exit_.add_in_connector('IN_out')
    exit_.add_out_connector('OUT_out')

    assert constant_offload_data(sdfg, {'tab', 'out'}) == {}


def test_device_written_data_is_not_constant():
    """``pout`` is produced inside the kernel, so the host copy is stale and dual residency would be
    wrong -- it must round-trip instead."""
    assert 'pout' not in constant_offload_data(blocked_sdfg(), {'pin', 'pout'})


def test_transients_promoted_and_scalars_registered():
    sdfg = blocked_sdfg()
    sdfg.add_transient('scratch', [4], dace.float64)
    sdfg.add_scalar('acc', dace.float64, transient=True)
    offload_cloudsc_to_gpu(sdfg)
    assert sdfg.arrays['scratch'].storage == dtypes.StorageType.GPU_Global
    assert sdfg.arrays['acc'].storage == dtypes.StorageType.Register


def test_excluded_array_stays_host_side():
    sdfg = blocked_sdfg()
    offload_cloudsc_to_gpu(sdfg, exclude_from_offload=('pin', ))
    assert 'gpu_pin' not in sdfg.arrays
    assert 'gpu_pout' in sdfg.arrays
    assert not sdfg.arrays['pin'].transient


def test_host_only_array_is_not_mirrored():
    """An array touched only by a top-level tasklet never reaches the device."""
    sdfg = dace.SDFG('host_only')
    sdfg.add_array('probe', [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = 3.0')
    state.add_edge(tasklet, 'o', state.add_access('probe'), None, Memlet(data='probe', subset='0'))
    offload_cloudsc_to_gpu(sdfg)
    assert 'gpu_probe' not in sdfg.arrays


def elementwise_producer_sdfg(count: int, size: int, dims=None) -> dace.SDFG:
    """``tab`` written by ``count`` single-element tasklets, then read by a kernel."""
    shape = dims if dims is not None else [size]
    sdfg = dace.SDFG('elementwise')
    sdfg.add_array('tab', shape, dace.float64)
    sdfg.add_array('out', [size], dace.float64)
    produce = sdfg.add_state('produce', is_start_block=True)
    write = produce.add_write('tab')
    for flat in range(count):
        index = []
        remaining = flat
        for extent in reversed(shape):
            index.append(remaining % extent)
            remaining //= extent
        subset = ', '.join(str(i) for i in reversed(index))
        tasklet = produce.add_tasklet(f'w{flat}', {}, {'o'}, f'o = {float(flat)}')
        produce.add_edge(tasklet, 'o', write, None, Memlet(data='tab', subset=subset))
    use = sdfg.add_state_after(produce, 'use')
    entry, exit_ = use.add_map('kernel', {'i': f'0:{size}'})
    tasklet = use.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    use.add_edge(use.add_read('tab'), None, entry, 'IN_tab', Memlet.from_array('tab', sdfg.arrays['tab']))
    use.add_edge(entry, 'OUT_tab', tasklet, 'a', Memlet(data='tab', subset='0' if len(shape) > 1 else 'i'))
    use.add_edge(tasklet, 'o', exit_, 'IN_out', Memlet(data='out', subset='i'))
    use.add_edge(exit_, 'OUT_out', use.add_write('out'), None, Memlet.from_array('out', sdfg.arrays['out']))
    entry.add_in_connector('IN_tab')
    entry.add_out_connector('OUT_tab')
    exit_.add_in_connector('IN_out')
    exit_.add_out_connector('OUT_out')
    return sdfg


def test_unrolled_elementwise_writes_are_constant():
    """N single-element writes that between them hit every element DO prove constancy.

    ``SubsetUnion.covers`` is a per-member test, so no single one of the N writes covers the array;
    the union is what covers it. This is the shape an unrolled assignment loop leaves behind.
    """
    sdfg = elementwise_producer_sdfg(count=4, size=4)
    assert set(constant_offload_data(sdfg, {'tab'})) == {'tab'}


def test_unrolled_elementwise_writes_multidim_are_constant():
    sdfg = elementwise_producer_sdfg(count=6, size=6, dims=[2, 3])
    assert set(constant_offload_data(sdfg, {'tab'})) == {'tab'}


def test_unrolled_elementwise_writes_with_a_gap_are_not_constant():
    """One element short of the array is not coverage -- the under-approximation must refuse."""
    sdfg = elementwise_producer_sdfg(count=3, size=4)
    assert constant_offload_data(sdfg, {'tab'}) == {}


def test_elementwise_cover_refuses_symbolic_shape():
    """A symbolic extent cannot be counted against, so the enumeration proof does not apply."""
    n = dace.symbol('n_elem')
    sdfg = dace.SDFG('symbolic_extent')
    sdfg.add_array('tab', [n], dace.float64)
    sdfg.add_array('out', [n], dace.float64)
    produce = sdfg.add_state('produce', is_start_block=True)
    write = produce.add_write('tab')
    for flat in range(2):
        tasklet = produce.add_tasklet(f'w{flat}', {}, {'o'}, f'o = {float(flat)}')
        produce.add_edge(tasklet, 'o', write, None, Memlet(data='tab', subset=str(flat)))
    use = sdfg.add_state_after(produce, 'use')
    entry, exit_ = use.add_map('kernel', {'i': '0:n_elem'})
    tasklet = use.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    use.add_edge(use.add_read('tab'), None, entry, 'IN_tab', Memlet.from_array('tab', sdfg.arrays['tab']))
    use.add_edge(entry, 'OUT_tab', tasklet, 'a', Memlet(data='tab', subset='i'))
    use.add_edge(tasklet, 'o', exit_, 'IN_out', Memlet(data='out', subset='i'))
    use.add_edge(exit_, 'OUT_out', use.add_write('out'), None, Memlet.from_array('out', sdfg.arrays['out']))
    entry.add_in_connector('IN_tab')
    entry.add_out_connector('OUT_tab')
    exit_.add_in_connector('IN_out')
    exit_.add_out_connector('OUT_out')
    assert constant_offload_data(sdfg, {'tab'}) == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
