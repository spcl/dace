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
from dace import data, dtypes
from dace.codegen.codegen import generate_code
from dace.config import Config
from dace.memlet import Memlet
from dace.sdfg import nodes

from tests.corpus.cloudsc.offload_cloudsc_to_gpu import (constant_offload_data, offload_cloudsc_to_gpu,
                                                         readonly_range_scalars, symbolize_readonly_range_scalars)

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
    """A transient the kernel uses goes to ``GPU_Global``; scalars go to ``Register``.

    ``scratch`` is wired into the kernel body rather than merely declared: promotion is gated on real
    device use, so a dangling descriptor would pin nothing and the assertion would pass vacuously.
    """
    sdfg = blocked_sdfg()
    sdfg.add_transient('scratch', [klev, klon, nblocks], dace.float64)
    sdfg.add_scalar('acc', dace.float64, transient=True)
    state = sdfg.states()[0]
    inner_exit = next(n for n in state.nodes() if isinstance(n, nodes.MapExit) and n.map.label == 'work')
    block_exit = next(n for n in state.nodes() if isinstance(n, nodes.MapExit) and n.map.label == 'blocks')
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    tasklet.add_out_connector('s')
    tasklet.code.as_string = 'o = 2.0 * a\ns = a'
    inner_exit.add_in_connector('IN_scratch')
    inner_exit.add_out_connector('OUT_scratch')
    block_exit.add_in_connector('IN_scratch')
    block_exit.add_out_connector('OUT_scratch')
    state.add_edge(tasklet, 's', inner_exit, 'IN_scratch', Memlet(data='scratch', subset='jk, jl, ibl'))
    state.add_edge(inner_exit, 'OUT_scratch', block_exit, 'IN_scratch',
                   Memlet(data='scratch', subset='0:klev, 0:klon, ibl'))
    state.add_edge(block_exit, 'OUT_scratch', state.add_write('scratch'), None,
                   Memlet.from_array('scratch', sdfg.arrays['scratch']))

    offload_cloudsc_to_gpu(sdfg)
    assert sdfg.arrays['scratch'].storage == dtypes.StorageType.GPU_Global
    assert sdfg.arrays['acc'].storage == dtypes.StorageType.Register


def test_host_only_transient_stays_host():
    """The other side of the gate: a transient no kernel touches must NOT be promoted, or its host
    readers would dereference device memory."""
    sdfg = blocked_sdfg()
    sdfg.add_transient('host_scratch', [4], dace.float64)
    state = sdfg.states()[0]
    producer = state.add_tasklet('init', {}, {'o'}, 'o = 1.0')
    state.add_edge(producer, 'o', state.add_write('host_scratch'), None, Memlet(data='host_scratch', subset='0'))
    offload_cloudsc_to_gpu(sdfg)
    assert sdfg.arrays['host_scratch'].storage in (dtypes.StorageType.Default, dtypes.StorageType.CPU_Heap)
    assert 'gpu_host_scratch' not in sdfg.arrays


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


def dynamic_range_sdfg(write_the_bound: bool = False) -> dace.SDFG:
    """CloudSC's horizontal shape in miniature: a map whose range is read from a scalar argument.

    The Fortran frontend gives ``kidia``/``kfdia`` as scalar arguments and every klon map reads them
    through a same-named dynamic map-range connector. ``write_the_bound`` additionally stores into
    the scalar, which is what must stop the rewrite.
    """
    sdfg = dace.SDFG('dynamic_range')
    sdfg.add_array('pin', [16], dace.float64)
    sdfg.add_array('pout', [16], dace.float64)
    sdfg.add_scalar('kfdia', dace.int32)
    state = sdfg.add_state()
    entry, exit_ = state.add_map('work', {'jl': '0:kfdia'})
    entry.add_in_connector('kfdia')
    state.add_edge(state.add_read('kfdia'), None, entry, 'kfdia', Memlet(data='kfdia', subset='0'))
    tasklet = state.add_tasklet('double', {'a'}, {'o'}, 'o = 2.0 * a')
    state.add_edge(state.add_read('pin'), None, entry, 'IN_pin', Memlet.from_array('pin', sdfg.arrays['pin']))
    state.add_edge(entry, 'OUT_pin', tasklet, 'a', Memlet(data='pin', subset='jl'))
    state.add_edge(tasklet, 'o', exit_, 'IN_pout', Memlet(data='pout', subset='jl'))
    state.add_edge(exit_, 'OUT_pout', state.add_write('pout'), None, Memlet.from_array('pout', sdfg.arrays['pout']))
    entry.add_in_connector('IN_pin')
    entry.add_out_connector('OUT_pin')
    exit_.add_in_connector('IN_pout')
    exit_.add_out_connector('OUT_pout')
    if write_the_bound:
        setter = sdfg.add_state_before(state, 'set_bound')
        producer = setter.add_tasklet('set', {}, {'o'}, 'o = 8')
        setter.add_edge(producer, 'o', setter.add_write('kfdia'), None, Memlet(data='kfdia', subset='0'))
    return sdfg


def map_range_connectors(sdfg: dace.SDFG):
    """Dynamic map-range connector names (the non-``IN_`` in-connectors of every MapEntry)."""
    out = set()
    for graph in sdfg.all_sdfgs_recursive():
        for state in graph.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    out |= {c for c in node.in_connectors if not c.startswith('IN_')}
    return out


def test_readonly_dynamic_map_range_becomes_a_symbol():
    """A never-written bound read only as a map range is turned into a symbol, connector and all.

    The CUDA target cannot emit a ``GPU_Device`` map with a dynamic range: it declares one local per
    dynamic input at the launch site, named after the container it reads (``int kfdia = kfdia[0];``,
    which shadows its own initializer), and a range symbol that reaches only the grid expression is
    left out of the kernel wrapper's parameter list entirely. Denying it that shape is what makes
    CloudSC code-generate.
    """
    sdfg = dynamic_range_sdfg()
    assert map_range_connectors(sdfg) == {'kfdia'}
    assert readonly_range_scalars(sdfg) == {'kfdia'}

    assert symbolize_readonly_range_scalars(sdfg) == 1
    assert 'kfdia' not in sdfg.arrays, 'the scalar descriptor must be gone'
    assert 'kfdia' in sdfg.symbols, 'and replaced by a symbol of the same name'
    assert map_range_connectors(sdfg) == set(), 'the dynamic-range connector must be gone with it'
    sdfg.validate()


def test_written_dynamic_range_scalar_is_left_alone():
    """The rewrite is only sound for a loop-invariant bound: a stored-to scalar keeps its container.

    Turning a written scalar into a symbol would silently drop the store, so the guard has to refuse
    it rather than merely produce a worse graph.
    """
    sdfg = dynamic_range_sdfg(write_the_bound=True)
    assert readonly_range_scalars(sdfg) == set()
    assert symbolize_readonly_range_scalars(sdfg) == 0
    assert isinstance(sdfg.arrays['kfdia'], data.Scalar)
    assert map_range_connectors(sdfg) == {'kfdia'}


def test_offload_pins_the_default_stream():
    """Every launch and async copy must run on the default (null) stream.

    ``max_concurrent_streams = -1`` is what makes the CUDA target fill ``gpu_streams`` with
    ``nullptr`` instead of creating streams, so ``gpu_streams[0]`` IS the default stream. Pinned by
    the pass rather than by the caller's environment, so the offloaded graph has one stream regime
    wherever it is built.
    """
    Config.set('compiler', 'cuda', 'max_concurrent_streams', value=0)
    offload_cloudsc_to_gpu(blocked_sdfg())
    assert int(Config.get('compiler', 'cuda', 'max_concurrent_streams')) == -1


def test_offloaded_graph_generates_default_stream_cuda():
    """The pinned setting reaches the emitted code: streams are nulled, never created."""
    sdfg = blocked_sdfg()
    offload_cloudsc_to_gpu(sdfg)
    cuda = '\n'.join(obj.clean_code for obj in generate_code(sdfg) if obj.language == 'cu')
    assert cuda, 'the offloaded graph must emit a .cu object'
    assert 'internal_streams[i] = nullptr' in cuda
    assert 'StreamCreateWithFlags' not in cuda, 'no concurrent stream may be created'


def test_register_scalars_are_state_lifetime():
    """A Register scalar is a C++ local; at the default ``Scope`` lifetime it is declared inside the
    braces of the component that writes it, so a kernel launched from a LATER component of the same
    state cannot see it (CloudSC's LICM preheaders hit exactly that). ``State`` lifetime hoists the
    declaration to the state's own block."""
    sdfg = blocked_sdfg()
    sdfg.add_scalar('acc', dace.float64, transient=True)
    offload_cloudsc_to_gpu(sdfg)
    assert sdfg.arrays['acc'].storage == dtypes.StorageType.Register
    assert sdfg.arrays['acc'].lifetime == dtypes.AllocationLifetime.State


def test_view_in_a_kernel_still_code_generates():
    """A transient View inside a kernel must survive GPU code generation.

    ``data.View`` subclasses ``data.Array``, so the offload promotes a view to ``GPU_Global`` exactly
    as DaCe's own offloading does. The hoist in
    ``InsertExplicitGPUGlobalMemoryCopies`` then used to treat it as an allocatable kernel-local
    array and hand it to ``MoveArrayOutOfKernel``, which reshapes the view descriptor and hangs a
    second access node off the kernel exit -- destroying the unique view edge and failing validation
    with "Ambiguous or invalid edge to/from a View access node". CloudSC hits this through its six
    ``zpfplsx_*`` views.
    """
    sdfg = dace.SDFG('view_in_kernel')
    sdfg.add_symbol('n', dace.int32)
    sdfg.add_array('pin', [klev, klev], dace.float64)
    sdfg.add_array('pout', [klev, klev], dace.float64)
    # Symbolic extents on purpose: a literal-shape kernel-local transient is demoted to Register
    # instead, which is the branch that does NOT reach the hoist under test.
    sdfg.add_transient('scratch', [klev, klev], dace.float64)
    sdfg.add_view('scratch_row', [klev], dace.float64)
    state = sdfg.add_state()

    outer_entry, outer_exit = state.add_map('kernel', {'i': '0:klev'})
    outer_entry.add_in_connector('IN_pin')
    outer_entry.add_out_connector('OUT_pin')
    outer_exit.add_in_connector('IN_pout')
    outer_exit.add_out_connector('OUT_pout')
    state.add_edge(state.add_read('pin'), None, outer_entry, 'IN_pin', Memlet.from_array('pin', sdfg.arrays['pin']))

    # Both the transient and its view live INSIDE the kernel scope, which is the shape the hoist
    # picks up (CloudSC's zpfplsx_* views sit inside their klon kernels the same way).
    fill_entry, fill_exit = state.add_map('fill', {'j': '0:klev'})
    fill_entry.add_in_connector('IN_pin')
    fill_entry.add_out_connector('OUT_pin')
    fill_exit.add_in_connector('IN_scratch')
    fill_exit.add_out_connector('OUT_scratch')
    fill = state.add_tasklet('copy', {'a'}, {'o'}, 'o = a')
    state.add_edge(outer_entry, 'OUT_pin', fill_entry, 'IN_pin', Memlet(data='pin', subset='i, 0:klev'))
    state.add_edge(fill_entry, 'OUT_pin', fill, 'a', Memlet(data='pin', subset='i, j'))
    state.add_edge(fill, 'o', fill_exit, 'IN_scratch', Memlet(data='scratch', subset='i, j'))
    scratch = state.add_access('scratch')
    state.add_edge(fill_exit, 'OUT_scratch', scratch, None, Memlet(data='scratch', subset='i, 0:klev'))

    view = state.add_access('scratch_row')
    view.add_in_connector('views')
    state.add_edge(scratch, None, view, 'views', Memlet(data='scratch', subset='i, 0:klev'))

    use_entry, use_exit = state.add_map('use', {'k': '0:klev'})
    use_entry.add_in_connector('IN_row')
    use_entry.add_out_connector('OUT_row')
    use_exit.add_in_connector('IN_pout')
    use_exit.add_out_connector('OUT_pout')
    use = state.add_tasklet('scale', {'a'}, {'o'}, 'o = 3.0 * a')
    state.add_edge(view, None, use_entry, 'IN_row', Memlet(data='scratch_row', subset='0:klev'))
    state.add_edge(use_entry, 'OUT_row', use, 'a', Memlet(data='scratch_row', subset='k'))
    state.add_edge(use, 'o', use_exit, 'IN_pout', Memlet(data='pout', subset='i, k'))
    state.add_edge(use_exit, 'OUT_pout', outer_exit, 'IN_pout', Memlet(data='pout', subset='i, 0:klev'))
    state.add_edge(outer_exit, 'OUT_pout', state.add_write('pout'), None,
                   Memlet.from_array('pout', sdfg.arrays['pout']))

    offload_cloudsc_to_gpu(sdfg)
    assert isinstance(sdfg.arrays['scratch_row'], data.View)
    assert sdfg.arrays['scratch'].storage == dtypes.StorageType.GPU_Global
    cuda = '\n'.join(obj.clean_code for obj in generate_code(sdfg) if obj.language == 'cu')
    assert '__global__ void' in cuda


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
