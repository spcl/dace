# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The unit-stride axis of an imperfect device nest becomes the thread dimension.

Fixtures are the two CloudSC kernel shapes (a bare ``JK { JL }`` nest and one that forwards a
host scalar through the outer scope), a per-row scalar chain, and the nests the pass must refuse.
Each is built at host level and offloaded the way the pipeline does it, so the pass sees the
``GPU_Device { Sequential }`` pair it meets in production.
"""
import json

import dace
import numpy as np
import pytest
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.passes.gpu_specialization.contiguous_axis_to_threads import ContiguousAxisToThreads
from dace.transformation.passes.gpu_specialization.sequentialize_nested_device_scopes import (
    SequentializeNestedDeviceScopes)

K = dace.symbol('K')
L = dace.symbol('L')
ROWS, COLS = 5, 300  # COLS spans three 128-thread blocks with a ragged tail


def nest(name):
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [K, L], dace.float64)
    sdfg.add_array('out', [K, L], dace.float64)
    state = sdfg.add_state()
    outer = state.add_map('outer', dict(k='0:K'))
    inner = state.add_map('inner', dict(l='0:L'))
    return sdfg, state, outer, inner


def bare_nest():
    sdfg = dace.SDFG('bare_nest')
    sdfg.add_array('x', [2, K, L], dace.float64)
    sdfg.add_array('out', [K, L], dace.float64)
    state = sdfg.add_state()
    (oe, ox), (ie, ix) = state.add_map('outer', dict(k='0:K')), state.add_map('inner', dict(l='0:L'))
    body = state.add_tasklet('body', {'u': None, 'v': None}, {'o': None}, 'o = u + v')
    x = state.add_access('x')
    state.add_memlet_path(x, oe, ie, body, dst_conn='u', memlet=dace.Memlet('x[0, k, l]'))
    state.add_memlet_path(x, oe, ie, body, dst_conn='v', memlet=dace.Memlet('x[1, k, l]'))
    state.add_memlet_path(body, ix, ox, state.add_access('out'), src_conn='o', memlet=dace.Memlet('out[k, l]'))
    return sdfg


def forwarded_scalar_nest():
    sdfg, state, (oe, ox), (ie, ix) = nest('forwarded_scalar_nest')
    sdfg.add_scalar('s', dace.float64)
    body = state.add_tasklet('body', {'v': None, 'w': None}, {'o': None}, 'o = v * w')
    forwarded = state.add_access('s')
    state.add_memlet_path(state.add_access('s'), oe, forwarded, memlet=dace.Memlet('s[0]'))
    state.add_memlet_path(forwarded, ie, body, dst_conn='w', memlet=dace.Memlet('s[0]'))
    state.add_memlet_path(state.add_access('a'), oe, ie, body, dst_conn='v', memlet=dace.Memlet('a[k, l]'))
    state.add_memlet_path(body, ix, ox, state.add_access('out'), src_conn='o', memlet=dace.Memlet('out[k, l]'))
    return sdfg


def chain_nest(name='chain_nest', written='out', read_back='a[k, l]'):
    sdfg, state, (oe, ox), (ie, ix) = nest(name)
    sdfg.add_scalar('t', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    pre = state.add_tasklet('pre', {'x': None}, {'y': None}, 'y = 2.0 * x')
    body = state.add_tasklet('body', {'v': None, 's': None}, {'o': None}, 'o = v + s')
    a, t = state.add_access('a'), state.add_access('t')
    state.add_memlet_path(a, oe, pre, dst_conn='x', memlet=dace.Memlet('a[k, 0]'))
    state.add_edge(pre, 'y', t, None, dace.Memlet('t[0]'))
    state.add_memlet_path(t, ie, body, dst_conn='s', memlet=dace.Memlet('t[0]'))
    state.add_memlet_path(a, oe, ie, body, dst_conn='v', memlet=dace.Memlet(read_back))
    state.add_memlet_path(body, ix, ox, state.add_access(written), src_conn='o', memlet=dace.Memlet(f'{written}[k, l]'))
    return sdfg


def read_of_written_nest():
    """Row k's scalar reads ``a[k, 0]``, which lane 0 of the same row overwrites."""
    return chain_nest('read_of_written_nest', written='a')


def per_row_reduction_nest():
    sdfg, state, (oe, ox), (ie, ix) = nest('per_row_reduction_nest')
    sdfg.add_scalar('t', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    re, rx = state.add_map('reduce', dict(r='0:L'))
    zero = state.add_tasklet('zero', {}, {'z': None}, 'z = 0.0')
    term = state.add_tasklet('term', {'x': None}, {'y': None}, 'y = x')
    body = state.add_tasklet('body', {'v': None, 's': None}, {'o': None}, 'o = v / s')
    a, t_init, t_sum = state.add_access('a'), state.add_access('t'), state.add_access('t')
    state.add_edge(oe, None, zero, None, dace.Memlet())
    state.add_edge(zero, 'z', t_init, None, dace.Memlet('t[0]'))
    state.add_edge(t_init, None, re, None, dace.Memlet())
    state.add_memlet_path(a, oe, re, term, dst_conn='x', memlet=dace.Memlet('a[k, r]'))
    state.add_memlet_path(term, rx, t_sum, src_conn='y', memlet=dace.Memlet('t[0]', wcr='lambda p, q: p + q'))
    state.add_memlet_path(t_sum, ie, body, dst_conn='s', memlet=dace.Memlet('t[0]'))
    state.add_memlet_path(a, oe, ie, body, dst_conn='v', memlet=dace.Memlet('a[k, l]'))
    state.add_memlet_path(body, ix, ox, state.add_access('out'), src_conn='o', memlet=dace.Memlet('out[k, l]'))
    return sdfg


def per_row_tasklet_reduction_nest():
    sdfg, state, (oe, ox), (ie, ix) = nest('per_row_tasklet_reduction_nest')
    sdfg.add_scalar('t', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    total = state.add_tasklet('total', {'row': None}, {'y': None}, 'y = 0.0\nfor i in range(L):\n    y = y + row[i]')
    body = state.add_tasklet('body', {'v': None, 's': None}, {'o': None}, 'o = v / s')
    a, t = state.add_access('a'), state.add_access('t')
    state.add_memlet_path(a, oe, total, dst_conn='row', memlet=dace.Memlet('a[k, 0:L]'))
    state.add_edge(total, 'y', t, None, dace.Memlet('t[0]'))
    state.add_memlet_path(t, ie, body, dst_conn='s', memlet=dace.Memlet('t[0]'))
    state.add_memlet_path(a, oe, ie, body, dst_conn='v', memlet=dace.Memlet('a[k, l]'))
    state.add_memlet_path(body, ix, ox, state.add_access('out'), src_conn='o', memlet=dace.Memlet('out[k, l]'))
    return sdfg


def per_row_output_nest():
    sdfg, state, (oe, ox), (ie, ix) = nest('per_row_output_nest')
    sdfg.add_array('first', [K], dace.float64)
    pre = state.add_tasklet('pre', {'x': None}, {'y': None}, 'y = x')
    body = state.add_tasklet('body', {'v': None}, {'o': None}, 'o = v')
    a = state.add_access('a')
    state.add_memlet_path(a, oe, pre, dst_conn='x', memlet=dace.Memlet('a[k, 0]'))
    state.add_memlet_path(pre, ox, state.add_access('first'), src_conn='y', memlet=dace.Memlet('first[k]'))
    state.add_memlet_path(a, oe, ie, body, dst_conn='v', memlet=dace.Memlet('a[k, l]'))
    state.add_memlet_path(body, ix, ox, state.add_access('out'), src_conn='o', memlet=dace.Memlet('out[k, l]'))
    return sdfg


def threads_already_contiguous_nest():
    """The CloudSC ``single_state_body_28`` shape: threads walk ``l``, each looping over ``k``."""
    sdfg = dace.SDFG('threads_already_contiguous_nest')
    sdfg.add_array('a', [K, L], dace.float64)
    sdfg.add_array('out', [K, L], dace.float64)
    state = sdfg.add_state()
    (oe, ox), (ie, ix) = state.add_map('outer', dict(l='0:L')), state.add_map('inner', dict(k='0:K'))
    body = state.add_tasklet('body', {'v': None}, {'o': None}, 'o = 3.0 * v')
    state.add_memlet_path(state.add_access('a'), oe, ie, body, dst_conn='v', memlet=dace.Memlet('a[k, l]'))
    state.add_memlet_path(body, ix, ox, state.add_access('out'), src_conn='o', memlet=dace.Memlet('out[k, l]'))
    return sdfg


def offloaded(builder):
    sdfg = builder()
    sdfg.validate()
    sdfg.apply_gpu_transformations(simplify=False)
    SequentializeNestedDeviceScopes().apply_pass(sdfg, {})
    return sdfg


def kernels(sdfg):
    return [(n, s) for n, s in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry) and s.entry_node(n) is None]


def only_kernel(sdfg):
    (entry, state), = kernels(sdfg)
    assert entry.map.schedule == dtypes.ScheduleType.GPU_Device, entry.map.schedule
    return entry, state


@pytest.mark.parametrize('builder', [bare_nest, forwarded_scalar_nest, chain_nest])
def test_the_unit_stride_axis_becomes_the_fastest_thread_index(builder):
    """The launch maps the LAST parameter to threadIdx.x, so that is where ``l`` has to land."""
    sdfg = offloaded(builder)
    assert ContiguousAxisToThreads().apply_pass(sdfg, {}) == 1
    sdfg.validate()
    entry, state = only_kernel(sdfg)
    assert entry.map.params == ['k', 'l'], entry.map.params
    nested = [n for n in state.scope_subgraph(entry).nodes() if isinstance(n, nodes.MapEntry) and n is not entry]
    assert not nested, nested


def test_a_forwarded_scalar_is_read_through_the_kernel_entry():
    """The CloudSC ``single_state_body_30`` shape: a per-row access node that moves no data."""
    sdfg = offloaded(forwarded_scalar_nest)
    ContiguousAxisToThreads().apply_pass(sdfg, {})
    entry, state = only_kernel(sdfg)
    inside = [n for n in state.scope_subgraph(entry).data_nodes()]
    assert not inside, [n.data for n in inside]


def test_a_per_row_scalar_chain_is_recomputed_in_every_lane():
    sdfg = offloaded(chain_nest)
    ContiguousAxisToThreads().apply_pass(sdfg, {})
    entry, state = only_kernel(sdfg)
    pre = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet) and n.label == 'pre')
    assert state.entry_node(pre) is entry
    assert [e.data.subset for e in state.in_edges(pre)] == [dace.subsets.Range.from_string('k, 0')]


@pytest.mark.parametrize('builder', [
    read_of_written_nest, per_row_reduction_nest, per_row_tasklet_reduction_nest, per_row_output_nest,
    threads_already_contiguous_nest
])
def test_a_nest_the_sink_or_collapse_would_change_is_left_untouched(builder):
    """Recomputing per lane is only sound when every lane would see the value the one per-row
    evaluation saw and nothing but the lanes consume it; a refusal must not leave half a rewrite."""
    sdfg = offloaded(builder)
    before = json.dumps(sdfg.to_json(), sort_keys=True)
    assert ContiguousAxisToThreads().apply_pass(sdfg, {}) is None
    assert json.dumps(sdfg.to_json(), sort_keys=True) == before


def reference(builder, inputs):
    a = inputs.get('a')
    return {
        bare_nest: lambda: inputs['x'][0] + inputs['x'][1],
        forwarded_scalar_nest: lambda: a * inputs['s'],
        chain_nest: lambda: a + 2.0 * a[:, :1],
    }[builder]()


@pytest.mark.gpu
@pytest.mark.parametrize('builder', [bare_nest, forwarded_scalar_nest, chain_nest])
def test_the_collapsed_kernel_computes_what_the_nest_computed(builder):
    rng = np.random.default_rng(0)
    inputs = {'x': rng.random((2, ROWS, COLS))} if builder is bare_nest else {'a': rng.random((ROWS, COLS))}
    if builder is forwarded_scalar_nest:
        inputs['s'] = 1.5
    sdfg = offloaded(builder)
    assert ContiguousAxisToThreads().apply_pass(sdfg, {}) == 1
    only_kernel(sdfg)[0].map.gpu_block_size = [128, 1, 1]
    out = np.zeros((ROWS, COLS))
    sdfg(**inputs, out=out, K=ROWS, L=COLS)
    np.testing.assert_allclose(out, reference(builder, inputs), rtol=0, atol=0)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
