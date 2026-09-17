# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests nested SDFG transients whose allocation is placed in an ancestor scope."""
import numpy as np
import pytest

import dace
from dace import registry
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.targets.cpp import sym2cpp

N = 20

if not hasattr(dace.ScheduleType, 'HoistingTestLoop'):
    dace.ScheduleType.register('HoistingTestLoop')
    dace.SCOPEDEFAULT_SCHEDULE[dace.ScheduleType.HoistingTestLoop] = dace.ScheduleType.Sequential
    dace.SCOPEDEFAULT_STORAGE[dace.ScheduleType.HoistingTestLoop] = dace.StorageType.CPU_Heap

    @registry.autoregister_params(name='hoisting_test_loop')
    class HoistingTestLoop(TargetCodeGenerator):
        """A sequential loop in which CPU heap memory cannot be allocated."""

        def __init__(self, frame_codegen, sdfg):
            self.frame = frame_codegen
            self.dispatcher = frame_codegen.dispatcher
            self.dispatcher.register_map_dispatcher(dace.ScheduleType.HoistingTestLoop, self)

        def generate_scope(self, sdfg, cfg, scope, state_id, function_stream, callsite_stream):
            entry_node = scope.source_nodes()[0]
            callsite_stream.write('{', sdfg, state_id, entry_node)
            for param, rng in zip(entry_node.map.params, entry_node.map.range):
                begin, end, stride = (sym2cpp(r) for r in rng)
                callsite_stream.write(f'for (auto {param} = {begin}; {param} <= {end}; {param} += {stride}) {{', sdfg,
                                      state_id, entry_node)
            self.frame.allocate_arrays_in_scope(sdfg, cfg, entry_node, function_stream, callsite_stream)
            self.dispatcher.dispatch_subgraph(sdfg,
                                              cfg,
                                              scope,
                                              state_id,
                                              function_stream,
                                              callsite_stream,
                                              skip_entry_node=True)


def _build(scalar: bool) -> dace.SDFG:
    inner = dace.SDFG('inner')
    inner.add_symbol('i', dace.int64)
    inner.add_array('a', [N], dace.float64)
    inner.add_array('b', [N], dace.float64)
    if scalar:
        inner.add_scalar('tmp', dace.float64, transient=True)
        tmp_memlet = 'tmp'
    else:
        inner.add_array('tmp', [5], dace.float64, transient=True)
        tmp_memlet = 'tmp[3]'
    state = inner.add_state()
    fill = state.add_tasklet('fill', {'inp'}, {'out'}, 'out = inp')
    use = state.add_tasklet('use', {'inp'}, {'out'}, 'out = 2 * inp')
    tmp = state.add_access('tmp')
    state.add_edge(state.add_read('a'), None, fill, 'inp', dace.Memlet('a[i]'))
    state.add_edge(fill, 'out', tmp, None, dace.Memlet(tmp_memlet))
    state.add_edge(tmp, None, use, 'inp', dace.Memlet(tmp_memlet))
    state.add_edge(use, 'out', state.add_write('b'), None, dace.Memlet('b[i]'))

    sdfg = dace.SDFG(f'hoisted_transient_{"scalar" if scalar else "array"}')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'), schedule=dace.ScheduleType.HoistingTestLoop)
    node = state.add_nested_sdfg(inner, {'a'}, {'b'}, {'i': 'i'})
    state.add_memlet_path(state.add_read('A'), me, node, dst_conn='a', memlet=dace.Memlet(f'A[0:{N}]'))
    state.add_memlet_path(node, mx, state.add_write('B'), src_conn='b', memlet=dace.Memlet(f'B[0:{N}]'))
    return sdfg


@pytest.mark.parametrize('scalar', [True, False])
def test_nested_sdfg_hoisted_transient(scalar):
    sdfg = _build(scalar)
    sdfg.validate()
    A = np.random.rand(N)
    B = np.zeros(N)
    sdfg(A=A, B=B)
    assert np.allclose(B, 2 * A)


if __name__ == '__main__':
    test_nested_sdfg_hoisted_transient(True)
    test_nested_sdfg_hoisted_transient(False)
