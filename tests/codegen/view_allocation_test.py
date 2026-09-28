# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests that views are allocated (bound) in the scope of their access node, with Scope lifetime."""
import re

import numpy as np
import pytest

import dace
from dace import registry
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.targets.cpp import sym2cpp

N = 20

if not hasattr(dace.ScheduleType, 'ViewTestLoop'):
    dace.ScheduleType.register('ViewTestLoop')
    dace.SCOPEDEFAULT_SCHEDULE[dace.ScheduleType.ViewTestLoop] = dace.ScheduleType.Sequential
    dace.SCOPEDEFAULT_STORAGE[dace.ScheduleType.ViewTestLoop] = dace.StorageType.CPU_Heap

    @registry.autoregister_params(name='view_test_loop')
    class ViewTestLoop(TargetCodeGenerator):
        """A sequential loop in which CPU heap memory cannot be allocated."""

        def __init__(self, frame_codegen, sdfg):
            self.frame = frame_codegen
            self.dispatcher = frame_codegen.dispatcher
            self.dispatcher.register_map_dispatcher(dace.ScheduleType.ViewTestLoop, self)

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


def test_view_in_custom_scope():
    """A view in a scope that cannot allocate its storage is still bound inside the scope."""
    sdfg = dace.SDFG('view_in_custom_scope')
    sdfg.add_array('A', [N, 3], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_view('V', [3], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', dict(i=f'0:{N}'), schedule=dace.ScheduleType.ViewTestLoop)
    view = state.add_access('V')
    tasklet = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp[0] + inp[1] + inp[2]')
    state.add_memlet_path(state.add_read('A'), me, view, memlet=dace.Memlet('A[i, 0:3]'))
    state.add_edge(view, None, tasklet, 'inp', dace.Memlet('V[0:3]'))
    state.add_memlet_path(tasklet, mx, state.add_write('B'), src_conn='out', memlet=dace.Memlet('B[i]'))
    sdfg.validate()

    A = np.random.rand(N, 3)
    B = np.zeros(N)
    sdfg(A=A, B=B)
    assert np.allclose(B, A.sum(axis=1))


def test_view_in_gpu_kernel():
    """A view of GPU memory inside a GPU kernel is bound inside the kernel. Only generates code."""
    sdfg = dace.SDFG('view_in_gpu_kernel')
    sdfg.add_array('A', [8, 4], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('B', [8], dace.float64, storage=dace.StorageType.GPU_Global)
    sdfg.add_view('V', [4], dace.float64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state()
    me, mx = state.add_map('kernel', dict(i='0:8'), schedule=dace.ScheduleType.GPU_Device)
    view = state.add_access('V')
    tasklet = state.add_tasklet('pick', {'inp'}, {'out'}, 'out = inp[1]')
    state.add_memlet_path(state.add_read('A'), me, view, memlet=dace.Memlet('A[i, 0:4]'))
    state.add_edge(view, None, tasklet, 'inp', dace.Memlet('V[0:4]'))
    state.add_memlet_path(tasklet, mx, state.add_write('B'), src_conn='out', memlet=dace.Memlet('B[i]'))

    code = sdfg.generate_code()
    frame = next(c for c in code if c.title == 'Frame').clean_code
    device = ''.join(c.clean_code for c in code if c.language == 'cu')
    binding = re.compile(r'\bV = &A\[')
    assert binding.search(device)
    assert not binding.search(frame)


def test_view_lifetime_is_scope():
    """Views and references of a container of any lifetime have Scope lifetime."""
    for lifetime in dace.AllocationLifetime:
        viewed = dace.data.Array(dace.float64, [20], lifetime=lifetime)
        view = dace.data.View.view(viewed)
        assert view.lifetime == dace.AllocationLifetime.Scope
        view.validate()
        reference = dace.data.Reference.view(viewed)
        assert reference.lifetime == dace.AllocationLifetime.Scope
        reference.validate()


def test_persistent_view_invalid():
    """A lifetime other than Scope on a view is rejected by validation."""
    sdfg = dace.SDFG('persistent_view_invalid')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_view('V', [10, 2], dace.float64)
    state = sdfg.add_state()
    state.add_edge(state.add_read('A'), None, state.add_access('V'), 'views', dace.Memlet('A'))
    sdfg.validate()

    sdfg.arrays['V'].lifetime = dace.AllocationLifetime.Persistent
    with pytest.raises(dace.sdfg.InvalidSDFGError):
        sdfg.validate()


if __name__ == '__main__':
    test_view_in_custom_scope()
    test_view_in_gpu_kernel()
    test_view_lifetime_is_scope()
    test_persistent_view_invalid()
