# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which GPU expansion a ``Sequential`` library node gets, decided by SCOPE.

:func:`~dace.transformation.auto.auto_optimize.set_fast_implementations` reads a ``Sequential``
schedule on GPU as "inside a kernel" and pins the node to ``pure``, the only lowering that may be
emitted as device code. A host loop and a taskloop body are ``Sequential`` too, and there ``pure``
is host code operating on ``GPU_Global`` memory -- the polybench trisolv and npbench stockham_fft
failure. :func:`~dace.transformation.passes.canonicalize.finalize.canonicalize_set_fast_implementations`
therefore decides from the enclosing scopes instead, which is what the cases below pin.
"""
import dace
from dace import dtypes
from dace.libraries.blas.nodes.dot import Dot
from dace.libraries.standard.nodes.reduce import Reduce
from dace.transformation.passes.canonicalize.finalize import canonicalize_set_fast_implementations


def toplevel_sequential_reduce() -> dace.SDFG:
    """A ``Reduce`` nobody encloses, carrying the schedule a nested node would carry."""
    sdfg = dace.SDFG('toplevel_sequential_reduce')
    sdfg.add_array('A', [256], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [1], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Reduce('reduce_sum', wcr='lambda a, b: a + b', axes=None, identity=0.0)
    node.schedule = dtypes.ScheduleType.Sequential
    state.add_node(node)
    state.add_edge(state.add_access('A'), None, node, None, dace.Memlet('A[0:256]'))
    state.add_edge(node, None, state.add_access('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg, node


def in_kernel_sequential_reduce() -> dace.SDFG:
    """The same node under a ``GPU_Device`` map: this one really is device code."""
    sdfg = dace.SDFG('in_kernel_sequential_reduce')
    sdfg.add_array('A', [8, 256], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [8], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('row', [256], dace.float64, transient=True, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('rows', dict(i='0:8'), schedule=dtypes.ScheduleType.GPU_Device)
    node = Reduce('reduce_sum', wcr='lambda a, b: a + b', axes=None, identity=0.0)
    node.schedule = dtypes.ScheduleType.Sequential
    row = state.add_access('row')
    state.add_memlet_path(state.add_read('A'), entry, row, memlet=dace.Memlet('A[i, 0:256]', other_subset='0:256'))
    state.add_edge(row, None, node, None, dace.Memlet('row[0:256]'))
    state.add_memlet_path(node, exit_node, state.add_write('out'), memlet=dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg, node


def dot_in_a_host_loop() -> dace.SDFG:
    """The trisolv shape: a ``Dot`` re-entered by a top-level loop, so nothing runs it in a kernel."""
    sdfg = dace.SDFG('dot_in_a_host_loop')
    sdfg.add_array('x', [8, 256], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('y', [256], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('out', [8], dace.float64, storage=dtypes.StorageType.GPU_Global)
    loop = dace.sdfg.state.LoopRegion('rows', 'i < 8', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    node = Dot('dot')
    node.schedule = dtypes.ScheduleType.Sequential
    state.add_node(node)
    state.add_edge(state.add_read('x'), None, node, '_x', dace.Memlet('x[i, 0:256]'))
    state.add_edge(state.add_read('y'), None, node, '_y', dace.Memlet('y[0:256]'))
    state.add_edge(node, '_result', state.add_write('out'), None, dace.Memlet('out[i]'))
    sdfg.validate()
    return sdfg, node


def test_a_sequential_node_at_a_host_level_calls_the_device_library():
    """Nothing encloses it, so it is host code, and host code is what issues a device call."""
    sdfg, node = toplevel_sequential_reduce()
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    assert node.implementation in Reduce.implementations, node.implementation
    assert node.implementation != 'pure', 'a host-level node kept the in-kernel lowering'


def test_a_sequential_node_inside_a_kernel_keeps_the_pure_expansion():
    """A device call cannot be issued from inside a kernel, so this one must stay ``pure``."""
    sdfg, node = in_kernel_sequential_reduce()
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    assert node.implementation == 'pure', node.implementation


def test_a_loop_does_not_make_its_body_device_code():
    """A loop re-enters the node, which is why it is ``Sequential`` -- it does not put it on a device."""
    sdfg, node = dot_in_a_host_loop()
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    assert node.implementation == 'cuBLAS', node.implementation


if __name__ == '__main__':
    test_a_sequential_node_at_a_host_level_calls_the_device_library()
    test_a_sequential_node_inside_a_kernel_keeps_the_pure_expansion()
    test_a_loop_does_not_make_its_body_device_code()
