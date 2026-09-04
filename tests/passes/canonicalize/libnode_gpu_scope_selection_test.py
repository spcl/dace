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
from dace.libraries.standard.nodes.arg_reduce import ArgReduce
from dace.libraries.standard.nodes.merge_node import MergeLibraryNode
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
    state.add_edge(state.add_access('A'), None, node, '_in', dace.Memlet('A[0:256]'))
    state.add_edge(node, '_out', state.add_access('out'), None, dace.Memlet('out[0]'))
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
    state.add_edge(row, None, node, '_in', dace.Memlet('row[0:256]'))
    state.add_memlet_path(node, exit_node, state.add_write('out'), memlet=dace.Memlet('out[i]'), src_conn='_out')
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


def vendor_blas() -> str:
    """The GPU BLAS this host's backend selects -- ``cuBLAS`` on CUDA, ``rocBLAS`` on HIP.

    Asked rather than assumed: ``canonicalize_fast_library_priority`` takes its GPU row straight
    from ``auto_optimize``, which is keyed on ``get_gpu_backend()``. Hardcoding one vendor makes the
    test assert the machine it was written on -- it failed here reading ``rocBLAS`` on an AMD node,
    which is the pipeline doing exactly the right thing.
    """
    from dace.codegen.common import get_gpu_backend
    try:
        return 'rocBLAS' if get_gpu_backend() == 'hip' else 'cuBLAS'
    except Exception:  # noqa: BLE001 -- no backend configured; the CUDA row is the historical default
        return 'cuBLAS'


def test_a_loop_does_not_make_its_body_device_code():
    """A loop re-enters the node, which is why it is ``Sequential`` -- it does not put it on a device."""
    sdfg, node = dot_in_a_host_loop()
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    assert node.implementation == vendor_blas(), node.implementation


def merge_in_a_host_loop():
    """The npbench bfs shape: a pure-only node re-entered by a host loop over device memory.

    ``MergeLibraryNode`` publishes no device expansion, so the implementation rule the cases above
    pin has nothing to select and the node keeps whatever schedule it arrived with.
    """
    sdfg = dace.SDFG('merge_in_a_host_loop')
    for name in ('t', 'f', 'out'):
        sdfg.add_array(name, [256], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('mask', [256], dace.bool_, storage=dtypes.StorageType.GPU_Global)
    loop = dace.sdfg.state.LoopRegion('sweeps', 'i < 8', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    state = loop.add_state('body', is_start_block=True)
    node = MergeLibraryNode('_where_')
    node.schedule = dtypes.ScheduleType.Sequential
    state.add_node(node)
    state.add_edge(state.add_read('t'), None, node, node.TRUE_CONNECTOR_NAME, dace.Memlet('t[0:256]'))
    state.add_edge(state.add_read('f'), None, node, node.FALSE_CONNECTOR_NAME, dace.Memlet('f[0:256]'))
    state.add_edge(state.add_read('mask'), None, node, node.MASK_CONNECTOR_NAME, dace.Memlet('mask[0:256]'))
    state.add_edge(node, node.OUTPUT_CONNECTOR_NAME, state.add_write('out'), None, dace.Memlet('out[0:256]'))
    sdfg.validate()
    return sdfg, node


def test_a_pure_only_node_in_a_host_loop_becomes_a_kernel():
    """With no device expansion to pick, the SCHEDULE is what has to move.

    Left ``Sequential`` the pure expansion lowers as a host map indexing ``GPU_Global`` operands,
    and validation rejects the SDFG outright rather than running it slowly -- npbench bfs died on
    exactly that, reported as a runtime error with no wrong number to trace.
    """
    sdfg, node = merge_in_a_host_loop()
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    assert node.schedule == dtypes.ScheduleType.GPU_Device, node.schedule

    sdfg.expand_library_nodes()
    maps = [n.map for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.sdfg.nodes.MapEntry)]
    assert maps, 'the expansion produced no map at all'
    assert all(m.schedule == dtypes.ScheduleType.GPU_Device for m in maps), [str(m.schedule) for m in maps]


def arg_reduce_at_host_level(stride: int = 1, transform: str = ''):
    """An ``ArgReduce`` nobody encloses, reading ``a[0:N:stride]`` through ``transform``."""
    sdfg = dace.SDFG(f'arg_reduce_{stride}_{transform or "id"}')
    sdfg.add_array('a', [768], dace.float64, storage=dtypes.StorageType.GPU_Global)
    # Host scalars, per ``ArgReduce.host_connectors``: every expansion answers on the host.
    sdfg.add_array('val', [1], dace.float64)
    sdfg.add_array('idx', [1], dace.int64)
    state = sdfg.add_state()
    node = ArgReduce('argmax', op='max', transform=transform)
    node.schedule = dtypes.ScheduleType.Sequential
    state.add_node(node)
    state.add_edge(state.add_read('a'), None, node, '_in', dace.Memlet(f'a[0:768:{stride}]'))
    state.add_edge(node, '_out_val', state.add_write('val'), None, dace.Memlet('val[0]'))
    state.add_edge(node, '_out_idx', state.add_write('idx'), None, dace.Memlet('idx[0]'))
    sdfg.validate()
    return sdfg, node


def test_a_contiguous_arg_reduce_takes_the_cub_expansion():
    # Unit stride and no transform is exactly what gpucub::DeviceReduce::ArgMax reads, so the
    # host-level node takes the device library call like any other.
    sdfg, node = arg_reduce_at_host_level()
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    assert node.implementation == 'CUDA', node.implementation


def cuda_unit(sdfg):
    with dace.config.set_temporary('compiler', 'cuda', 'implementation', value='experimental'):
        return '\n'.join(obj.clean_code for obj in sdfg.generate_code() if obj.language == 'cu')


def test_a_strided_arg_reduce_still_takes_the_cub_expansion():
    """A strided ``_in`` is a device lowering too: CUB reduces over an input ITERATOR, not a pointer.

    tsvc s318 spells its operand ``argmax |a[inc*i]|``, and the answer used to be the ``pure`` serial
    scan because ``ExpandArgReduceCUDA`` refused anything it could not hand CUB as a raw pointer.
    ``dace::cub::gather_iterator`` presents ``j -> base[j * stride]`` instead, so the same one
    streaming pass serves it and the selector no longer has a refusal to route around.
    """
    sdfg, node = arg_reduce_at_host_level(stride=3)
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    assert node.implementation == 'CUDA', node.implementation
    device_code = cuda_unit(sdfg)
    assert 'gather_iterator<::dace::cub::IdentityXf>' in device_code, device_code
    assert '__ar_best' not in device_code, ('the serial scan is still emitted, so the node did not take the '
                                            f'device library call:\n{device_code}')


def test_a_transformed_arg_reduce_gathers_through_its_transform():
    # The transform is read per element, which the gather functor composes rather than refuses -- and
    # it must not depend on the stride being the thing that is odd, so this one is contiguous.
    sdfg, node = arg_reduce_at_host_level(transform='abs')
    canonicalize_set_fast_implementations(sdfg, dtypes.DeviceType.GPU)
    assert node.implementation == 'CUDA', node.implementation
    assert 'gather_iterator<::dace::cub::AbsXf>' in cuda_unit(sdfg)


if __name__ == '__main__':
    test_a_sequential_node_at_a_host_level_calls_the_device_library()
    test_a_contiguous_arg_reduce_takes_the_cub_expansion()
    test_a_strided_arg_reduce_still_takes_the_cub_expansion()
    test_a_transformed_arg_reduce_gathers_through_its_transform()
    test_a_sequential_node_inside_a_kernel_keeps_the_pure_expansion()
    test_a_loop_does_not_make_its_body_device_code()
    test_a_pure_only_node_in_a_host_loop_becomes_a_kernel()
