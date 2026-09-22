# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A vendor tensor contraction the library cannot do in the operands' type falls back to ``pure``.

hipTensor contracts ``float16`` and ``float32`` only, and its expansion RAISED for ``float64``: every
double contraction on the HIP canon GPU column was unsupported (cp2k_grid_integrate, ls3df_scf).
The environment already documents the pure expansion as the fallback, which on GPU-resident operands
is still a device map.
"""
import dace
from dace import dtypes
from dace.libraries.linalg.nodes.tensordot import TensorDot


def double_contraction(implementation: str) -> dace.SDFG:
    """``C[i, k] = sum_j A[i, j] * B[j, k]`` on the device, in ``float64``."""
    sdfg = dace.SDFG(f'double_tensordot_{implementation}')
    for name, shape in (('A', [4, 5]), ('B', [5, 6]), ('C', [4, 6])):
        sdfg.add_array(name, shape, dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = TensorDot('contract', left_axes=[1], right_axes=[0])
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_left_tensor', dace.Memlet('A[0:4, 0:5]'))
    state.add_edge(state.add_read('B'), None, node, '_right_tensor', dace.Memlet('B[0:5, 0:6]'))
    state.add_edge(node, '_out_tensor', state.add_write('C'), None, dace.Memlet('C[0:4, 0:6]'))
    sdfg.validate()
    return sdfg


def test_a_double_hiptensor_contraction_expands_to_the_pure_map():
    sdfg = double_contraction('hipTENSOR')
    sdfg.expand_library_nodes()
    code = '\n'.join(n.code.as_string for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet))
    assert 'hiptensor' not in code.lower(), code[:400]
    assert any(isinstance(n, dace.nodes.MapEntry) for n, _ in sdfg.all_nodes_recursive())


def test_the_pure_fallback_maps_device_operands_on_the_device():
    """Both maps it builds, the zero-init and the contraction, must run on the device.

    The init map took the default schedule, so the fallback wrote GPU_Global memory from host code
    and validation rejected the expansion: ls3df_scf's canon GPU run died on
    ``Data container "__inl9_acc_0" is stored as StorageType.GPU_Global but accessed on host``.
    """
    sdfg = double_contraction('hipTENSOR')
    sdfg.expand_library_nodes()
    sdfg.validate()
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert maps and all(n.map.schedule == dtypes.ScheduleType.GPU_Device for n in maps), \
        [(n.map.label, n.map.schedule) for n in maps]


def test_a_host_contraction_keeps_the_default_schedule():
    """Nothing changes for host operands: the schedule follows the storage."""
    sdfg = dace.SDFG('host_tensordot')
    for name, shape in (('A', [4, 5]), ('B', [5, 6]), ('C', [4, 6])):
        sdfg.add_array(name, shape, dace.float64)
    state = sdfg.add_state()
    node = TensorDot('contract', left_axes=[1], right_axes=[0])
    node.implementation = 'pure'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_left_tensor', dace.Memlet('A[0:4, 0:5]'))
    state.add_edge(state.add_read('B'), None, node, '_right_tensor', dace.Memlet('B[0:5, 0:6]'))
    state.add_edge(node, '_out_tensor', state.add_write('C'), None, dace.Memlet('C[0:4, 0:6]'))
    sdfg.expand_library_nodes()
    sdfg.validate()
    maps = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]
    assert maps and all(n.map.schedule == dtypes.ScheduleType.Default for n in maps), \
        [(n.map.label, n.map.schedule) for n in maps]
