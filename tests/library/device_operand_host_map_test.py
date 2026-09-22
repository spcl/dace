# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""An expansion over device-resident operands builds device maps, never host ones.

Each of these expansions builds a map of its own beside the vendor call it wraps, and each took
the default schedule: on GPU_Global operands that is host code touching device memory, which
validation rejects and which took down two canon GPU kernels.

* ``Cholesky`` zeroes the unused triangle after the factorization -- cegterg died at the
  ``__inl18_chol`` edge.
* ``TensorTranspose`` falls back to a mapped permute for a dtype hipTensor does not permute (it
  permutes no doubles) -- ls3df_scf died at the ``moveaxis_expr_0_1`` edge.

The identity fill in ``Inv`` and the zero-init in the pure ``TensorDot`` are the same rule, tested
next to their own nodes.
"""
import pytest

import dace
from dace import dtypes
from dace.libraries.linalg import Cholesky
from dace.libraries.linalg.nodes.ttranspose import TensorTranspose

N = 8


def map_schedules(sdfg: dace.SDFG) -> list[tuple[str, dtypes.ScheduleType]]:
    """``(label, schedule)`` for every map in the expanded graph."""
    return [(n.map.label, n.map.schedule) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]


def cholesky_graph(storage: dtypes.StorageType, implementation: str) -> dace.SDFG:
    """``B = cholesky(A)`` with both operands in ``storage``."""
    sdfg = dace.SDFG(f'cholesky_{implementation}_{storage.name}')
    for name in ('A', 'B'):
        sdfg.add_array(name, [N, N], dace.float64, storage=storage)
    state = sdfg.add_state()
    node = Cholesky('chol', lower=True)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_a', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(node, '_b', state.add_write('B'), None, dace.Memlet(f'B[0:{N}, 0:{N}]'))
    sdfg.expand_library_nodes(recursive=False)
    return sdfg


def transpose_graph(storage: dtypes.StorageType) -> dace.SDFG:
    """``B = moveaxis(A, 0, 1)`` in float64, which hipTensor does not permute."""
    sdfg = dace.SDFG(f'ttranspose_{storage.name}')
    sdfg.add_array('A', [4, 5, 6], dace.float64, storage=storage)
    sdfg.add_array('B', [5, 4, 6], dace.float64, storage=storage)
    state = sdfg.add_state()
    node = TensorTranspose('tt', axes=[1, 0, 2])
    node.implementation = 'hipTENSOR'
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_inp_tensor', dace.Memlet('A[0:4, 0:5, 0:6]'))
    state.add_edge(node, '_out_tensor', state.add_write('B'), None, dace.Memlet('B[0:5, 0:4, 0:6]'))
    sdfg.expand_library_nodes()
    return sdfg


@pytest.mark.parametrize('implementation', ['rocSOLVER', 'cuSolverDn'])
def test_the_cholesky_triangle_is_zeroed_on_the_device(implementation):
    sdfg = cholesky_graph(dtypes.StorageType.GPU_Global, implementation)
    sdfg.validate()
    zeroing = [schedule for label, schedule in map_schedules(sdfg) if label.startswith('_uzero_')]
    assert zeroing and all(s == dtypes.ScheduleType.GPU_Device for s in zeroing), map_schedules(sdfg)


def test_a_host_cholesky_keeps_the_default_schedule():
    sdfg = cholesky_graph(dtypes.StorageType.Default, 'OpenBLAS')
    schedules = map_schedules(sdfg)
    assert schedules and all(s == dtypes.ScheduleType.Default for _, s in schedules), schedules


def test_a_device_tensor_transpose_maps_on_the_device():
    sdfg = transpose_graph(dtypes.StorageType.GPU_Global)
    sdfg.validate()
    schedules = map_schedules(sdfg)
    assert schedules and all(s == dtypes.ScheduleType.GPU_Device for _, s in schedules), schedules


def test_a_host_tensor_transpose_keeps_the_default_schedule():
    sdfg = transpose_graph(dtypes.StorageType.Default)
    sdfg.validate()
    schedules = map_schedules(sdfg)
    assert schedules and all(s == dtypes.ScheduleType.Default for _, s in schedules), schedules
