# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The identity a device ``Inv`` solves against is filled by a kernel, not by the host.

The getrs path writes ``I`` into the right-hand side before ``getrs`` overwrites it with the inverse.
That buffer lives in the operand's storage, and the fill map took the default schedule: on
``GPU_Global`` memory that is host code writing device memory, which validation rejects (quatrex_rgf
on the canon GPU column).
"""
import pytest

import dace
from dace import dtypes
from dace.libraries.linalg import Inv

N = 8


def device_inverse(implementation: str) -> dace.SDFG:
    """``B = inv(A)`` through getrf + getrs, both operands on the device."""
    sdfg = dace.SDFG(f'device_inverse_{implementation}')
    sdfg.add_array('A', [N, N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [N, N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    node = Inv('inv', use_getri=False)
    node.implementation = implementation
    state.add_node(node)
    state.add_edge(state.add_read('A'), None, node, '_ain', dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_edge(node, '_aout', state.add_write('B'), None, dace.Memlet(f'B[0:{N}, 0:{N}]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('implementation', ['rocSOLVER', 'cuSolverDn'])
def test_the_device_identity_fill_is_a_kernel(implementation):
    sdfg = device_inverse(implementation)
    sdfg.expand_library_nodes(recursive=False)
    sdfg.validate()
    fills = [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, dace.nodes.MapEntry) and n.map.label.startswith('_eye_')
    ]
    assert fills and all(n.map.schedule == dtypes.ScheduleType.GPU_Device for n in fills), fills
