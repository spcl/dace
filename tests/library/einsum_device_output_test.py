# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A pure einsum whose output lives on the device builds device maps.

``create_einsum_sdfg`` falls back to mapped tasklets whenever the contraction is not a plain
(batched) GEMM, and both maps it builds, the reset and the contraction, took the default schedule.
On a GPU-resident output that is host code writing GPU_Global memory, which validation rejects:
cp2k_density_matrix_trs4's canon GPU run died on ``Data container "__out" is stored as
StorageType.GPU_Global but accessed on host`` at the ``einsum_reset`` edge.
"""
import pytest

import dace
from dace import dtypes
from dace.frontend.common.einsum import create_einsum_sdfg

#: Contractions no (batched) GEMM covers, so each takes the mapped path: an elementwise product
#: and a three-operand chain.
CONTRACTIONS = {
    'elementwise': ('ij,ij->ij', {
        'A': [2, 3],
        'B': [2, 3],
        'out': [2, 3]
    }),
    'three_operand': ('ij,jk,kl->il', {
        'A': [2, 3],
        'B': [3, 4],
        'C': [4, 5],
        'out': [2, 5]
    }),
}


def einsum_graph(case: str, storage: dtypes.StorageType) -> dace.SDFG:
    """The einsum of ``case``, with every operand in ``storage``."""
    subscript, shapes = CONTRACTIONS[case]
    sdfg = dace.SDFG(f'einsum_{case}_{storage.name}')
    for name, shape in shapes.items():
        sdfg.add_array(name, shape, dace.float64, storage=storage)
    state = sdfg.add_state()
    operands = [name for name in shapes if name != 'out']
    create_einsum_sdfg(sdfg, state, subscript, *operands, output='out')
    return sdfg


def map_schedules(sdfg: dace.SDFG) -> list[tuple[str, dtypes.ScheduleType]]:
    """``(label, schedule)`` for every map in the graph."""
    return [(n.map.label, n.map.schedule) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry)]


@pytest.mark.parametrize('case', sorted(CONTRACTIONS))
def test_a_device_einsum_maps_on_the_device(case):
    sdfg = einsum_graph(case, dtypes.StorageType.GPU_Global)
    sdfg.validate()
    schedules = map_schedules(sdfg)
    assert schedules, 'the einsum took a library node, not the mapped path'
    assert all(schedule == dtypes.ScheduleType.GPU_Device for _, schedule in schedules), schedules


@pytest.mark.parametrize('case', sorted(CONTRACTIONS))
def test_a_host_einsum_keeps_the_default_schedule(case):
    sdfg = einsum_graph(case, dtypes.StorageType.Default)
    sdfg.validate()
    schedules = map_schedules(sdfg)
    assert schedules and all(schedule == dtypes.ScheduleType.Default for _, schedule in schedules), schedules


def test_the_reset_map_is_scheduled_with_the_contraction():
    """The reset is a separate map in its own state; leaving it on the host is what failed."""
    sdfg = einsum_graph('elementwise', dtypes.StorageType.GPU_Global)
    resets = [schedule for label, schedule in map_schedules(sdfg) if 'reset' in label]
    assert resets and all(s == dtypes.ScheduleType.GPU_Device for s in resets), map_schedules(sdfg)
