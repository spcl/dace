# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Only host heap and device global memory may become ``Persistent``."""
import pytest

import dace
from dace import dtypes
from dace.transformation.auto.auto_optimize import make_transients_persistent

PERSISTENT = dtypes.AllocationLifetime.Persistent


def sdfg_with_transient(storage: dtypes.StorageType, lifetime: dtypes.AllocationLifetime, in_map: bool) -> dace.SDFG:
    sdfg = dace.SDFG('persistence_by_storage')
    sdfg.add_array('buffer', [5], dace.float64, transient=True, storage=storage, lifetime=lifetime)
    state = sdfg.add_state()
    access = state.add_access('buffer')
    if in_map:
        entry, exit_node = state.add_map('loop', dict(i='0:5'))
        state.add_nedge(entry, access, dace.Memlet())
        state.add_nedge(access, exit_node, dace.Memlet())
    return sdfg


@pytest.mark.parametrize('storage,persistent', [
    (dtypes.StorageType.CPU_Heap, True),
    (dtypes.StorageType.GPU_Global, True),
    (dtypes.StorageType.Default, True),
    (dtypes.StorageType.GPU_Shared, False),
    (dtypes.StorageType.CPU_Pinned, False),
    (dtypes.StorageType.Register, False),
])
def test_a_top_level_transient_is_persistent_only_on_heap_or_global_storage(storage, persistent):
    sdfg = sdfg_with_transient(storage, dtypes.AllocationLifetime.Scope, in_map=False)

    make_transients_persistent(sdfg, dace.DeviceType.CPU)

    assert (sdfg.arrays['buffer'].lifetime == PERSISTENT) is persistent


@pytest.mark.parametrize('storage,persistent', [
    (dtypes.StorageType.CPU_Heap, True),
    (dtypes.StorageType.Default, False),
])
def test_a_default_transient_inside_a_map_is_not_persistent(storage, persistent):
    """Inside a map, ``Default`` resolves to the map's scope storage, a register or shared memory."""
    sdfg = sdfg_with_transient(storage, dtypes.AllocationLifetime.State, in_map=True)

    make_transients_persistent(sdfg, dace.DeviceType.CPU, toplevel_only=False)

    assert (sdfg.arrays['buffer'].lifetime == PERSISTENT) is persistent


def test_each_distinct_extent_is_compared_with_one_once():
    """Transients sharing extents share one ``symbolic.equal(extent, 1)``; the verdicts are unchanged."""
    from unittest import mock
    from dace import symbolic

    sdfg = dace.SDFG('shared_extents')
    sdfg.add_symbol('N', dace.int64)
    shapes = {'a': ['N', 5], 'b': [5, 'N'], 'c': [1, 1], 'd': ['N', 1], 'e': [1]}
    state = sdfg.add_state()
    for name, shape in shapes.items():
        sdfg.add_array(name, shape, dace.float64, transient=True)
        state.add_access(name)
    with mock.patch.object(symbolic, 'equal', side_effect=symbolic.equal) as spy:
        make_transients_persistent(sdfg, dace.DeviceType.CPU)
    assert spy.call_count == 3, spy.call_args_list  # N, 5 and 1
    assert {name for name, desc in sdfg.arrays.items() if desc.lifetime == PERSISTENT} == {'a', 'b', 'd'}
