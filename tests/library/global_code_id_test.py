# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Names of the free functions library-node expansions append to the global code.

Every wrapper of one program lands in the same translation unit, so two expansions that pick the
same name are a C++ redefinition error. ``ls3df_scf`` on ``dace_gpu_canonicalize`` failed to build
exactly that way: two ``ScatterConflictCheck`` nodes sat at the same block/node ids of two sibling
control-flow regions and both emitted ``__dace_scatter_conflict_canon_gpu_19_13``.
"""
import re

import dace
from dace.codegen.common import global_code_id
from dace.libraries.sort.nodes.scatter_conflict_check import ScatterConflictCheck
from dace.sdfg.state import ControlFlowRegion

N = dace.symbol('N')


def add_check(state: dace.SDFGState, index: str, flag: str, owner: str) -> ScatterConflictCheck:
    """One device-lowered ``ScatterConflictCheck`` of ``index`` into ``flag``, sized by ``owner``."""
    node = ScatterConflictCheck('check')
    node.implementation = 'CUDA'
    node.schedule = dace.ScheduleType.Sequential
    state.add_node(node)
    node.add_out_connector('_owner_out')
    state.add_edge(state.add_read(index), None, node, '_idx_in', dace.Memlet(f'{index}[0:N]'))
    state.add_edge(node, '_count_out', state.add_write(flag), None, dace.Memlet(f'{flag}[0]'))
    state.add_edge(node, '_owner_out', state.add_write(owner), None, dace.Memlet(f'{owner}[0:N]'))
    return node


def make_sibling_region_checks() -> dace.SDFG:
    """Two regions, each one state holding one check at the SAME local block and node ids."""
    sdfg = dace.SDFG('sibling_checks')
    sdfg.add_array('ip', [N], dace.int64, storage=dace.StorageType.GPU_Global)
    for i in range(2):
        sdfg.add_array(f'flag{i}', [1], dace.int64)
        sdfg.add_transient(f'owner{i}', [N], dace.int64, storage=dace.StorageType.CPU_Heap)
    previous = sdfg.add_state('entry', is_start_block=True)
    for i in range(2):
        region = ControlFlowRegion(f'region{i}', sdfg=sdfg)
        sdfg.add_node(region)
        sdfg.add_edge(previous, region, dace.InterstateEdge())
        add_check(region.add_state('check', is_start_block=True), 'ip', f'flag{i}', f'owner{i}')
        previous = region
    sdfg.validate()
    return sdfg


def wrapper_definitions(sdfg: dace.SDFG) -> list:
    """Names of every ``__dace_scatter_conflict_*`` function DEFINED in the device global code."""
    code = sdfg.global_code['cuda'].as_string
    return re.findall(r'^gpuError_t (__dace_scatter_conflict_\w+)\(', code, re.M)


def test_checks_in_sibling_regions_get_distinct_wrappers():
    """Same block and node ids in two regions: two wrappers, two names."""
    sdfg = make_sibling_region_checks()
    sdfg.expand_library_nodes()
    names = wrapper_definitions(sdfg)
    assert len(names) == 2, names
    assert len(set(names)) == 2, f'both expansions defined {names[0]}'


def test_two_checks_in_one_state_get_distinct_wrappers():
    """Two checks side by side in one state keep the two node ids they were expanded at."""
    sdfg = dace.SDFG('one_state_checks')
    sdfg.add_array('ip', [N], dace.int64, storage=dace.StorageType.GPU_Global)
    state = sdfg.add_state('check', is_start_block=True)
    for i in range(2):
        sdfg.add_array(f'flag{i}', [1], dace.int64)
        sdfg.add_transient(f'owner{i}', [N], dace.int64, storage=dace.StorageType.CPU_Heap)
        add_check(state, 'ip', f'flag{i}', f'owner{i}')
    sdfg.expand_library_nodes()
    names = wrapper_definitions(sdfg)
    assert len(names) == 2 and len(set(names)) == 2, names


def test_top_level_state_keeps_the_plain_name():
    """A node in a top-level state is still named ``<sdfg>_<state>_<node>``."""
    sdfg = dace.SDFG('top_level_check')
    sdfg.add_array('ip', [N], dace.int64, storage=dace.StorageType.GPU_Global)
    sdfg.add_array('flag', [1], dace.int64)
    sdfg.add_transient('owner', [N], dace.int64, storage=dace.StorageType.CPU_Heap)
    sdfg.add_state('entry', is_start_block=True)
    state = sdfg.add_state_after(sdfg.start_block, 'check')
    node = add_check(state, 'ip', 'flag', 'owner')
    expected = f'top_level_check_{sdfg.node_id(state)}_{state.node_id(node)}'
    assert global_code_id(sdfg, state, node) == expected


def test_nested_region_id_spells_the_region_path():
    """A state inside a region is named by the region's block id, then its own."""
    sdfg = make_sibling_region_checks()
    region = [b for b in sdfg.nodes() if isinstance(b, ControlFlowRegion)][1]
    state = region.start_block
    node = next(n for n in state.nodes() if isinstance(n, ScatterConflictCheck))
    expected = f'sibling_checks_{sdfg.node_id(region)}r{region.node_id(state)}_{state.node_id(node)}'
    assert global_code_id(sdfg, state, node) == expected


if __name__ == '__main__':
    test_checks_in_sibling_regions_get_distinct_wrappers()
    test_two_checks_in_one_state_get_distinct_wrappers()
    test_top_level_state_keeps_the_plain_name()
    test_nested_region_id_spells_the_region_path()
