# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the context collected before an AI-generated expansion.

The nesting walk is the part most likely to break, so it is exercised against a deliberately deep
hierarchy: a GPU kernel containing a nested SDFG containing a loop containing another nested SDFG
containing a sequential map, with a different symbol name at every level.
"""

import dace
from dace import dtypes
from dace.libraries.ai.context import collect_context, collect_nesting
from dace.libraries.ai.nodes import AINode
from dace.libraries.ai.prompts import build_user_prompt
from dace.sdfg.scope import is_devicelevel_gpu
from dace.sdfg.state import LoopRegion

DESCRIPTION = 'Copy one element, doubling it.'


def _build_level2() -> dace.SDFG:
    """
    Builds the innermost SDFG: a sequential map around an :class:`AINode`.

    :return: The SDFG, whose symbols are ``m``, ``k`` and ``off``.
    """
    sdfg = dace.SDFG('L2')
    sdfg.add_symbol('m', dace.int64)
    sdfg.add_symbol('k', dace.int64)
    sdfg.add_symbol('off', dace.int64)
    k = dace.symbol('k', dace.int64)
    sdfg.add_array('a2', [k], dace.float64)
    sdfg.add_array('b2', [k], dace.float64)

    state = sdfg.add_state('l2_body')
    entry, exit_node = state.add_map('inner', {'j': '0:k'}, schedule=dtypes.ScheduleType.Sequential)
    node = AINode('kernel', DESCRIPTION, inputs={'_in'}, outputs={'_out'})
    state.add_node(node)
    state.add_memlet_path(state.add_read('a2'), entry, node, dst_conn='_in', memlet=dace.Memlet('a2[j]'))
    state.add_memlet_path(node, exit_node, state.add_write('b2'), src_conn='_out', memlet=dace.Memlet('b2[j]'))
    return sdfg


def _build_level1() -> dace.SDFG:
    """
    Builds the middle SDFG: a loop region around the level-2 nested SDFG.

    :return: The SDFG, whose symbols are ``n``, ``tile`` and ``base``.
    """
    sdfg = dace.SDFG('L1')
    sdfg.add_symbol('n', dace.int64)
    sdfg.add_symbol('tile', dace.int64)
    sdfg.add_symbol('base', dace.int64)
    n = dace.symbol('n', dace.int64)
    sdfg.add_array('a1', [n], dace.float64)
    sdfg.add_array('b1', [n], dace.float64)

    loop = LoopRegion('sweep',
                      condition_expr='s < n',
                      loop_var='s',
                      initialize_expr='s = 0',
                      update_expr='s = s + tile')
    sdfg.add_node(loop, is_start_block=True)
    sdfg.add_symbol('s', dace.int64)

    body = loop.add_state('sweep_body', is_start_block=True)
    nested = body.add_nested_sdfg(_build_level2(), {'a2'}, {'b2'}, symbol_mapping={'m': 'n', 'k': 'tile', 'off': 's'})
    body.add_edge(body.add_read('a1'), None, nested, 'a2', dace.Memlet('a1[s:s+tile]'))
    body.add_edge(nested, 'b2', body.add_write('b1'), None, dace.Memlet('b1[s:s+tile]'))
    return sdfg


def _build_hierarchy() -> dace.SDFG:
    """
    Builds the full hierarchy: a GPU kernel around the level-1 nested SDFG.

    :return: The top-level SDFG, whose symbols are ``N`` and ``M``.
    """
    sdfg = dace.SDFG('hier')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('M', dace.int64)
    N = dace.symbol('N', dace.int64)
    sdfg.add_array('A', [N], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [N], dace.float64, storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('top')
    entry, exit_node = state.add_map('grid', {'bi': '0:N:32'}, schedule=dtypes.ScheduleType.GPU_Device)
    nested = state.add_nested_sdfg(_build_level1(), {'a1'}, {'b1'},
                                   symbol_mapping={
                                       'n': 'N',
                                       'tile': '32',
                                       'base': 'bi'
                                   })
    state.add_memlet_path(state.add_read('A'), entry, nested, dst_conn='a1', memlet=dace.Memlet('A[bi:bi+32]'))
    state.add_memlet_path(nested, exit_node, state.add_write('B'), src_conn='b1', memlet=dace.Memlet('B[bi:bi+32]'))
    return sdfg


def _find_ai_node(sdfg: dace.SDFG):
    """
    Locates the :class:`AINode` and the state containing it.

    :param sdfg: The SDFG to search, recursively.
    :return: A tuple of (node, state, containing SDFG).
    """
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, AINode):
                return node, state, state.sdfg
            if isinstance(node, dace.nodes.NestedSDFG):
                found = _find_ai_node(node.sdfg)
                if found is not None:
                    return found
    return None


def test_nesting_walk_crosses_maps_loops_and_nested_sdfgs():
    node, state, sdfg = _find_ai_node(_build_hierarchy())
    frames = collect_nesting(node, state, sdfg)

    assert [f.kind for f in frames] == ['map', 'nested_sdfg', 'loop', 'nested_sdfg', 'map', 'sdfg']
    assert [f.label for f in frames] == ['inner', 'L2', 'sweep', 'L1', 'grid', 'hier']

    # Schedules survive across both nested SDFG boundaries
    assert frames[0].schedule == 'Sequential'
    assert frames[4].schedule == 'GPU_Device'
    assert frames[0].detail.startswith('j = 0:k')
    assert '0:N:32' in frames[4].detail

    # The loop's variable, condition and update are all recorded
    assert 's' in frames[2].detail
    assert 's < n' in frames[2].detail
    assert 's + tile' in frames[2].detail

    # The walk terminates at the top-level SDFG
    assert frames[-1].kind == 'sdfg'
    assert 'N' in frames[-1].detail


def test_symbol_remapping_chain_is_recorded():
    node, state, sdfg = _find_ai_node(_build_hierarchy())
    frames = collect_nesting(node, state, sdfg)

    inner_mapping = frames[1].symbol_mapping
    outer_mapping = frames[3].symbol_mapping

    # k -> tile -> 32, and m -> n -> N
    assert inner_mapping['k'] == 'tile'
    assert outer_mapping['tile'] == '32'
    assert inner_mapping['m'] == 'n'
    assert outer_mapping['n'] == 'N'
    # The loop variable is threaded through as well
    assert inner_mapping['off'] == 's'


def test_device_level_is_detected_through_nested_sdfgs():
    node, state, sdfg = _find_ai_node(_build_hierarchy())

    assert is_devicelevel_gpu(sdfg, state, node)
    ctx = collect_context(node, state, sdfg)
    assert ctx.capabilities.device_level
    assert not ctx.capabilities.state_available
    assert ctx.capabilities.environments_allowed == 'device-headers-only'


def test_connectors_and_descriptors_are_collected():
    node, state, sdfg = _find_ai_node(_build_hierarchy())
    ctx = collect_context(node, state, sdfg)

    by_name = {c.name: c for c in ctx.connectors}
    assert set(by_name) == {'_in', '_out'}

    # A single element of an array reaches the node as a scalar value, not a pointer
    assert not by_name['_in'].is_pointer
    assert by_name['_in'].element_type == 'double'
    assert by_name['_in'].container_kind == 'Array'
    assert by_name['_in'].data == 'a2'
    assert by_name['_in'].shape == ('k', )
    assert by_name['_in'].storage == 'Default'
    assert by_name['_in'].num_elements == '1'
    assert by_name['_out'].direction == 'out'


def test_prompt_is_stable_across_runs():
    # Connectors are usually declared as a set, so without an explicit order the prompt would
    # differ between runs of the same program for no reason
    prompts = []
    for _ in range(3):
        node, state, sdfg = _find_ai_node(_build_hierarchy())
        ctx = collect_context(node, state, sdfg)
        assert [c.name for c in ctx.connectors] == ['_in', '_out']
        prompts.append(build_user_prompt(ctx))

    assert len(set(prompts)) == 1, 'the same SDFG produced different prompts on different runs'


def test_prompt_contains_the_whole_hierarchy():
    node, state, sdfg = _find_ai_node(_build_hierarchy())
    prompt = build_user_prompt(collect_context(node, state, sdfg))

    # Outermost first, each level present and in order
    positions = [prompt.index(f'"{label}"') for label in ('hier', 'grid', 'L1', 'sweep', 'L2', 'inner')]
    assert positions == sorted(positions), 'the nesting outline is not rendered outermost first'

    assert DESCRIPTION in prompt
    assert 'GPU_Device' in prompt
    assert '__state available in the body: no' in prompt
    assert '_in' in prompt and '_out' in prompt


if __name__ == '__main__':
    test_nesting_walk_crosses_maps_loops_and_nested_sdfgs()
    test_symbol_remapping_chain_is_recorded()
    test_device_level_is_detected_through_nested_sdfgs()
    test_connectors_and_descriptors_are_collected()
    test_prompt_contains_the_whole_hierarchy()
