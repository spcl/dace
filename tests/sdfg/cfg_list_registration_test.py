# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every control-flow region must be registered in its SDFG's ``cfg_list`` as soon as it is added."""

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion


def assert_all_registered(sdfg: dace.SDFG) -> None:
    """Every reachable region is in ``cfg_list`` and owns a distinct ``cfg_id``.

    ``cfg_id`` is a position in ``cfg_list`` and each region starts out holding a list containing
    only itself, so an unregistered region does not raise -- it reports 0, the same id as the root
    and as every other unregistered region. Callers that key results by ``cfg_id`` (for instance
    ``ControlFlowRegionPass.apply_pass``) then overwrite each other's entries silently.
    """
    reachable = list(sdfg.all_control_flow_regions(recursive=True))
    registered = {id(cfg) for cfg in sdfg.cfg_list}
    unregistered = [cfg.label for cfg in reachable if id(cfg) not in registered]
    assert not unregistered, f'regions missing from cfg_list: {unregistered}'

    ids = [cfg.cfg_id for cfg in reachable]
    assert len(set(ids)) == len(ids), f'cfg_id collision: {ids}'

    detached = [b.label for b in sdfg.all_control_flow_blocks() if b.sdfg is None]
    assert not detached, f'blocks left without an SDFG: {detached}'


def test_regions_are_registered_when_added():
    """The four construction paths that build a fresh, never-serialized SDFG."""
    sdfg = dace.SDFG('fresh')
    sdfg.add_array('a', [10], dace.float64)

    loop = LoopRegion('myloop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    loop.add_state('body', is_start_block=True)
    assert_all_registered(sdfg)

    inner = ControlFlowRegion('inner')
    loop.add_node(inner)
    inner.add_state('istate', is_start_block=True)
    assert_all_registered(sdfg)

    cond = ConditionalBlock('cond')
    sdfg.add_node(cond)
    branch = ControlFlowRegion('br')
    branch.add_state('bstate', is_start_block=True)
    cond.add_branch(CodeBlock('i < 5'), branch)
    assert_all_registered(sdfg)

    # A region carrying a subtree registers that subtree too, not just its own root.
    subtree = ControlFlowRegion('subtree')
    nested = LoopRegion('nested', 'k < 3', 'k', 'k = 0', 'k = k + 1')
    subtree.add_node(nested, is_start_block=True)
    nested.add_state('nstate', is_start_block=True)
    sdfg.add_node(subtree)
    assert_all_registered(sdfg)

    # Registering eagerly must agree with a full recompute, or ids would shift underneath
    # anything that cached one.
    before = [cfg.label for cfg in sdfg.cfg_list]
    sdfg.reset_cfg_list()
    assert [cfg.label for cfg in sdfg.cfg_list] == before


def test_registration_survives_serialization_round_trip():
    sdfg = dace.SDFG('roundtrip')
    sdfg.add_array('a', [10], dace.float64)
    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    loop.add_state('body', is_start_block=True)
    cond = ConditionalBlock('cond')
    loop.add_node(cond)
    branch = ControlFlowRegion('br')
    branch.add_state('bstate', is_start_block=True)
    cond.add_branch(CodeBlock('i < 5'), branch)

    assert_all_registered(sdfg)
    assert_all_registered(dace.SDFG.from_json(sdfg.to_json()))


if __name__ == '__main__':
    test_regions_are_registered_when_added()
    test_registration_survives_serialization_round_trip()
