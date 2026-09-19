# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Every control-flow region must be registered in its SDFG's ``cfg_list`` as soon as it is added."""

import copy

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


def nested_with_conditional(name: str) -> dace.SDFG:
    """A nested SDFG whose body is a region subtree, so registering only its root is not enough."""
    inner = dace.SDFG(name)
    inner.add_array('x', [10], dace.float64)
    cond = ConditionalBlock(f'{name}_cond')
    inner.add_node(cond, is_start_block=True)
    branch = ControlFlowRegion(f'{name}_branch')
    branch.add_state(f'{name}_bstate', is_start_block=True)
    cond.add_branch(CodeBlock('1 < 5'), branch)
    return inner


def test_copied_nested_sdfg_is_registered_when_added():
    """A deep-copied nested SDFG carries an empty ``cfg_list``; attaching it must register its subtree.

    Map-fusion clones of a producer body attach a copied ``NestedSDFG`` node with ``add_node``, and
    statement splitting hands a copied SDFG to ``add_nested_sdfg``. Either way the copy used to stay
    out of the list and the first ``cfg_id`` asked of it raised ``list.index(x): x not in list``.
    """
    sdfg = dace.SDFG('host')
    sdfg.add_array('x', [10], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    node = state.add_nested_sdfg(nested_with_conditional('inner'), inputs={}, outputs={'x'})
    assert_all_registered(sdfg)

    state.add_node(copy.deepcopy(node))
    assert_all_registered(sdfg)

    state.add_nested_sdfg(copy.deepcopy(node.sdfg), inputs={}, outputs={'x'})
    assert_all_registered(sdfg)

    before = [id(cfg) for cfg in sdfg.cfg_list]
    sdfg.reset_cfg_list()
    assert [id(cfg) for cfg in sdfg.cfg_list] == before


if __name__ == '__main__':
    test_regions_are_registered_when_added()
    test_registration_survives_serialization_round_trip()
    test_copied_nested_sdfg_is_registered_when_added()
