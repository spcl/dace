# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Checks that the CFG list and the parent pointers a pass kept in place are what a fresh reset builds."""
import copy

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion


def assert_tree_consistent(sdfg: dace.SDFG) -> None:
    """``cfg_list`` is the pre-order a reset would build, shared by every region, and every block and nested
    SDFG points at where it sits."""
    fresh = list(sdfg.all_control_flow_regions(recursive=True))
    listed = sdfg.cfg_list
    assert len(listed) == len(fresh) and all(a is b for a, b in zip(listed, fresh)), ([r.label for r in listed],
                                                                                      [r.label for r in fresh])
    for region in fresh:
        assert region._cfg_list is listed, region.label
        owner = region if isinstance(region, dace.SDFG) else region.sdfg
        for block in region.nodes():
            assert block.parent_graph is region, (block.label, region.label)
            assert block.sdfg is owner, (block.label, region.label)
            if isinstance(block, dace.SDFGState):
                for node in block.nodes():
                    if isinstance(node, nodes.NestedSDFG):
                        assert node.sdfg.parent is block and node.sdfg.parent_nsdfg_node is node, node.label
                        assert node.sdfg.parent_sdfg is owner, node.label


def assert_tree_matches_a_reset(sdfg: dace.SDFG) -> None:
    """The kept tree is consistent, and a fresh ``reset_cfg_list`` on a copy lists the same regions at the same
    ``cfg_id`` s."""
    assert_tree_consistent(sdfg)
    fresh = copy.deepcopy(sdfg)
    fresh.reset_cfg_list()
    kept = [(type(r).__name__, r.label, r.cfg_id) for r in sdfg.cfg_list]
    assert kept == [(type(r).__name__, r.label, r.cfg_id) for r in fresh.cfg_list]


def spy_on_resets(monkeypatch) -> list:
    """Every ``reset_cfg_list`` call from here on, recorded by the region it was called on; the reset still runs."""
    resets = []
    original = dace.sdfg.state.AbstractControlFlowRegion.reset_cfg_list
    monkeypatch.setattr(dace.sdfg.state.AbstractControlFlowRegion, 'reset_cfg_list',
                        lambda self: resets.append(self) or original(self))
    return resets


def loop(label: str) -> LoopRegion:
    region = LoopRegion(label, f'{label}_i < 4', f'{label}_i', f'{label}_i = 0', f'{label}_i = {label}_i + 1')
    region.add_state(f'{label}_body', is_start_block=True)
    return region


def conditional(label: str) -> ConditionalBlock:
    block = ConditionalBlock(label)
    for k, condition in enumerate(('n > 0', None)):
        branch = ControlFlowRegion(f'{label}_b{k}')
        branch.add_state(f'{label}_b{k}_s', is_start_block=True)
        block.add_branch(CodeBlock(condition) if condition else None, branch)
    return block


def inner_sdfg(label: str, depth: int) -> dace.SDFG:
    sdfg = dace.SDFG(label)
    sdfg.add_symbol('n', dace.int64)
    first = sdfg.add_state(f'{label}_first', is_start_block=True)
    body = loop(f'{label}_loop')
    sdfg.add_node(body)
    sdfg.add_edge(first, body, dace.InterstateEdge())
    guard = conditional(f'{label}_if')
    sdfg.add_node(guard)
    sdfg.add_edge(body, guard, dace.InterstateEdge())
    if depth > 0:
        host = body.nodes()[0]
        host.add_nested_sdfg(inner_sdfg(f'{label}_n', depth - 1), {}, {}, symbol_mapping={'n': 'n'})
    return sdfg
