# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``CfgIds`` resolves a region's ``cfg_id`` from one index of the CFG list."""
import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate import ConditionFusion
from dace.transformation.passes.pattern_matching import CfgIds, match_patterns


def guarded_loops(count: int) -> dace.SDFG:
    sdfg = dace.SDFG('guarded_loops')
    sdfg.add_symbol('n', dace.int64)
    last = sdfg.add_state('entry', is_start_block=True)
    for k in range(count):
        guard = ConditionalBlock(f'guard{k}')
        body = ControlFlowRegion(f'guard{k}_body')
        loop = LoopRegion(f'loop{k}', f'i{k} < n', f'i{k}', f'i{k} = 0', f'i{k} = i{k} + 1')
        loop.add_state(f'body{k}', is_start_block=True)
        body.add_node(loop, is_start_block=True)
        guard.add_branch(dace.properties.CodeBlock(f'n > {k}'), body)
        sdfg.add_node(guard)
        sdfg.add_edge(last, guard, dace.InterstateEdge())
        last = guard
    return sdfg


def test_every_region_resolves_to_its_cfg_id():
    sdfg = guarded_loops(4)
    ids = CfgIds(sdfg)
    for region in sdfg.all_control_flow_regions(recursive=True):
        assert ids.cfg_id(region) == region.cfg_id, region


def test_a_region_holding_another_list_resolves_through_its_own():
    """A region grafted without a reset carries its own list; its id is what that list says."""
    sdfg = guarded_loops(2)
    ids = CfgIds(sdfg)
    detached = LoopRegion('detached', 'j < 2', 'j', 'j = 0', 'j = j + 1')
    assert ids.cfg_id(detached) == detached.cfg_id == 0


def test_a_matcher_sweep_resolves_no_cfg_id_by_scanning(monkeypatch):
    """``cfg_id`` is ``cfg_list.index``; asked per candidate it made each sweep quadratic in the
    region count (warpx_field_gather: 13000 regions)."""
    sdfg = guarded_loops(6)
    scans = []
    original = ControlFlowRegion.cfg_id.fget

    def counted(self):
        scans.append(self)
        return original(self)

    monkeypatch.setattr(ControlFlowRegion, 'cfg_id', property(counted))
    monkeypatch.setattr(ConditionalBlock, 'cfg_id', property(counted), raising=False)
    matches = list(match_patterns(sdfg, [ConditionFusion]))
    assert matches
    assert scans == [], len(scans)
