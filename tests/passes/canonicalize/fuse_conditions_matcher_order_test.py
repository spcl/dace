# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``FuseConditions(matcher_order=True)`` fuses what the plain matcher fuses, in its order, without re-walking."""
import copy

import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.interstate import ConditionFusion
from dace.transformation.passes.canonicalize.fuse_conditions import FuseConditions
from dace.transformation.passes.pattern_matching import PatternApplyOnceEverywhere


@dace.program
def guards(a: dace.float64[6], n: dace.int64):
    s = 0.0
    if a[0] > 1:
        if a[1] > 1:
            if n > 2:
                s += 1.0
    if a[2] > 1:
        s += 2.0
    if a[3] > 1:
        s += 3.0
    for i in range(4):
        if n > i:
            if a[4] > 1:
                s += a[i]
    a[5] = s


def layout(sdfg: dace.SDFG) -> list:
    rows = []
    for region in sdfg.all_control_flow_regions(recursive=True):
        rows.append((type(region).__name__, region.label))
        if isinstance(region, ConditionalBlock):
            rows += [(c.as_string if c is not None else None, b.label) for c, b in region.branches]
        rows += [(e.src.label, e.dst.label, e.data.condition.as_string, str(e.data.assignments))
                 for e in region.edges()]
    return rows


def test_matcher_order_fusion_is_the_plain_matchers_fusion():
    """The canonicalization recipe was built on the matcher's order; any other order fuses differently
    (ConditionFusion is not confluent) and shifts everything downstream of the ``fuse`` stage."""
    sdfg = guards.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    expected = PatternApplyOnceEverywhere([ConditionFusion()], validate=False).apply_pass(reference, {})
    fused = FuseConditions(matcher_order=True).apply_pass(sdfg, {})
    sdfg.validate()
    assert fused == len(expected['ConditionFusion']), (fused, expected)
    assert layout(sdfg) == layout(reference)


def guard_chain(region: ControlFlowRegion, count: int, fusable: bool, tag: str) -> None:
    """``count`` guards in a row in ``region``: all on one condition (they fuse into one), or each
    reading the symbol assigned on the edge into it (none fuses)."""
    last = region.add_state(f'{tag}_entry', is_start_block=True)
    for k in range(count):
        guard = ConditionalBlock(f'{tag}{k}')
        body = ControlFlowRegion(f'{tag}{k}_body')
        body.add_state(f'{tag}{k}_state', is_start_block=True)
        guard.add_branch(CodeBlock('n > 0' if fusable else f'{tag}{k} > 0'), body)
        region.add_node(guard)
        region.add_edge(last, guard, dace.InterstateEdge(assignments=None if fusable else {f'{tag}{k}': 'n'}))
        last = guard


def refused_then_fusable(count: int) -> dace.SDFG:
    """A branch holding ``count`` guards that refuse to fuse, then a branch holding ``count`` that do."""
    sdfg = dace.SDFG('refused_then_fusable')
    sdfg.add_symbol('n', dace.int64)
    entry = sdfg.add_state('entry', is_start_block=True)
    last = entry
    for tag, fusable in (('r', False), ('f', True)):
        outer = ConditionalBlock(f'outer_{tag}')
        branch = ControlFlowRegion(f'outer_{tag}_body')
        guard_chain(branch, count, fusable, tag)
        outer.add_branch(CodeBlock(f'outer_{tag} > 0'), branch)
        sdfg.add_node(outer)
        sdfg.add_edge(last, outer, dace.InterstateEdge(assignments={f'outer_{tag}': 'n'}))
        last = outer
    return sdfg


def probes_of(fusion, sdfg: dace.SDFG, monkeypatch) -> int:
    probes = [0]
    original = ConditionFusion.can_be_applied

    def counted(self, graph, expr_index, sd, permissive=False):
        probes[0] += 1
        return original(self, graph, expr_index, sd, permissive)

    monkeypatch.setattr(ConditionFusion, 'can_be_applied', counted)
    fusion.apply_pass(sdfg, {})
    monkeypatch.undo()
    return probes[0]


def test_refused_candidates_are_not_probed_again_per_fusion(monkeypatch):
    """The plain matcher re-walks every region after each fusion and probes every refused candidate
    again: 1.5M probes for 978 fusions on warpx_field_gather."""
    count = 12
    sdfg, reference = refused_then_fusable(count), refused_then_fusable(count)
    plain = probes_of(PatternApplyOnceEverywhere([ConditionFusion()], validate=False, progress=False), reference,
                      monkeypatch)
    local = probes_of(FuseConditions(matcher_order=True), sdfg, monkeypatch)
    assert layout(sdfg) == layout(reference)
    assert plain >= 2 * count * count, plain
    assert local <= 5 * count, local


def test_an_sdfg_without_conditionals_is_left_alone():
    sdfg = dace.SDFG('no_guards')
    sdfg.add_state('only')
    assert FuseConditions(matcher_order=True).apply_pass(sdfg, {}) is None
