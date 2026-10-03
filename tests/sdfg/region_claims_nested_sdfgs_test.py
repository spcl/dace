# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Claiming a control-flow region claims the nested SDFGs its blocks hold.

A nested SDFG keeps three back-references to where it sits -- ``parent`` (the state), ``parent_sdfg``
(the SDFG that state belongs to) and ``parent_nsdfg_node`` (the node wrapping it).
``SDFGState.add_node`` maintains all three when it accepts a nested-SDFG node. Every other
operation that CLAIMS a block owes the same -- ``AbstractControlFlowRegion.add_node`` for a region
or a bare state, and ``ConditionalBlock.add_branch`` for a branch, which is reached only through
the branch list and so never passes through ``add_node`` at all. They share ``rehome_claimed_block``
so a block cannot be half-claimed depending on which door it came through.

Two ordinary constructions leave them unset otherwise, and neither involves anything exotic:

* assembling a region while it is still detached (its ``sdfg`` is ``None``, so every nested SDFG
  added to one of its states records ``None``), then handing the finished region to an SDFG;
* ``copy.deepcopy`` of a region that IS owned -- ``SDFG.__deepcopy__`` resolves the parent through
  the memo, the owner is not in it because only the region was copied, and the self-repair branch
  does not fire because the original was never detached. Loop fission, loop specialization,
  guarded-loop partitioning and the segment-chain clones in parallelization prep all do this.

Both end at "Parent SDFG not properly set for nested SDFG node" in validation. The claim stops at
nested-SDFG boundaries on purpose: a nested SDFG one level deeper was copied together with the
state that owns it, so its references are already consistent and belong to that SDFG, not to this
one.
"""
import copy

import dace
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.interstate.condition_fusion import ConditionFusion
from dace.transformation.passes.loop_fission import LoopFission


def writer_sdfg(name: str) -> dace.SDFG:
    """A one-state SDFG writing a single element, for use as a nested SDFG."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('x', [10], dace.float64)
    state = sdfg.add_state('compute')
    tasklet = state.add_tasklet('t', {}, {'o'}, 'o = 1.0')
    state.add_edge(tasklet, 'o', state.add_access('x'), None, dace.Memlet('x[0]'))
    return sdfg


def wrapping_sdfg(name: str, inner: dace.SDFG) -> dace.SDFG:
    """An SDFG whose single state holds ``inner`` as a nested SDFG."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('x', [10], dace.float64)
    state = sdfg.add_state('s')
    node = state.add_nested_sdfg(inner, {}, {'x'})
    state.add_edge(node, 'x', state.add_access('x'), None, dace.Memlet('x[0:10]'))
    return sdfg


def detached_loop_holding(inner: dace.SDFG, array: str = 'a', label: str = 'loop') -> LoopRegion:
    """A LoopRegion, not yet owned by any SDFG, whose body state holds ``inner``."""
    loop = LoopRegion(label, 'i < 10', 'i', 'i = 0', 'i = i + 1')
    state = loop.add_state(f'{label}_body', is_start_block=True)
    node = state.add_nested_sdfg(inner, {}, {'x'})
    state.add_edge(node, 'x', state.add_access(array), None, dace.Memlet(f'{array}[0:10]'))
    return loop


def host_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array('a', [10], dace.float64)
    return sdfg


def nested_nodes(sdfg: dace.SDFG):
    """Every nested-SDFG node in ``sdfg``'s own namespace."""
    return [n for state in sdfg.all_states() for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]


def assert_homed(sdfg: dace.SDFG, expected_count: int) -> None:
    """Every nested SDFG directly under ``sdfg`` names ``sdfg``, its own state and its own node."""
    found = nested_nodes(sdfg)
    assert len(found) == expected_count, [n.label for n in found]
    states = list(sdfg.all_states())
    for node in found:
        assert node.sdfg.parent_sdfg is sdfg, node.label
        assert node.sdfg.parent_nsdfg_node is node, node.label
        assert node.sdfg.parent in states, node.label
    sdfg.validate()


def test_a_region_assembled_detached_is_claimed_when_it_is_added():
    """No copy involved: a region built before it has an owner still records that owner."""
    sdfg = host_sdfg('detached_build')
    loop = detached_loop_holding(writer_sdfg('leaf'))
    # Built while the loop had no SDFG, so the state had none either and this is the state of it.
    assert loop.nodes()[0].nodes()[0].sdfg.parent_sdfg is None
    sdfg.add_node(loop, is_start_block=True)
    assert_homed(sdfg, 1)


def test_a_deepcopied_region_is_claimed_when_it_is_added():
    """The clone shape every loop-splitting pass uses: deepcopy an owned region, add the copy."""
    sdfg = host_sdfg('clone')
    loop = detached_loop_holding(writer_sdfg('leaf'))
    sdfg.add_node(loop, is_start_block=True)
    sdfg.validate()

    clone = copy.deepcopy(loop)
    assert clone.nodes()[0].nodes()[0].sdfg.parent_sdfg is None, 'deepcopy is supposed to lose it'
    sdfg.add_node(clone, ensure_unique_name=True)
    sdfg.add_edge(loop, clone, dace.InterstateEdge())
    assert_homed(sdfg, 2)


def test_a_region_assembled_behind_a_conditional_is_claimed_through_its_branches():
    """The assembly order loop specialization uses.

    Every intermediate here is detached -- the branch region, then the conditional itself -- so
    nothing is homed until the finished conditional reaches the SDFG. A ``ConditionalBlock`` holds
    its branches in a list rather than in the graph, and ``nodes()`` returns them, so the claim has
    to traverse into them.
    """
    sdfg = host_sdfg('specialize')
    original = detached_loop_holding(writer_sdfg('leaf'))
    sdfg.add_node(original, is_start_block=True)
    sdfg.validate()

    conditional = ConditionalBlock('specialize')
    for label, condition in (('par', 'N > 1'), ('seq', None)):
        region = ControlFlowRegion(f'branch_{label}')
        region.add_node(copy.deepcopy(original), is_start_block=True, ensure_unique_name=True)
        conditional.add_branch(condition, region)
    sdfg.add_node(conditional, ensure_unique_name=True)
    sdfg.remove_node(original)
    sdfg.start_block = sdfg.node_id(conditional)

    assert_homed(sdfg, 2)


def test_a_conditional_claims_a_branch_it_is_handed():
    """``add_branch`` is the other way in, and owes the same claim as ``add_node``.

    A branch is reached only through ``ConditionalBlock._branches``, never through the graph, so
    none of the generic bookkeeping runs for it. Propagating just ``sdfg`` into the branch's blocks
    left their nested SDFGs pointing at wherever they were copied from.
    """
    sdfg = host_sdfg('branch_claim')
    conditional = ConditionalBlock('cond')
    sdfg.add_node(conditional, is_start_block=True)

    branch = ControlFlowRegion('branch')
    branch.add_node(detached_loop_holding(writer_sdfg('leaf')), is_start_block=True)
    conditional.add_branch(CodeBlock('True'), copy.deepcopy(branch))

    assert_homed(sdfg, 1)


def test_a_bare_state_handed_to_a_region_is_claimed():
    """A cloned ``SDFGState`` is claimed like a cloned region: it holds nested SDFGs the same way.

    ``SDFGState`` is not an ``AbstractControlFlowRegion``, so gating the claim on that type left
    exactly this shape -- ``region.add_node(copy.deepcopy(state))``, which ``MoveIfIntoLoop`` and
    ``BranchElimination`` both do -- with the nested SDFGs still naming the donor.
    """
    donor = host_sdfg('donor')
    state = donor.add_state('s')
    node = state.add_nested_sdfg(writer_sdfg('leaf'), {}, {'x'})
    state.add_edge(node, 'x', state.add_access('a'), None, dace.Memlet('a[0:10]'))
    donor.validate()

    sdfg = host_sdfg('state_claim')
    sdfg.add_node(copy.deepcopy(state), is_start_block=True)

    assert_homed(sdfg, 1)


def test_condition_fusion_produces_a_valid_sdfg():
    """The regression the ``add_branch`` claim closes, in the pass that hit it.

    ``ConditionFusion.merge_matching_guards`` folds complementary guards by handing ``add_branch``
    a deep copy of the other block's body, and returns at that point -- before the sweep at the end
    of ``fuse_consecutive_conditions`` that used to repair parent references after the fact. So the
    branch's nested SDFGs stayed unclaimed and the fused SDFG failed validation.
    """
    sdfg = host_sdfg('condition_fusion')
    sdfg.add_symbol('c', dace.bool)

    first = ConditionalBlock('cb1')
    taken = ControlFlowRegion('cb1_body')
    taken.add_state('cb1_s', is_start_block=True).add_tasklet('nop', {}, {}, '')
    first.add_branch(CodeBlock('c'), taken)
    sdfg.add_node(first, is_start_block=True)

    second = ConditionalBlock('cb2')
    otherwise = ControlFlowRegion('cb2_body')
    state = otherwise.add_state('cb2_s', is_start_block=True)
    node = state.add_nested_sdfg(writer_sdfg('leaf'), {}, {'x'})
    state.add_edge(node, 'x', state.add_access('a'), None, dace.Memlet('a[0:10]'))
    second.add_branch(CodeBlock('not c'), otherwise)
    sdfg.add_node(second)
    sdfg.add_edge(first, second, dace.InterstateEdge())
    sdfg.validate()

    fusion = ConditionFusion()
    fusion.cblck1, fusion.cblck2, fusion.expr_index = first, second, 0
    fusion.apply(sdfg, sdfg)

    assert len(first.branches) == 2, first.branches
    assert_homed(sdfg, 1)


def test_a_claim_inside_a_nested_sdfg_homes_to_that_nested_sdfg():
    """``parent_sdfg`` names the SDFG the block belongs to, which is not always the root."""
    inner = host_sdfg('inner')
    loop = detached_loop_holding(writer_sdfg('leaf'))
    inner.add_node(loop, is_start_block=True)

    root = host_sdfg('root')
    state = root.add_state('s')
    node = state.add_nested_sdfg(inner, {}, {'a'})
    state.add_edge(node, 'a', state.add_access('a'), None, dace.Memlet('a[0:10]'))
    root.validate()

    inner.add_node(copy.deepcopy(loop), ensure_unique_name=True)
    assert_homed(inner, 2)
    root.validate()


def test_a_deeper_nested_sdfg_keeps_the_parent_it_was_copied_with():
    """The claim stops at nested-SDFG boundaries, and that is the correct place to stop.

    A nested SDFG two levels down was copied together with the state that owns it, so its
    references already point inside the copy. Re-homing it to the claiming SDFG would be wrong --
    it does not live there.
    """
    sdfg = host_sdfg('depth')
    leaf = writer_sdfg('leaf')
    middle = wrapping_sdfg('middle', leaf)
    loop = detached_loop_holding(middle)
    sdfg.add_node(loop, is_start_block=True)
    sdfg.validate()
    sdfg.add_node(copy.deepcopy(loop), ensure_unique_name=True)

    directly_nested = nested_nodes(sdfg)
    assert len(directly_nested) == 2
    for node in directly_nested:
        assert node.sdfg.parent_sdfg is sdfg
        deeper = nested_nodes(node.sdfg)
        assert len(deeper) == 1
        # The middle SDFG owns the leaf, not the host, and each copy owns its own leaf.
        assert deeper[0].sdfg.parent_sdfg is node.sdfg
        assert deeper[0].sdfg.parent_nsdfg_node is deeper[0]
    sdfg.validate()


def test_loop_fission_clones_keep_their_nested_sdfg_parents_two_levels_down():
    """Loop fission clones the loop per group and relies on the claim, with no reattach of its own.

    Two levels of nesting, so this also covers the half the claim does NOT do: the inner SDFG comes
    out of ``deepcopy`` already consistent, and the pass must not need a recursive sweep to fix it.
    """
    sdfg = host_sdfg('fission')
    sdfg.add_array('b', [10], dace.float64)
    loop = LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    for label, array in (('s0', 'a'), ('s1', 'b')):
        state = loop.add_state(label, is_start_block=(label == 's0'))
        node = state.add_nested_sdfg(wrapping_sdfg(f'middle_{label}', writer_sdfg(f'leaf_{label}')), {}, {'x'})
        state.add_edge(node, 'x', state.add_access(array), None, dace.Memlet(f'{array}[0:10]'))
    loop.add_edge(loop.nodes()[0], loop.nodes()[1], dace.InterstateEdge())
    sdfg.validate()

    LoopFission._fission_blocks(loop, [[loop.nodes()[0]], [loop.nodes()[1]]])

    assert_homed(sdfg, 2)
    for node in nested_nodes(sdfg):
        deeper = nested_nodes(node.sdfg)
        assert len(deeper) == 1
        assert deeper[0].sdfg.parent_sdfg is node.sdfg


if __name__ == '__main__':
    test_a_region_assembled_detached_is_claimed_when_it_is_added()
    test_a_deepcopied_region_is_claimed_when_it_is_added()
    test_a_region_assembled_behind_a_conditional_is_claimed_through_its_branches()
    test_a_conditional_claims_a_branch_it_is_handed()
    test_a_bare_state_handed_to_a_region_is_claimed()
    test_condition_fusion_produces_a_valid_sdfg()
    test_a_claim_inside_a_nested_sdfg_homes_to_that_nested_sdfg()
    test_a_deeper_nested_sdfg_keeps_the_parent_it_was_copied_with()
    test_loop_fission_clones_keep_their_nested_sdfg_parents_two_levels_down()
