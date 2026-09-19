# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np

import dace
from dace.transformation.interstate import ConditionFusion
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


def _branch_conditions(sdfg):
    return [
        cnd.as_string for cb, _ in sdfg.all_nodes_recursive() if isinstance(cb, ConditionalBlock)
        for cnd, _ in cb.branches if cnd is not None
    ]


def test_consecutive_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        if a1 > 10:
            s += 1
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 1


def test_consecutive_conditions2():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        else:
            s += 2
        if a1 > 10:
            s += 1
        else:
            s += 2
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 1


def test_nested_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            if a1 > 10:
                s += 2
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 1


def test_deeply_nested_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 + 2 > 10:
            if a1 + 2 > 10:
                if a0 + 1 > 10:
                    if a1 + 1 > 10:
                        if a0 > 10:
                            if a1 > 10:
                                s += 2
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 6

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 1


def test_dependent_consecutive_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        if s == 0:
            s += 1
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have not been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2


def test_dependent_consecutive_conditions2():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        a1 = a1 + 1
        if a1 > 10:
            s += 1
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have not been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2


def test_independent_consecutive_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 > 10:
            s += 1
        a3 = a[2] + 1
        if a1 > 10:
            s += 1
        a[2] = s + a3

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have not been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 2


def test_mixed_conditions():

    @dace.program
    def tester(a: dace.float64[3]):
        s = 0
        a0 = a[0]
        a1 = a[1]
        if a0 + 1 > 10:
            if a1 + 1 > 10:
                if a0 > 10:
                    if a1 > 10:
                        s += 2
        if a0 + 1 > 5:
            if a1 + 1 > 5:
                if a0 > 5:
                    if a1 > 5:
                        s += 2
        a[2] = s

    sdfg = tester.to_sdfg(simplify=True)
    sdfg.validate()

    # Should have exactly two conditional block
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 8

    # Apply the transformation
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.simplify()  # To fuse empty states
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    # Check that the conditional blocks have been fused
    cond_nodes = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cond_nodes) == 1


def test_identical_guards_no_duplicated_conjunct():
    """ConditionFusion of two identical guards must not emit a redundant
    ``(c) and (c)``: no fused branch condition repeats the predicate. The
    satisfiable branch is exactly the original predicate. Value-preserving."""

    @dace.program
    def tester(a: dace.float64[4]):
        c = a[0]
        s = 0.0
        if c > 10:
            s += 1.0
        if c > 10:
            s += 2.0
        a[3] = s

    base = tester.to_sdfg(simplify=True)
    sdfg = tester.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    cbs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cbs) == 1, f"guards not merged: {len(cbs)}"
    conds = _branch_conditions(sdfg)
    # ConditionFusion simplifies its own output: the redundant ``(c) and
    # (c)`` collapses and the unsatisfiable cross-terms are dropped, so no
    # surviving condition is a conjunction.
    assert all(' and ' not in c for c in conds), f"non-minimal condition: {conds}"

    for v in (5.0, 20.0):  # guard not-taken / taken
        ref, out = np.array([v, 0, 0, 0]), np.array([v, 0, 0, 0])
        copy.deepcopy(base)(a=ref)
        sdfg(a=out)
        assert np.allclose(out, ref), f"value mismatch at a[0]={v}: {out} vs {ref}"


def test_three_identical_guards_no_duplicated_conjunct():
    """Three chained identical guards: ConditionFusion still emits no
    branch with a duplicated predicate, and is value-preserving."""

    @dace.program
    def tester(a: dace.float64[5]):
        c = a[0]
        s = 0.0
        if c > 10:
            s += 1.0
        if c > 10:
            s += 2.0
        if c > 10:
            s += 4.0
        a[4] = s

    base = tester.to_sdfg(simplify=True)
    sdfg = tester.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(ConditionFusion)
    sdfg.validate()

    cbs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, ConditionalBlock)]
    assert len(cbs) == 1, f"guards not merged: {len(cbs)}"
    conds = _branch_conditions(sdfg)
    assert all(' and ' not in c for c in conds), f"non-minimal condition: {conds}"

    for v in (5.0, 20.0):
        ref, out = np.array([v, 0, 0, 0, 0]), np.array([v, 0, 0, 0, 0])
        copy.deepcopy(base)(a=ref)
        sdfg(a=out)
        assert np.allclose(out, ref), f"value mismatch at a[0]={v}: {out} vs {ref}"


def guarded_nested_sdfg(label, sdfg, condition, value):
    """A ConditionalBlock whose single branch writes ``value`` to ``A[0]`` through a NESTED SDFG."""
    cb = ConditionalBlock(label)
    sdfg.add_node(cb)
    body = ControlFlowRegion(label + '_body', sdfg=sdfg)
    cb.add_branch(CodeBlock(condition), body)
    state = body.add_state(label + '_write', is_start_block=True)
    inner = dace.SDFG(label + '_inner')
    inner.add_array('a', [4], dace.float64)
    inner_state = inner.add_state('w', is_start_block=True)
    tasklet = inner_state.add_tasklet('w', {}, {'o'}, f'o = {value}')
    inner_state.add_edge(tasklet, 'o', inner_state.add_write('a'), None, dace.Memlet('a[0]'))
    nsdfg = state.add_nested_sdfg(inner, {}, {'a'})
    state.add_edge(nsdfg, 'a', state.add_write('A'), None, dace.Memlet('A[0:4]'))
    return cb


def test_fusing_copied_branches_leaves_every_region_addressable():
    """A fused branch is a DEEP COPY, and a copy arrives with no usable CFG list.

    ``SDFG.__deepcopy__`` leaves a nested copy's ``_cfg_list`` empty and ``ControlFlowBlock`` skips
    the attribute outright -- both by design, leaving it to whoever grafts the copy into a tree. So
    the branch this fusion copies brings a NestedSDFG whose ``cfg_id`` raises ``SDFG (...) is not in
    list`` the first time any later pass asks for it, which is how the CloudSC parallelize pipeline
    died in its fuse phase. Every region must be addressable in the tree it now lives in.
    """
    sdfg = dace.SDFG('cond_fusion_keeps_the_cfg_list')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_symbol('c', dace.int32)
    sdfg.add_symbol('d', dace.int32)
    # INDEPENDENT guards, so the merge-matching-guards shortcut declines and the fusion takes the
    # cartesian branch product -- the path that deep-copies each branch region wholesale.
    first = guarded_nested_sdfg('g1', sdfg, 'c > 0', 1.0)
    second = guarded_nested_sdfg('g2', sdfg, 'd > 0', 2.0)
    start = sdfg.add_state('start', is_start_block=True)
    sdfg.add_edge(start, first, dace.InterstateEdge())
    sdfg.add_edge(first, second, dace.InterstateEdge())
    sdfg.validate()

    assert sdfg.apply_transformations_repeated(ConditionFusion) > 0, 'the independent guards did not fuse'
    sdfg.validate()

    # The property that failed: every region resolves its own index in the list it carries.
    for region in sdfg.all_control_flow_regions(recursive=True):
        assert region.cfg_list, f'{type(region).__name__} {region.label!r} carries an empty CFG list'
        assert region.cfg_id >= 0

    # And the copied branches still compute: the later write wins where both guards hold.
    for c, d, expected in ((1, 0, 1.0), (0, 1, 2.0), (1, 1, 2.0), (0, 0, 0.0)):
        out = np.zeros(4)
        sdfg(A=out, c=c, d=d)
        assert out[0] == expected, f'c={c} d={d}: got {out[0]}, want {expected}'


if __name__ == "__main__":
    test_consecutive_conditions()
    test_consecutive_conditions2()
    test_nested_conditions()
    test_deeply_nested_conditions()
    test_dependent_consecutive_conditions()
    test_dependent_consecutive_conditions2()
    test_independent_consecutive_conditions()
    test_mixed_conditions()
    test_identical_guards_no_duplicated_conjunct()
    test_three_identical_guards_no_duplicated_conjunct()
    test_fusing_copied_branches_leaves_every_region_addressable()
