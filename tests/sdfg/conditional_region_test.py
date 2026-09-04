# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import copy

import numpy as np
import dace
from dace.properties import CodeBlock
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation.dataflow import PruneConnectors
import dace.serialize


def test_cond_region_if():
    sdfg = dace.SDFG('regular_if')
    sdfg.add_array('A', (1, ), dace.float32)
    sdfg.add_symbol('i', dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)

    if1 = ConditionalBlock('if1')
    sdfg.add_node(if1)
    sdfg.add_edge(state0, if1, InterstateEdge())

    if_body = ControlFlowRegion('if_body', sdfg=sdfg)
    if1.add_branch(CodeBlock('i == 1'), if_body)

    state1 = if_body.add_state('state1', is_start_block=True)
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = 100')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[0]'))

    assert sdfg.is_valid()
    A = np.ones((1, ), dtype=np.float32)
    sdfg(i=1, A=A)
    assert A[0] == 100

    A = np.ones((1, ), dtype=np.float32)
    sdfg(i=0, A=A)
    assert A[0] == 1


def test_serialization():
    sdfg = SDFG('test_serialization')
    cond_region = ConditionalBlock('cond_region')
    sdfg.add_node(cond_region, is_start_block=True)
    sdfg.add_symbol('i', dace.int32)

    for j in range(10):
        cfg = ControlFlowRegion(f'cfg_{j}', sdfg)
        cfg.add_state('noop')
        cond_region.add_branch(CodeBlock(f'i == {j}'), cfg)

    assert sdfg.is_valid()

    new_sdfg = SDFG.from_json(sdfg.to_json())
    assert new_sdfg.is_valid()
    new_cond_region: ConditionalBlock = new_sdfg.nodes()[0]
    for j in range(10):
        condition, cfg = new_cond_region.branches[j]
        assert condition == CodeBlock(f'i == {j}')
        assert cfg.label == f'cfg_{j}'


def test_if_else():
    sdfg = dace.SDFG('regular_if_else')
    sdfg.add_array('A', (1, ), dace.float32)
    sdfg.add_symbol('i', dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)

    if1 = ConditionalBlock('if1')
    sdfg.add_node(if1)
    sdfg.add_edge(state0, if1, InterstateEdge())

    if_body = ControlFlowRegion('if_body', sdfg=sdfg)
    state1 = if_body.add_state('state1', is_start_block=True)
    acc_a = state1.add_access('A')
    t1 = state1.add_tasklet('t1', None, {'a'}, 'a = 100')
    state1.add_edge(t1, 'a', acc_a, None, dace.Memlet('A[0]'))
    if1.add_branch(CodeBlock('i == 1'), if_body)

    else_body = ControlFlowRegion('else_body', sdfg=sdfg)
    state2 = else_body.add_state('state1', is_start_block=True)
    acc_a2 = state2.add_access('A')
    t2 = state2.add_tasklet('t2', None, {'a'}, 'a = 200')
    state2.add_edge(t2, 'a', acc_a2, None, dace.Memlet('A[0]'))
    if1.add_branch(CodeBlock('i == 0'), else_body)

    assert sdfg.is_valid()
    A = np.ones((1, ), dtype=np.float32)
    sdfg(i=1, A=A)
    assert A[0] == 100

    A = np.ones((1, ), dtype=np.float32)
    sdfg(i=0, A=A)
    assert A[0] == 200


def test_conditional_block_read_and_write_sets_branch_condition():
    """A ConditionalBlock's own branch conditions live in ``_branches``, invisible to
    ``nodes()``/``edges()``, which only see the branch bodies -- they must still show up in
    read_and_write_sets().
    """
    nsdfg = dace.SDFG('nested')
    nsdfg.add_array('cond', [1], dace.bool_)
    nsdfg.add_array('a', [1], dace.float64)

    if_region = ConditionalBlock('if')
    nsdfg.add_node(if_region, is_start_block=True)

    then_body = ControlFlowRegion('then', sdfg=nsdfg)
    if_region.add_branch(CodeBlock('cond[0]'), then_body)
    s1 = then_body.add_state('s1', is_start_block=True)
    s1.add_edge(s1.add_tasklet('t1', {}, {'o'}, 'o = 1'), 'o', s1.add_write('a'), None, dace.Memlet('a[0]'))

    else_body = ControlFlowRegion('else', sdfg=nsdfg)
    if_region.add_branch(CodeBlock('not (cond[0])'), else_body)
    s2 = else_body.add_state('s2', is_start_block=True)
    s2.add_edge(s2.add_tasklet('t2', {}, {'o'}, 'o = 2'), 'o', s2.add_write('a'), None, dace.Memlet('a[0]'))

    nsdfg.validate()

    read_set, _ = if_region.read_and_write_sets()
    assert 'cond' in read_set

    sdfg = dace.SDFG('outer')
    sdfg.add_array('COND_arr', [1], dace.bool_)
    sdfg.add_array('A', [1], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    nsdfg_node = state.add_nested_sdfg(nsdfg, inputs={'cond'}, outputs={'a'})
    state.add_edge(state.add_read('COND_arr'), None, nsdfg_node, 'cond', dace.Memlet('COND_arr[0]'))
    state.add_edge(nsdfg_node, 'a', state.add_write('A'), None, dace.Memlet('A[0]'))
    sdfg.validate()

    # Non-regression guard: PruneConnectors reads via SDFG.read_and_write_sets(), which already
    # special-cases control-flow regions on its own; this just locks in that it stays sound.
    assert not PruneConnectors.can_be_applied_to(sdfg=sdfg, nsdfg=nsdfg_node, expr_index=0, permissive=False)


def test_add_branch_owns_deep_copied_subtree():
    """``copy.deepcopy`` of a region taken out of its SDFG resolves ``_sdfg`` through the memo and
    comes back owner-less. ``add_branch`` is what re-establishes ownership -- for the whole subtree,
    not just the branch root, exactly as ``add_node`` does."""
    sdfg = dace.SDFG('add_branch_ownership')
    sdfg.add_array('A', (1, ), dace.float32)
    sdfg.add_symbol('i', dace.int32)
    state0 = sdfg.add_state('state0', is_start_block=True)

    if1 = ConditionalBlock('if1')
    sdfg.add_node(if1)
    sdfg.add_edge(state0, if1, InterstateEdge())

    if_body = ControlFlowRegion('if_body', sdfg=sdfg)
    inner = ControlFlowRegion('inner')
    if_body.add_node(inner, is_start_block=True)
    state1 = inner.add_state('state1', is_start_block=True)
    state1.add_edge(state1.add_tasklet('t1', None, {'a'}, 'a = 100'), 'a', state1.add_access('A'), None,
                    dace.Memlet('A[0]'))
    if1.add_branch(CodeBlock('i == 1'), if_body)

    orphan = copy.deepcopy(if_body)
    assert all(block.sdfg is None for block in orphan.all_control_flow_blocks())

    if1.add_branch(None, orphan)
    assert orphan.sdfg is sdfg
    for block in orphan.all_control_flow_blocks():
        assert block.sdfg is sdfg, f'{block} kept a stale owner after add_branch'
    sdfg.validate()

    A = np.ones((1, ), dtype=np.float32)
    sdfg(i=0, A=A)
    assert A[0] == 100


if __name__ == '__main__':
    test_cond_region_if()
    test_serialization()
    test_if_else()
    test_conditional_block_read_and_write_sets_branch_condition()
    test_add_branch_owns_deep_copied_subtree()
