# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A transient sized by a symbol that an interstate edge assigns is allocated after that assignment.

The symbol is not free, so the transient cannot be allocated at the start of the SDFG; it is allocated at the
dominator of its accesses. When the assignment sits on the edge leaving that dominator and enters a control flow
region, the allocation was emitted in the dominator, ahead of the assignment, and sized the array with an undefined
value.
"""
import re

import numpy as np

import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

LENGTH = 4


def add_fill(region: ControlFlowRegion, label: str) -> LoopRegion:
    """``b[i] = i`` for ``i`` in ``0:LENGTH``, as a loop in ``region``."""
    loop = LoopRegion(label, f'i < {LENGTH}', 'i', 'i = 0', 'i = i + 1')
    region.add_node(loop)
    body = loop.add_state(f'{label}_body', is_start_block=True)
    tasklet = body.add_tasklet(f'{label}_write', {}, {'o'}, 'o = i')
    body.add_edge(tasklet, 'o', body.add_write('b'), None, dace.Memlet('b[i]'))
    return loop


def add_drain(region: ControlFlowRegion, label: str) -> LoopRegion:
    """``out[i] = b[i]`` for ``i`` in ``0:LENGTH``, as a loop in ``region``."""
    loop = LoopRegion(label, f'i < {LENGTH}', 'i', 'i = 0', 'i = i + 1')
    region.add_node(loop)
    body = loop.add_state(f'{label}_body', is_start_block=True)
    tasklet = body.add_tasklet(f'{label}_read', {'x'}, {'o'}, 'o = x')
    body.add_edge(body.add_read('b'), None, tasklet, 'x', dace.Memlet('b[i]'))
    body.add_edge(tasklet, 'o', body.add_write('out'), None, dace.Memlet('out[i]'))
    return loop


def sized_by_assignment(name: str) -> dace.SDFG:
    """``K = Nt`` on the edge into two loops that write and read ``b[K]``; ``K`` is defined by nothing else."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('Nt', dace.int64)
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_array('out', [LENGTH], dace.float64)
    sdfg.add_transient('b', ['K'], dace.float64)

    head = sdfg.add_state('head', is_start_block=True)
    fill = add_fill(sdfg, 'fill')
    drain = add_drain(sdfg, 'drain')
    sdfg.add_edge(head, fill, dace.InterstateEdge(assignments={'K': 'Nt'}))
    sdfg.add_edge(fill, drain, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def sized_by_assignment_into_a_branch(name: str) -> dace.SDFG:
    """Like :func:`sized_by_assignment`, but the assignment enters a conditional that holds both loops."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('Nt', dace.int64)
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_array('out', [LENGTH], dace.float64)
    sdfg.add_transient('b', ['K'], dace.float64)

    head = sdfg.add_state('head', is_start_block=True)
    branch = ConditionalBlock('branch')
    sdfg.add_node(branch)
    taken = ControlFlowRegion('taken')
    branch.add_branch(dace.properties.CodeBlock('Nt > 0'), taken)
    fill = add_fill(taken, 'fill')
    drain = add_drain(taken, 'drain')
    taken.add_edge(fill, drain, dace.InterstateEdge())
    sdfg.add_edge(head, branch, dace.InterstateEdge(assignments={'K': 'Nt'}))
    sdfg.validate()
    return sdfg


def sized_by_assignment_inside_both_branches(name: str) -> dace.SDFG:
    """``K`` is assigned on an edge inside each branch of a conditional, and both loops follow the conditional."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('Nt', dace.int64)
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_array('out', [LENGTH], dace.float64)
    sdfg.add_transient('b', ['K'], dace.float64)

    head = sdfg.add_state('head', is_start_block=True)
    branch = ConditionalBlock('branch')
    sdfg.add_node(branch)
    for label, condition, size in (('taken', 'Nt > 0', 'Nt'), ('not_taken', 'Nt <= 0', 'Nt + 4')):
        region = ControlFlowRegion(label)
        branch.add_branch(dace.properties.CodeBlock(condition), region)
        enter = region.add_state(f'{label}_enter', is_start_block=True)
        region.add_edge(enter, region.add_state(f'{label}_sized'), dace.InterstateEdge(assignments={'K': size}))
    fill = add_fill(sdfg, 'fill')
    drain = add_drain(sdfg, 'drain')
    sdfg.add_edge(head, branch, dace.InterstateEdge())
    sdfg.add_edge(branch, fill, dace.InterstateEdge())
    sdfg.add_edge(fill, drain, dace.InterstateEdge())
    sdfg.validate()
    return sdfg


def sized_by_assignment_in_a_nested_sdfg(name: str) -> dace.SDFG:
    """:func:`sized_by_assignment` as the body of a nested SDFG."""
    sdfg = dace.SDFG(name)
    sdfg.add_symbol('Nt', dace.int64)
    sdfg.add_array('out', [LENGTH], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    nested = state.add_nested_sdfg(sized_by_assignment(f'{name}_inner'), set(), {'out'}, symbol_mapping={'Nt': 'Nt'})
    state.add_edge(nested, 'out', state.add_write('out'), None, dace.Memlet('out[0:4]'))
    sdfg.validate()
    return sdfg


def allocation_and_assignment_lines(sdfg: dace.SDFG):
    """The line indices of the allocation of ``b`` and of the last assignment of ``K``, and whether ``b`` is freed."""
    lines = sdfg.generate_code()[0].clean_code.splitlines()
    # The allocation and the free read differently with and without an aligned allocation.
    alloc = next(i for i, line in enumerate(lines) if re.search(r'\bb = new\b', line))
    assign = max(i for i, line in enumerate(lines) if re.match(r'\s*K = ', line))
    freed = any(re.search(r'delete\[\]\s*b\b|delete\[\]\s*\(\s*b\b', line) for line in lines)
    return alloc, assign, freed


def run(sdfg: dace.SDFG) -> np.ndarray:
    out = np.zeros(LENGTH)
    sdfg(out=out, Nt=LENGTH)
    return out


def test_a_size_assigned_into_loops_is_defined_before_the_allocation():
    sdfg = sized_by_assignment('alloc_after_assignment_loops')
    alloc, assign, freed = allocation_and_assignment_lines(sdfg)
    assert assign < alloc
    assert freed, 'the array is never freed'
    assert np.allclose(run(sdfg), np.arange(LENGTH))


def test_a_size_assigned_into_a_conditional_is_defined_before_the_allocation():
    sdfg = sized_by_assignment_into_a_branch('alloc_after_assignment_branch')
    alloc, assign, freed = allocation_and_assignment_lines(sdfg)
    assert assign < alloc
    assert freed, 'the array is never freed'
    assert np.allclose(run(sdfg), np.arange(LENGTH))


def test_a_size_assigned_inside_both_branches_is_defined_before_the_allocation():
    sdfg = sized_by_assignment_inside_both_branches('alloc_after_assignment_inside_branches')
    alloc, assign, freed = allocation_and_assignment_lines(sdfg)
    assert assign < alloc
    assert freed, 'the array is never freed'
    assert np.allclose(run(sdfg), np.arange(LENGTH))


def test_a_size_assigned_in_a_nested_sdfg_is_defined_before_the_allocation():
    sdfg = sized_by_assignment_in_a_nested_sdfg('alloc_after_assignment_nested')
    alloc, assign, freed = allocation_and_assignment_lines(sdfg)
    assert assign < alloc
    assert freed, 'the array is never freed'
    assert np.allclose(run(sdfg), np.arange(LENGTH))


if __name__ == '__main__':
    test_a_size_assigned_into_loops_is_defined_before_the_allocation()
    test_a_size_assigned_into_a_conditional_is_defined_before_the_allocation()
    test_a_size_assigned_inside_both_branches_is_defined_before_the_allocation()
    test_a_size_assigned_in_a_nested_sdfg_is_defined_before_the_allocation()
