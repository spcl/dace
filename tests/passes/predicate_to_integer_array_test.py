# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the pass that stores boolean arrays used as predicates in loops as integers."""
import numpy as np
import pytest

import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes import PredicateToIntegerArray

N = dace.symbol('N')


@dace.program
def masked_select(A: dace.float32[N], B: dace.float32[N]):
    mask = np.ndarray([N], dtype=np.bool_)
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            m >> mask[i]
            m = a > 0.5
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            m << mask[i]
            b >> B[i]
            b = a * 2 if m else a - 1


@dace.program
def derived_predicate(A: dace.float32[N], B: dace.float32[N]):
    mask = np.ndarray([N], dtype=np.bool_)
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            m >> mask[i]
            m = a > 0.5
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            m << mask[i]
            b >> B[i]
            t = m and a < 0.9
            b = a if t else 0


@dace.program
def data_use(A: dace.float32[N], B: dace.float32[N]):
    mask = np.ndarray([N], dtype=np.bool_)
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            m >> mask[i]
            m = a > 0.5
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            m << mask[i]
            b >> B[i]
            b = a + m


@dace.program
def combined_predicate(A: dace.float32[N], B: dace.float32[N]):
    above = np.ndarray([N], dtype=np.bool_)
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            m >> above[i]
            m = a > 0.5
    for i in range(1, N):
        flag = np.ndarray([1], dtype=np.bool_)
        with dace.tasklet:
            previous << above[i - 1]
            current << above[i]
            f >> flag[0]
            f = previous or current
        if flag[0]:
            B[i] = A[i]
        else:
            B[i] = -A[i]


def test_select_mask_retyped_and_results_unchanged():
    sdfg = masked_select.to_sdfg(simplify=True)
    assert PredicateToIntegerArray().apply_pass(sdfg, {}) == {sdfg.label: {'mask'}}
    assert sdfg.arrays['mask'].dtype == dace.int32
    # Elements are loaded and stored as integer values, not through pointer casts
    code = sdfg.generate_code()[0].clean_code
    assert '(bool *)' not in code and 'int m = mask[' in code
    a = np.random.rand(33).astype(np.float32)
    b = np.zeros(33, dtype=np.float32)
    sdfg(A=a, B=b, N=33)
    assert np.allclose(b, np.where(a > 0.5, a * 2, a - 1))


def test_predicate_through_local_retyped():
    sdfg = derived_predicate.to_sdfg(simplify=True)
    assert PredicateToIntegerArray().apply_pass(sdfg, {}) == {sdfg.label: {'mask'}}
    a = np.random.rand(33).astype(np.float32)
    b = np.zeros(33, dtype=np.float32)
    sdfg(A=a, B=b, N=33)
    assert np.allclose(b, np.where((a > 0.5) & (a < 0.9), a, 0))


def test_predicate_combined_into_branch_condition_retyped():
    """``above`` is not tested itself: in a loop, a tasklet combines two of its elements into ``flag``, which a branch
    tests."""
    sdfg = dace.SDFG('combined_predicate')
    sdfg.add_array('A', [8], dace.float32)
    sdfg.add_array('B', [8], dace.float32)
    sdfg.add_array('above', [8], dace.bool_, transient=True)
    sdfg.add_array('flag', [1], dace.bool_, transient=True)
    fill = sdfg.add_state('fill', is_start_block=True)
    fill.add_mapped_tasklet('aboves', {'i': '0:8'}, {'a': dace.Memlet('A[i]')},
                            'm = a > 0.5', {'m': dace.Memlet('above[i]')},
                            external_edges=True)
    loop = LoopRegion('loop', 'i < 8', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(loop)
    sdfg.add_edge(fill, loop, dace.InterstateEdge())
    combine = loop.add_state('combine', is_start_block=True)
    tasklet = combine.add_tasklet('combine', {
        'previous': None,
        'current': None
    }, {'f': None}, 'f = previous or current')
    above = combine.add_read('above')
    combine.add_edge(above, None, tasklet, 'previous', dace.Memlet('above[i - 1]'))
    combine.add_edge(above, None, tasklet, 'current', dace.Memlet('above[i]'))
    combine.add_edge(tasklet, 'f', combine.add_write('flag'), None, dace.Memlet('flag[0]'))
    join = loop.add_state('join')
    for label, code, condition in (('taken', 'b = a', 'flag[0]'), ('other', 'b = -a', 'not flag[0]')):
        branch = loop.add_state(label)
        write = branch.add_tasklet('write', {'a'}, {'b'}, code)
        branch.add_edge(branch.add_read('A'), None, write, 'a', dace.Memlet('A[i]'))
        branch.add_edge(write, 'b', branch.add_write('B'), None, dace.Memlet('B[i]'))
        loop.add_edge(combine, branch, dace.InterstateEdge(condition))
        loop.add_edge(branch, join, dace.InterstateEdge())

    assert PredicateToIntegerArray().apply_pass(sdfg, {}) == {sdfg.label: {'above', 'flag'}}
    a = np.random.rand(8).astype(np.float32)
    b = np.zeros(8, dtype=np.float32)
    sdfg(A=a, B=b)
    taken = (a[:-1] > 0.5) | (a[1:] > 0.5)
    assert np.allclose(b[1:], np.where(taken, a[1:], -a[1:]))


def test_predicate_combined_in_frontend_program_retyped():
    """Simplified, ``flag`` becomes a symbol assigned ``above[i - 1] or above[i]`` on an interstate edge, and the branch
    tests the symbol."""
    sdfg = combined_predicate.to_sdfg(simplify=True)
    result = PredicateToIntegerArray().apply_pass(sdfg, {})
    assert result is not None and 'above' in result[sdfg.label]
    a = np.random.rand(33).astype(np.float32)
    b = np.zeros(33, dtype=np.float32)
    sdfg(A=a, B=b, N=33)
    taken = (a[:-1] > 0.5) | (a[1:] > 0.5)
    assert np.allclose(b[1:], np.where(taken, a[1:], -a[1:]))


def test_data_use_not_retyped():
    sdfg = data_use.to_sdfg(simplify=True)
    assert PredicateToIntegerArray().apply_pass(sdfg, {}) is None


def test_mask_across_nested_sdfg_not_retyped():
    """Unsimplified, each tasklet sits in a nested SDFG, whose own copy of the array would need retyping too."""
    sdfg = masked_select.to_sdfg(simplify=False)
    assert PredicateToIntegerArray().apply_pass(sdfg, {}) is None


def _branch_sdfg(in_loop: bool, transient: bool = True, code: str = 'm = a > 0.5'):
    """``mask[i] = <code of A[i]>``, then ``B[i] = 1 if mask[i] else 2`` as a branch, in a loop over ``i`` (or for
    ``i = 0`` only). Returns the SDFG and the state computing the mask."""
    sdfg = dace.SDFG(f'predicate_branch_{int(in_loop)}')
    sdfg.add_array('A', [4], dace.float32)
    sdfg.add_array('B', [4], dace.float32)
    sdfg.add_array('mask', [4], dace.bool_, transient=transient)
    if in_loop:
        region = LoopRegion('loop', 'i < 4', 'i', 'i = 0', 'i = i + 1')
        sdfg.add_node(region, is_start_block=True)
        index = 'i'
    else:
        region, index = sdfg, '0'
    compute = region.add_state('compute', is_start_block=True)
    tasklet = compute.add_tasklet('compute', {'a'}, {'m'}, code)
    compute.add_edge(compute.add_read('A'), None, tasklet, 'a', dace.Memlet(f'A[{index}]'))
    compute.add_edge(tasklet, 'm', compute.add_write('mask'), None, dace.Memlet(f'mask[{index}]'))
    join = region.add_state('join')
    for label, value, condition in (('taken', 1, f'mask[{index}]'), ('other', 2, f'not mask[{index}]')):
        branch = region.add_state(label)
        write = branch.add_tasklet('write', {}, {'b'}, f'b = {value}')
        branch.add_edge(write, 'b', branch.add_write('B'), None, dace.Memlet(f'B[{index}]'))
        region.add_edge(compute, branch, dace.InterstateEdge(condition))
        region.add_edge(branch, join, dace.InterstateEdge())
    return sdfg, compute


@pytest.mark.parametrize(
    'code, predicate',
    [
        ('m = a > 0.5', lambda a: a > 0.5),
        ('m = a - 0.5', lambda a: a != 0.5),  # A float assigned to a bool is true unless zero: must not be truncated
    ])
def test_branch_in_loop_retyped_and_results_unchanged(code, predicate):
    sdfg, _ = _branch_sdfg(in_loop=True, code=code)
    assert PredicateToIntegerArray().apply_pass(sdfg, {}) == {sdfg.label: {'mask'}}
    a = np.array([0.2, 0.8, 0.5, 0.7], dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    sdfg(A=a, B=b)
    assert np.array_equal(b, np.where(predicate(a), 1, 2))


def test_branch_outside_loop_not_retyped():
    sdfg, _ = _branch_sdfg(in_loop=False)
    assert PredicateToIntegerArray().apply_pass(sdfg, {}) is None


def test_argument_not_retyped():
    sdfg, _ = _branch_sdfg(in_loop=True, transient=False)
    assert PredicateToIntegerArray().apply_pass(sdfg, {}) is None
    assert sdfg.arrays['mask'].dtype == dace.bool_


def test_copied_mask_not_retyped():
    sdfg, compute = _branch_sdfg(in_loop=True)
    sdfg.add_array('out', [4], dace.bool_)
    mask = next(n for n in compute.data_nodes() if n.data == 'mask')
    compute.add_nedge(mask, compute.add_write('out'), dace.Memlet('mask[0:4]'))
    assert PredicateToIntegerArray().apply_pass(sdfg, {}) is None


if __name__ == '__main__':
    test_select_mask_retyped_and_results_unchanged()
    test_predicate_through_local_retyped()
    test_predicate_combined_into_branch_condition_retyped()
    test_predicate_combined_in_frontend_program_retyped()
    test_data_use_not_retyped()
    test_mask_across_nested_sdfg_not_retyped()
    test_branch_in_loop_retyped_and_results_unchanged('m = a - 0.5', lambda a: a != 0.5)
    test_branch_outside_loop_not_retyped()
    test_argument_not_retyped()
    test_copied_mask_not_retyped()
