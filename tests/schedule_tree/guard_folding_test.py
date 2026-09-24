# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the schedule-tree guard folding and index-set splitting passes."""
import numpy as np
import pytest

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import (fold_guards, forward_substitute_conditions, fuse_rolled_loops,
                                                     merge_contiguous_loops, pair_complementary_guards,
                                                     remove_dead_assignments, reroll_statements, split_iteration_spaces)

N = dace.symbol('N')
M = dace.symbol('M')
CSTARR = np.array([0, 0, 0, 1, 1, 0, 0, 2], dtype=np.int32)


def _tree(sdfg: dace.SDFG) -> tn.ScheduleTreeRoot:
    stree = sdfg.as_schedule_tree()
    tn.validate_children_and_parents_align(stree, root=True)
    return stree


def _nodes(stree: tn.ScheduleTreeRoot, kind: type) -> list:
    return [n for n in stree.preorder_traversal() if isinstance(n, kind)]


def _conditions(stree: tn.ScheduleTreeRoot) -> list:
    return [n.condition.as_string for n in stree.preorder_traversal() if isinstance(n, (tn.IfScope, tn.ElifScope))]


def _for_headers(stree: tn.ScheduleTreeRoot, var: str = None) -> list:
    return [
        f.loop.init_statement.as_string + ' ; ' + f.loop.loop_condition.as_string for f in _nodes(stree, tn.ForScope)
        if var is None or f.loop.loop_variable == var
    ]


def _fold(stree: tn.ScheduleTreeRoot) -> int:
    count = fold_guards(stree)
    tn.validate_children_and_parents_align(stree, root=True)
    return count


def _split(stree: tn.ScheduleTreeRoot) -> int:
    count = split_iteration_spaces(stree)
    tn.validate_children_and_parents_align(stree, root=True)
    return count


def _run(stree: tn.ScheduleTreeRoot, **kwargs):
    # Simplification of the converted SDFG follows the configuration, as in ``roundtrip_test``.
    sdfg = stree.as_sdfg(simplify=dace.config.Config.get_bool('optimizer', 'automatic_simplification'))
    sdfg(**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# Guard folding
# ----------------------------------------------------------------------------------------------------------------------


def test_fold_atom_implied_by_loop_range():

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in range(1, N):
            if i >= 1 and A[i] > 0.5:
                B[i] = A[i] + 1.0

    stree = _tree(prog.to_sdfg())
    # Without simplification, the frontend computes the condition into scalars first (``__tmp1 = ...; if __tmp1``)
    forward_substitute_conditions(stree)
    assert _fold(stree) == 1
    (condition, ) = _conditions(stree)
    assert '>= 1' not in condition
    a = np.random.rand(10)
    b = np.zeros(10)
    _run(stree, A=a, B=b, N=10)
    expected = np.where(a > 0.5, a + 1.0, 0.0)
    expected[0] = 0.0
    assert np.allclose(b, expected)


def test_fold_removes_never_taken_and_splices_always_taken():

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in range(2, N):
            if i < 1:
                B[i] = 1.0
            elif i < 2:
                B[i] = 2.0
            else:
                B[i] = A[i] * 3.0

    stree = _tree(prog.to_sdfg(simplify=False))
    assert _fold(stree) == 2
    assert not _nodes(stree, tn.IfScope) and not _nodes(stree, tn.ElifScope) and not _nodes(stree, tn.ElseScope)
    a = np.random.rand(10)
    b = np.zeros(10)
    _run(stree, A=a, B=b, N=10)
    expected = a * 3.0
    expected[:2] = 0.0
    assert np.allclose(b, expected)


def test_fold_elif_promoted_to_if_and_else_kept():

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            if i < 0:
                B[i] = 1.0
            elif A[i] > 0.5:
                B[i] = 2.0
            else:
                B[i] = 3.0

    stree = _tree(prog.to_sdfg(simplify=False))
    assert _fold(stree) == 1
    assert len(_nodes(stree, tn.IfScope)) == 1 and not _nodes(stree, tn.ElifScope)
    assert len(_nodes(stree, tn.ElseScope)) == 1
    a = np.random.rand(10)
    b = np.zeros(10)
    _run(stree, A=a, B=b, N=10)
    assert np.allclose(b, np.where(a > 0.5, 2.0, 3.0))


def test_fold_nested_condition_narrows_range():
    """``i < 5`` inside ``if i < 3`` always holds; symbolic bounds need the ``Min``-aware comparison."""

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            if i < 3:
                if i < 5:
                    B[i] = A[i] + 1.0

    stree = _tree(prog.to_sdfg())
    assert _fold(stree) == 1
    assert _conditions(stree) == ['(i < 3)']


def test_fold_constant_array_atom():

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], cstarr: dace.int32[8]):
        for i in range(3, 5):
            if cstarr[i] > 0:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg()
    sdfg.add_constant('cstarr', CSTARR)
    stree = _tree(sdfg)
    fold_guards(stree)
    # The guard reads a scalar the frontend computes from ``cstarr[i]``, which folding alone does not see through
    if not _nodes(stree, tn.IfScope):
        a = np.arange(8, dtype=np.float64)
        b = np.zeros(8)
        _run(stree, A=a, B=b, cstarr=CSTARR)
        expected = np.zeros(8)
        expected[3:5] = a[3:5] + 1.0
        assert np.allclose(b, expected)


def test_fold_does_not_use_symbol_assigned_in_body():
    """A bound that the loop body reassigns is not loop-invariant; nothing may be folded against it."""
    sdfg = dace.SDFG('fold_assigned_symbol')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    loop = dace.sdfg.state.LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    guard = tn.IfScope(condition=dace.properties.CodeBlock('i < M'), children=[])
    assign = tn.AssignNode(name='M', value=dace.properties.CodeBlock('i + 1'), edge=dace.InterstateEdge())
    stree.children = []
    stree.add_child(tn.ForScope(loop=loop, children=[guard, assign]))
    assert fold_guards(stree) == 0
    assert _conditions(stree) == ['(i < M)']


# ----------------------------------------------------------------------------------------------------------------------
# Index-set splitting
# ----------------------------------------------------------------------------------------------------------------------


def test_split_outer_loop_on_guards_in_inner_loop():
    """Guards on ``j`` inside the ``i`` loop split the ``j`` loop (the pattern of stencil boundary conditions)."""

    @dace.program
    def prog(A: dace.float64[N, M], B: dace.float64[N, M]):
        for j in range(N):
            for i in range(M):
                B[j, i] = A[j, i]
                if j < 1:
                    B[j, i] = A[j, i] * 2.0
                if j >= N - 1:
                    B[j, i] = A[j, i] * 3.0

    stree = _tree(prog.to_sdfg())
    assert _split(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    # First row, interior, last row, and the row that is both first and last (only non-empty for ``N == 1``)
    assert len(_for_headers(stree, 'j')) == 4
    for n, m in ((6, 5), (1, 3)):
        a = np.random.rand(n, m)
        b = np.zeros((n, m))
        _run(stree, A=a, B=b, N=n, M=m)
        expected = a.copy()
        expected[0] = a[0] * 2.0
        expected[-1] = a[-1] * 3.0
        assert np.allclose(b, expected)


@pytest.mark.parametrize('min_trip_count', [1, 8])
def test_split_both_dimensions_of_nested_loops(min_trip_count):
    """Splitting ``j`` yields a boundary row (1 iteration) and the interior (15). The inner loop is split in both
    only without a trip-count threshold; with one, the boundary row keeps its (folded) guard on ``i``."""

    @dace.program
    def prog(A: dace.float64[16, 16], B: dace.float64[16, 16]):
        for j in range(16):
            for i in range(16):
                if i < 1 and j < 1:
                    B[j, i] = A[j, i] * 2.0
                if i >= 1 and j >= 1:
                    B[j, i] = A[j, i] * 3.0

    stree = _tree(prog.to_sdfg())
    forward_substitute_conditions(stree)
    assert split_iteration_spaces(stree, min_trip_count=min_trip_count) >= 2
    tn.validate_children_and_parents_align(stree, root=True)
    assert _conditions(stree) == ([] if min_trip_count == 1 else ['(i < 1)'])
    a = np.random.rand(16, 16)
    b = np.zeros((16, 16))
    _run(stree, A=a, B=b)
    expected = np.zeros((16, 16))
    expected[0, 0] = a[0, 0] * 2.0
    expected[1:, 1:] = a[1:, 1:] * 3.0
    assert np.allclose(b, expected)


def test_split_map_privatizes_transients():

    @dace.program
    def prog(A: dace.float64[10], B: dace.float64[10]):
        for i in dace.map[0:10]:
            t = A[i] * 2.0
            if i < 3:
                t = t + 1.0
            B[i] = t

    stree = _tree(prog.to_sdfg())
    assert _split(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    assert len(_nodes(stree, tn.MapScope)) == 2
    a = np.random.rand(10)
    b = np.zeros(10)
    _run(stree, A=a, B=b)
    expected = a * 2.0
    expected[:3] += 1.0
    assert np.allclose(b, expected)


def test_split_loop_keeps_values_flowing_between_parts():
    """A value written in the first iterations and read in later ones must reach the later copies (no renaming)."""

    @dace.program
    def prog(A: dace.float64[10], B: dace.float64[10]):
        t = np.zeros([1], dtype=np.float64)
        for k in range(10):
            if k == 0:
                t[0] = A[0] * 5.0
            B[k] = t[0] + A[k]

    stree = _tree(prog.to_sdfg())
    assert _split(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    a = np.random.rand(10)
    b = np.zeros(10)
    _run(stree, A=a, B=b)
    assert np.allclose(b, a[0] * 5.0 + a)


def test_split_constant_array_guard():

    sdfg = dace.SDFG('split_constant_array')
    sdfg.add_array('A', [8, 4], dace.float64)
    sdfg.add_array('B', [8, 4], dace.float64)
    sdfg.add_constant('cst', CSTARR)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    inner = dace.sdfg.state.LoopRegion('inner', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    guard = tn.IfScope(condition=dace.properties.CodeBlock('cst[k] > 0'), children=[_scale_tasklet('k, i', 7.0)])
    outer = dace.sdfg.state.LoopRegion('outer', 'k < 8', 'k', 'k = 0', 'k = k + 1')
    stree.children = []
    stree.add_child(tn.ForScope(loop=outer, children=[tn.ForScope(loop=inner, children=[guard])]))
    assert _split(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    assert _for_headers(stree, 'k') == ['k = 3 ; (k < 5)', 'k = 7 ; (k < 8)']
    a = np.random.rand(8, 4)
    b = a.copy()
    _run(stree, A=a, B=b)
    expected = a.copy()
    expected[CSTARR > 0] *= 7.0
    assert np.allclose(b, expected)


def _scale_tasklet(index: str, factor: float) -> tn.TaskletNode:
    """``B[index] = A[index] * factor`` as a tasklet node."""
    tasklet = dace.nodes.Tasklet('scale', {'a'}, {'b'}, f'b = a * {factor}')
    return tn.TaskletNode(node=tasklet,
                          in_memlets={'a': dace.Memlet(f'A[{index}]')},
                          out_memlets={'b': dace.Memlet(f'B[{index}]')})


def test_split_not_applied_when_loop_variable_read_after():

    @dace.program
    def prog(A: dace.float64[20], B: dace.float64[20], out: dace.int64[1]):
        for i in range(20):
            for j in range(3):
                if i >= 3:
                    B[i] = A[i] + j
        out[0] = i

    stree = _tree(prog.to_sdfg())
    assert _split(stree) == 0


def test_split_not_applied_with_break():
    sdfg = dace.SDFG('split_break')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    loop = dace.sdfg.state.LoopRegion('loop', 'i < 10', 'i', 'i = 0', 'i = i + 1')
    guard = tn.IfScope(condition=dace.properties.CodeBlock('i < 3'), children=[tn.BreakNode()])
    stree.children = []
    stree.add_child(tn.ForScope(loop=loop, children=[guard]))
    assert split_iteration_spaces(stree) == 0


# ----------------------------------------------------------------------------------------------------------------------
# Forward substitution into conditions
# ----------------------------------------------------------------------------------------------------------------------


def _bool_tasklet(target: str, code: str, inputs: dict) -> tn.TaskletNode:
    """``target = <code>`` as a tasklet reading ``inputs`` (connector -> memlet string)."""
    tasklet = dace.nodes.Tasklet('compute', set(inputs), {'out'}, f'out = {code}')
    return tn.TaskletNode(node=tasklet,
                          in_memlets={
                              c: dace.Memlet(m)
                              for c, m in inputs.items()
                          },
                          out_memlets={'out': dace.Memlet(f'{target}[0]')})


def _mask_tree(between: list = None) -> tn.ScheduleTreeRoot:
    """``for k: mask = cst[k] > 0; <between>; for i: if mask: B[k, i] = A[k, i] * 7``, with a constant ``cst``."""
    sdfg = dace.SDFG('substitute_mask')
    sdfg.add_array('A', [8, 4], dace.float64)
    sdfg.add_array('B', [8, 4], dace.float64)
    sdfg.add_array('cst', [8], dace.int32)
    sdfg.add_constant('cst', CSTARR, sdfg.arrays['cst'])
    sdfg.add_scalar('mask', dace.bool_, transient=True)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    inner = dace.sdfg.state.LoopRegion('inner', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    guard = tn.IfScope(condition=dace.properties.CodeBlock('mask'), children=[_scale_tasklet('k, i', 7.0)])
    outer = dace.sdfg.state.LoopRegion('outer', 'k < 8', 'k', 'k = 0', 'k = k + 1')
    body = [_bool_tasklet('mask', 'c > 0', {'c': 'cst[k]'})] + (between
                                                                or []) + [tn.ForScope(loop=inner, children=[guard])]
    stree.children = []
    stree.add_child(tn.ForScope(loop=outer, children=body))
    return stree


def test_substitute_mask_then_split_on_constant():
    stree = _mask_tree()
    assert forward_substitute_conditions(stree) == 1
    assert _conditions(stree) == ['(cst[k] > 0)']
    assert remove_dead_assignments(stree) == 1
    assert not _nodes(stree, tn.TaskletNode)[:-1]  # Only the guarded tasklet remains
    assert _split(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    assert _for_headers(stree, 'k') == ['k = 3 ; (k < 5)', 'k = 7 ; (k < 8)']
    a = np.random.rand(8, 4)
    b = a.copy()
    _run(stree, A=a, B=b, cst=CSTARR)
    expected = a.copy()
    expected[CSTARR > 0] *= 7.0
    assert np.allclose(b, expected)


def test_substitute_blocked_by_write_to_input_in_between():
    """``mask = cst[k] > 0; cst-independent write to mask's input``: here the input ``A`` is written in between."""
    stree = _mask_tree()
    loop_body = stree.children[0].children
    # Rewrite the mask to read ``A[k, 0]`` and write ``A[k, 0]`` between the assignment and the guard
    loop_body[0] = _bool_tasklet('mask', 'c > 0.5', {'c': 'A[k, 0]'})
    loop_body[0].parent = stree.children[0]
    writer = _scale_tasklet('k, 0', 2.0)
    writer.out_memlets = {'b': dace.Memlet('A[k, 0]')}
    stree.children[0].children.insert(1, writer)
    writer.parent = stree.children[0]
    assert forward_substitute_conditions(stree) == 0
    assert _conditions(stree) == ['mask']


def test_substitute_constant_scalar_from_outer_scope():
    """A flag set once at the top reaches guards nested in later loops (e.g. a configuration flag)."""
    sdfg = dace.SDFG('substitute_flag')
    sdfg.add_array('A', [8, 4], dace.float64)
    sdfg.add_array('B', [8, 4], dace.float64)
    sdfg.add_scalar('flag', dace.bool_, transient=True)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    inner = dace.sdfg.state.LoopRegion('inner', 'i < 4', 'i', 'i = 0', 'i = i + 1')
    guard = tn.IfScope(condition=dace.properties.CodeBlock('flag'), children=[_scale_tasklet('k, i', 7.0)])
    outer = dace.sdfg.state.LoopRegion('outer', 'k < 8', 'k', 'k = 0', 'k = k + 1')
    stree.children = []
    stree.add_children([
        _bool_tasklet('flag', 'True', {}),
        tn.ForScope(loop=outer, children=[tn.ForScope(loop=inner, children=[guard])])
    ])
    assert forward_substitute_conditions(stree) == 1
    assert remove_dead_assignments(stree) == 1
    assert _fold(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    a = np.random.rand(8, 4)
    b = np.zeros((8, 4))
    _run(stree, A=a, B=b)
    assert np.allclose(b, a * 7.0)


def test_substitute_not_across_loop_writing_the_value():
    """The loop between the assignment and the guard rewrites the flag in a later iteration."""
    sdfg = dace.SDFG('substitute_loop_write')
    sdfg.add_array('A', [8, 4], dace.float64)
    sdfg.add_array('B', [8, 4], dace.float64)
    sdfg.add_scalar('flag', dace.bool_, transient=True)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    guard = tn.IfScope(condition=dace.properties.CodeBlock('flag'), children=[_scale_tasklet('k, 0', 7.0)])
    outer = dace.sdfg.state.LoopRegion('outer', 'k < 8', 'k', 'k = 0', 'k = k + 1')
    stree.children = []
    stree.add_children([
        _bool_tasklet('flag', 'True', {}),
        tn.ForScope(loop=outer, children=[guard, _bool_tasklet('flag', 'c > 0.5', {'c': 'A[k, 1]'})])
    ])
    assert forward_substitute_conditions(stree) == 0


def test_substitute_skips_lossy_conversion():
    """``t: int32 = x * 0.5; if t:`` truncates; the condition must keep reading ``t``."""
    sdfg = dace.SDFG('substitute_lossy')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_scalar('t', dace.int32, transient=True)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    loop = dace.sdfg.state.LoopRegion('loop', 'k < 8', 'k', 'k = 0', 'k = k + 1')
    guard = tn.IfScope(condition=dace.properties.CodeBlock('t'), children=[_scale_tasklet('k', 7.0)])
    stree.children = []
    stree.add_child(tn.ForScope(loop=loop, children=[_bool_tasklet('t', 'a * 0.5', {'a': 'A[k]'}), guard]))
    assert forward_substitute_conditions(stree) == 0
    assert remove_dead_assignments(stree) == 0


# ----------------------------------------------------------------------------------------------------------------------
# Pairing of complementary guards
# ----------------------------------------------------------------------------------------------------------------------


def _pair_tree(first_condition: str, second_condition: str, first_body: list, trailing: list = None):
    """``for k in range(8): if <first>: <first_body>; if <second>: B[k] = A[k] * 3; <trailing>``."""
    sdfg = dace.SDFG('pair_guards')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_array('flag', [1], dace.float64)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    first = tn.IfScope(condition=dace.properties.CodeBlock(first_condition), children=first_body)
    second = tn.IfScope(condition=dace.properties.CodeBlock(second_condition), children=[_scale_tasklet('k', 3.0)])
    loop = dace.sdfg.state.LoopRegion('loop', 'k < 8', 'k', 'k = 0', 'k = k + 1')
    stree.children = []
    stree.add_child(tn.ForScope(loop=loop, children=[first, second] + (trailing or [])))
    return stree


@pytest.mark.parametrize('first, second, taken', [
    ('flag[0] > 0.5', 'not (flag[0] > 0.5)', lambda k, flag: np.full(8, flag > 0.5)),
    ('k < 4', 'k >= 4', lambda k, flag: k < 4),
    ('(k < 4) and (flag[0] > 0.5)', '(k >= 4) or (flag[0] <= 0.5)', lambda k, flag: (k < 4) & (flag > 0.5)),
])
def test_pair_complementary_guards(first, second, taken):
    stree = _pair_tree(first, second, [_scale_tasklet('k', 2.0)])
    assert pair_complementary_guards(stree) == 1
    assert len(_nodes(stree, tn.IfScope)) == 1 and len(_nodes(stree, tn.ElseScope)) == 1
    for flag in (0.0, 1.0):
        a = np.random.rand(8)
        b = np.zeros(8)
        _run(stree, A=a, B=b, flag=np.full(1, flag))
        assert np.allclose(b, np.where(taken(np.arange(8), flag), a * 2.0, a * 3.0))


def test_pair_not_when_first_body_writes_condition():
    clear = _bool_tasklet('flag', '0.0', {})
    stree = _pair_tree('flag[0] > 0.5', 'not (flag[0] > 0.5)', [_scale_tasklet('k', 2.0), clear])
    assert pair_complementary_guards(stree) == 0
    assert len(_nodes(stree, tn.IfScope)) == 2


def test_pair_not_for_unrelated_conditions():
    stree = _pair_tree('k < 4', 'k > 4', [_scale_tasklet('k', 2.0)])
    assert pair_complementary_guards(stree) == 0


def test_pair_not_when_second_guard_has_else():
    stree = _pair_tree('k < 4', 'k >= 4', [_scale_tasklet('k', 2.0)], trailing=[tn.ElseScope(children=[])])
    assert pair_complementary_guards(stree) == 0


def test_pair_then_fold_uses_negation():
    """After pairing, guards in the ``else`` are folded with the negated condition as a fact."""
    stree = _pair_tree('k >= 4', 'k < 4', [_scale_tasklet('k', 2.0)])
    loop = stree.children[0]
    undecided = tn.IfScope(condition=dace.properties.CodeBlock('k < 2'), children=[_scale_tasklet('k', 5.0)])
    decided = tn.IfScope(condition=dace.properties.CodeBlock('k < 6'), children=[_scale_tasklet('k', 7.0)])
    loop.children[1].add_children([undecided, decided])  # Within ``k < 4``: ``k < 6`` always holds
    assert pair_complementary_guards(stree) == 1
    assert _fold(stree) == 1
    assert _conditions(stree) == ['(k >= 4)', '(k < 2)']


# ----------------------------------------------------------------------------------------------------------------------
# Merging of contiguous loops
# ----------------------------------------------------------------------------------------------------------------------


def _loops_tree(ranges: list, bodies: list, prologue: list = None) -> tn.ScheduleTreeRoot:
    """Consecutive ``for k in range(lo, hi): <body>`` loops over ``A``/``B`` of 8 elements."""
    sdfg = dace.SDFG('merge_loops')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    stree.children = []
    stree.add_children(prologue or [])
    for k, ((lo, hi), body) in enumerate(zip(ranges, bodies)):
        loop = dace.sdfg.state.LoopRegion(f'loop{k}', f'k < {hi}', 'k', f'k = {lo}', 'k = k + 1')
        stree.add_child(tn.ForScope(loop=loop, children=body))
    return stree


def test_merge_contiguous_identical_loops():
    stree = _loops_tree([(0, 3), (3, 5), (5, 8)], [[_scale_tasklet('k', 2.0)] for _ in range(3)])
    assert merge_contiguous_loops(stree) == 2
    assert _for_headers(stree) == ['k = 0 ; (k < 8)']
    a = np.random.rand(8)
    b = np.zeros(8)
    _run(stree, A=a, B=b)
    assert np.allclose(b, a * 2.0)


@pytest.mark.parametrize(
    'ranges, second',
    [
        ([(0, 3), (3, 8)], 3.0),  # Same memlets, different code
        ([(0, 3), (4, 8)], 2.0),  # Not contiguous
        ([(0, 3), (2, 8)], 2.0),  # Overlapping
    ])
def test_merge_not_applied(ranges, second):
    stree = _loops_tree(ranges, [[_scale_tasklet('k', 2.0)], [_scale_tasklet('k', second)]])
    assert merge_contiguous_loops(stree) == 0
    assert len(_nodes(stree, tn.ForScope)) == 2


def test_merge_not_applied_when_first_body_changes_bound_of_second():
    assign = tn.AssignNode(name='M', value=dace.properties.CodeBlock('8'), edge=dace.InterstateEdge())
    stree = _loops_tree([(0, 3), (3, 'M')], [[_scale_tasklet('k', 2.0), assign], [_scale_tasklet('k', 2.0), assign]])
    assert merge_contiguous_loops(stree) == 0


def test_merge_restores_map_split_with_privatized_transients():
    """Splitting on ``i < 3`` and ``i >= 3`` of a tautological guard yields two maps with equal bodies, the second
    with its own copy of ``t``; merging restores a single map."""

    @dace.program
    def prog(A: dace.float64[10], B: dace.float64[10]):
        for i in dace.map[0:10]:
            t = A[i] * 2.0
            if i < 3 or i >= 3:
                B[i] = t

    stree = _tree(prog.to_sdfg())
    forward_substitute_conditions(stree)
    assert _split(stree) == 1
    assert len(_nodes(stree, tn.MapScope)) == 2 and not _nodes(stree, tn.IfScope)
    assert merge_contiguous_loops(stree) == 1
    (map_scope, ) = _nodes(stree, tn.MapScope)
    assert str(map_scope.node.map.range) == '0:10'
    a = np.random.rand(10)
    b = np.zeros(10)
    _run(stree, A=a, B=b)
    assert np.allclose(b, a * 2.0)


# ----------------------------------------------------------------------------------------------------------------------
# Rerolling
# ----------------------------------------------------------------------------------------------------------------------


def _copy_tasklet(out: str, inp: str, code: str = 'b = a') -> tn.TaskletNode:
    tasklet = dace.nodes.Tasklet('copy', {'a'}, {'b'}, code)
    return tn.TaskletNode(node=tasklet, in_memlets={'a': dace.Memlet(inp)}, out_memlets={'b': dace.Memlet(out)})


def _statements_tree(statements: list) -> tn.ScheduleTreeRoot:
    sdfg = dace.SDFG('reroll')
    sdfg.add_array('A', [6, 6], dace.float64)
    sdfg.add_array('B', [6, 6], dace.float64)
    sdfg.add_array('C', [6, 6], dace.float64)
    sdfg.add_state(is_start_block=True)
    stree = _tree(sdfg)
    stree.children = []
    stree.add_children(statements)
    return stree


def _check_statements(stree: tn.ScheduleTreeRoot, reference: tn.ScheduleTreeRoot):
    a = np.random.rand(6, 6)
    expected, result = [np.zeros((6, 6)), np.zeros((6, 6))], [np.zeros((6, 6)), np.zeros((6, 6))]
    _run(reference, A=a, B=expected[0], C=expected[1])
    _run(stree, A=a, B=result[0], C=result[1])
    assert np.allclose(result[0], expected[0]) and np.allclose(result[1], expected[1])


def test_reroll_corner_block_into_2d_loop():
    """``B[x, y] = A[5 - y, x]`` for ``y`` outer and ``x`` inner (the SW corner fill of a cubed-sphere tile)."""
    statements = lambda: [_copy_tasklet(f'B[{x}, {y}]', f'A[{5 - y}, {x}]') for y in range(3) for x in range(3)]
    stree, reference = _statements_tree(statements()), _statements_tree(statements())
    assert reroll_statements(stree) == 1
    (outer, ) = stree.children
    assert isinstance(outer, tn.ForScope) and isinstance(outer.children[0], tn.ForScope)
    assert len(_nodes(stree, tn.TaskletNode)) == 1
    _check_statements(stree, reference)


def test_reroll_keeps_order_between_signatures():
    """Runs are only formed from consecutive statements of one signature, so interleaved code keeps its order."""
    statements = lambda: ([_copy_tasklet(f'B[0, {x}]', f'A[1, {x}]')
                           for x in range(4)] + [_copy_tasklet('B[0, 1]', 'A[5, 5]', 'b = 2 * a')] +
                          [_copy_tasklet(f'B[0, {x}]', f'A[2, {x}]') for x in range(4)])
    stree, reference = _statements_tree(statements()), _statements_tree(statements())
    assert reroll_statements(stree) == 2
    assert [type(n).__name__ for n in stree.children] == ['ForScope', 'TaskletNode', 'ForScope']
    _check_statements(stree, reference)


def test_reroll_leaves_irregular_runs():
    statements = lambda: [_copy_tasklet(f'B[0, {x}]', f'A[0, {x}]') for x in (0, 1, 3, 5)]
    stree = _statements_tree(statements())
    assert reroll_statements(stree) == 0
    assert len(stree.children) == 4


def test_reroll_stacks_equal_runs():
    """Points 0, 1, 3, 4 are ``x + 3 * y`` over a 2x2 box."""
    statements = lambda: [_copy_tasklet(f'B[0, {x}]', f'A[1, {x}]') for x in (0, 1, 3, 4)]
    stree, reference = _statements_tree(statements()), _statements_tree(statements())
    assert reroll_statements(stree) == 1
    assert len(_nodes(stree, tn.ForScope)) == 2
    _check_statements(stree, reference)


def _two_corner_fills(second_input: str) -> list:
    """SW corner fills of ``B`` (from ``A``) and then of ``C`` (from the mirrored corner of ``second_input``)."""
    return ([_copy_tasklet(f'B[{x}, {y}]', f'A[{5 - y}, {x}]') for y in range(3) for x in range(3)] +
            [_copy_tasklet(f'C[{x}, {y}]', f'{second_input}[{2 - x}, {2 - y}]') for y in range(3) for x in range(3)])


def test_fuse_rolled_loops_of_independent_fields():
    stree, reference = _statements_tree(_two_corner_fills('A')), _statements_tree(_two_corner_fills('A'))
    assert reroll_statements(stree) == 2
    assert fuse_rolled_loops(stree) == 1
    (outer, ) = stree.children
    assert len(outer.children[0].children) == 2  # Both fields in one 3x3 nest
    _check_statements(stree, reference)


def test_fuse_rolled_loops_not_across_dependence():
    """``C[x, y]`` reads ``B[2 - x, 2 - y]``: in a merged nest, the first iterations would read ``B`` elements that
    are written only in later ones."""
    stree = _statements_tree(_two_corner_fills('B'))
    assert reroll_statements(stree) == 2
    assert fuse_rolled_loops(stree) == 0
    assert len(stree.children) == 2


if __name__ == '__main__':
    test_fold_atom_implied_by_loop_range()
    test_fold_removes_never_taken_and_splices_always_taken()
    test_fold_elif_promoted_to_if_and_else_kept()
    test_fold_nested_condition_narrows_range()
    test_fold_constant_array_atom()
    test_fold_does_not_use_symbol_assigned_in_body()
    test_split_outer_loop_on_guards_in_inner_loop()
    test_split_both_dimensions_of_nested_loops(8)
    test_split_map_privatizes_transients()
    test_split_loop_keeps_values_flowing_between_parts()
    test_split_constant_array_guard()
    test_split_not_applied_when_loop_variable_read_after()
    test_split_not_applied_with_break()
    test_substitute_mask_then_split_on_constant()
    test_substitute_blocked_by_write_to_input_in_between()
    test_substitute_constant_scalar_from_outer_scope()
    test_substitute_not_across_loop_writing_the_value()
    test_substitute_skips_lossy_conversion()
    test_pair_complementary_guards('k < 4', 'k >= 4', lambda k, flag: k < 4)
    test_pair_not_when_first_body_writes_condition()
    test_pair_not_for_unrelated_conditions()
    test_pair_not_when_second_guard_has_else()
    test_pair_then_fold_uses_negation()
    test_merge_contiguous_identical_loops()
    test_merge_not_applied([(0, 3), (3, 8)], 3.0)
    test_merge_not_applied_when_first_body_changes_bound_of_second()
    test_merge_restores_map_split_with_privatized_transients()
    test_reroll_corner_block_into_2d_loop()
    test_reroll_keeps_order_between_signatures()
    test_reroll_leaves_irregular_runs()
    test_reroll_stacks_equal_runs()
    test_fuse_rolled_loops_of_independent_fields()
    test_fuse_rolled_loops_not_across_dependence()
