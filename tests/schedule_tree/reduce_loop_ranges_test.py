# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the schedule-tree loop range reduction pass."""
import numpy as np
import pytest

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.passes import reduce_loop_ranges

N = dace.symbol('N')
M = dace.symbol('M')
CSTARR = np.array([0, 0, 0, 1, 1, 0, 0, 2], dtype=np.int32)


def _tree(sdfg: dace.SDFG) -> tn.ScheduleTreeRoot:
    stree = sdfg.as_schedule_tree()
    tn.validate_children_and_parents_align(stree, root=True)
    return stree


def _reduce(stree: tn.ScheduleTreeRoot) -> int:
    count = reduce_loop_ranges(stree)
    tn.validate_children_and_parents_align(stree, root=True)
    return count


def _nodes(stree: tn.ScheduleTreeRoot, kind: type) -> list:
    return [n for n in stree.preorder_traversal() if isinstance(n, kind)]


def _for_headers(stree: tn.ScheduleTreeRoot) -> list:
    return [
        f.loop.init_statement.as_string + ' ; ' + f.loop.loop_condition.as_string for f in _nodes(stree, tn.ForScope)
    ]


def _map_ranges(stree: tn.ScheduleTreeRoot) -> list:
    return sorted(str(m.node.map.range) for m in _nodes(stree, tn.MapScope))


def _run(stree: tn.ScheduleTreeRoot, **kwargs):
    # Simplification of the converted SDFG follows the configuration, as in ``roundtrip_test``.
    sdfg = stree.as_sdfg(simplify=dace.config.Config.get_bool('optimizer', 'automatic_simplification'))
    sdfg(**kwargs)


@pytest.mark.parametrize('use_map', [False, True])
def test_stree_symbolic_guard(use_map):

    @dace.program
    def loop(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            if i >= 1 and i < M:
                B[i] = A[i] * 2.0

    @dace.program
    def mapped(A: dace.float64[N], B: dace.float64[N]):
        for i in dace.map[0:N]:
            if i >= 1 and i < M:
                B[i] = A[i] * 2.0

    sdfg = (mapped if use_map else loop).to_sdfg()
    if 'M' not in sdfg.symbols:
        sdfg.add_symbol('M', dace.int64)
    stree = _tree(sdfg)
    assert _reduce(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    if use_map:
        (rng, ) = _map_ranges(stree)
        assert rng.startswith('1:') and 'Min' in rng
    else:
        (header, ) = _for_headers(stree)
        assert header.startswith('i = 1 ;') and 'Min' in header
    for n, m in ((16, 9), (16, 40), (16, 1)):
        a = np.random.rand(n)
        b = np.zeros(n)
        _run(stree, A=a, B=b, N=n, M=m)
        expected = np.zeros(n)
        expected[1:min(n, m)] = a[1:min(n, m)] * 2.0
        assert np.allclose(b, expected)


def test_stree_constant_array_loop_split():
    """The frontend's ``cstarr_index = cstarr[i]`` prologue is folded in and dropped; the loop splits in two."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], cstarr: dace.int32[8]):
        for i in range(8):
            if cstarr[i] > 0:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg()
    sdfg.add_constant('cstarr', CSTARR)
    stree = _tree(sdfg)
    assert _reduce(stree) == 1
    assert _for_headers(stree) == ['i = 3 ; (i < 5)', 'i = 7 ; (i < 8)']
    assert not _nodes(stree, tn.IfScope) and not _nodes(stree, tn.AssignNode)
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    _run(stree, A=a, B=b, cstarr=CSTARR)
    expected = np.zeros(8)
    expected[[3, 4, 7]] = a[[3, 4, 7]] + 1.0
    assert np.allclose(b, expected)


def test_stree_constant_array_map_split_hand_built():
    """A hand-built map tree (``assign flag = (cstarr[i] > 0); if flag: ...``) splits into two maps that convert
    back to a valid, correct SDFG."""
    from dace import data, subsets
    from dace.properties import CodeBlock
    from dace.sdfg import InterstateEdge, nodes

    tasklet = nodes.Tasklet('t', {'a'}, {'b'}, 'b = a + 1')
    body = tn.TaskletNode(node=tasklet, in_memlets={'a': dace.Memlet('A[i]')}, out_memlets={'b': dace.Memlet('B[i]')})
    assign = tn.AssignNode(name='flag',
                           value=CodeBlock('(cstarr[i] > 0)'),
                           edge=InterstateEdge(assignments={'flag': '(cstarr[i] > 0)'}))
    guard = tn.IfScope(condition=CodeBlock('flag'), children=[body])
    entry = nodes.MapEntry(nodes.Map('m', ['i'], subsets.Range.from_string('0:8')))
    scope = tn.MapScope(node=entry, children=[assign, guard])
    stree = tn.ScheduleTreeRoot(name='split',
                                containers={
                                    'A': data.Array(dace.float64, [8]),
                                    'B': data.Array(dace.float64, [8]),
                                    'cstarr': data.Array(dace.int32, [8]),
                                },
                                symbols={'flag': dace.int32},
                                constants={'cstarr': (data.Array(dace.int32, [8]), CSTARR)},
                                arg_names=['A', 'B', 'cstarr'],
                                children=[scope])
    tn.validate_children_and_parents_align(stree, root=True)
    assert _reduce(stree) == 1
    assert _map_ranges(stree) == ['3:5', '7']
    assert not _nodes(stree, tn.IfScope) and not _nodes(stree, tn.AssignNode)
    maps = _nodes(stree, tn.MapScope)
    assert maps[0].node is not maps[1].node and maps[0].children[0].node is not maps[1].children[0].node
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    _run(stree, A=a, B=b, cstarr=CSTARR)
    expected = np.zeros(8)
    expected[[3, 4, 7]] = a[[3, 4, 7]] + 1.0
    assert np.allclose(b, expected)


def test_stree_frontend_map_constant_array_split():
    """The frontend's map program: the nested SDFG is flattened into the tree, the guard reads ``cstarr[i]`` through
    a symbol assignment, and the map splits into two maps with their own copies of the scope-local temporaries."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], cstarr: dace.int32[8]):
        for i in dace.map[0:8]:
            if cstarr[i] > 0:
                B[i] = A[i] + 1.0

    sdfg = prog.to_sdfg()
    sdfg.add_constant('cstarr', CSTARR)
    stree = _tree(sdfg)
    assert _reduce(stree) == 1
    assert _map_ranges(stree) == ['3:5', '7']
    assert not _nodes(stree, tn.IfScope) and not _nodes(stree, tn.AssignNode)
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    _run(stree, A=a, B=b, cstarr=CSTARR)
    expected = np.zeros(8)
    expected[[3, 4, 7]] = a[[3, 4, 7]] + 1.0
    assert np.allclose(b, expected)


def test_stree_residual_and_prologue_kept_when_read():

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            if i >= 2 and A[i] > 4.5:
                B[i] = A[i] + 1.0

    stree = _tree(prog.to_sdfg())
    assert _reduce(stree) == 1
    (header, ) = _for_headers(stree)
    assert header.startswith('i = 2 ;')
    (guard, ) = _nodes(stree, tn.IfScope)
    assert '4.5' in guard.condition.as_string and 'i >= 2' not in guard.condition.as_string
    a = np.arange(20, dtype=np.float64)
    b = np.zeros(20)
    _run(stree, A=a, B=b, N=20)
    expected = np.zeros(20)
    expected[5:] = a[5:] + 1.0
    assert np.allclose(b, expected)


def test_stree_two_dimensional_map_both_params():

    @dace.program
    def prog(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i, j in dace.map[0:N, 0:N]:
            if i >= 2 and j < M:
                B[i, j] = A[i, j] + 1.0

    sdfg = prog.to_sdfg()
    if 'M' not in sdfg.symbols:
        sdfg.add_symbol('M', dace.int64)
    stree = _tree(sdfg)
    assert _reduce(stree) == 2
    assert not _nodes(stree, tn.IfScope)
    (m, ) = _nodes(stree, tn.MapScope)
    assert str(m.node.map.range.ranges[0][0]) == '2' and 'Min' in str(m.node.map.range.ranges[1][1])
    n, mm = 6, 4
    a = np.random.rand(n, n)
    b = np.zeros((n, n))
    _run(stree, A=a, B=b, N=n, M=mm)
    expected = np.zeros((n, n))
    expected[2:, :mm] = a[2:, :mm] + 1.0
    assert np.allclose(b, expected)


def test_stree_nested_loop_in_map():

    @dace.program
    def prog(A: dace.float64[N, N], B: dace.float64[N, N]):
        for i in dace.map[0:N]:
            if i >= 1:
                for j in range(N):
                    if j < M:
                        B[i, j] = A[i, j] + 1.0

    sdfg = prog.to_sdfg()
    if 'M' not in sdfg.symbols:
        sdfg.add_symbol('M', dace.int64)
    stree = _tree(sdfg)
    assert _reduce(stree) == 2
    assert not _nodes(stree, tn.IfScope)
    n, m = 6, 4
    a = np.random.rand(n, n)
    b = np.zeros((n, n))
    _run(stree, A=a, B=b, N=n, M=m)
    expected = np.zeros((n, n))
    expected[1:, :m] = a[1:, :m] + 1.0
    assert np.allclose(b, expected)


def test_stree_contradiction_removes_scope():

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8]):
        for i in dace.map[0:8]:
            if i > 100:
                B[i] = A[i] + 1.0

    stree = _tree(prog.to_sdfg())
    assert _reduce(stree) == 1
    assert not _nodes(stree, tn.MapScope)


def test_stree_loop_variable_read_after_loop_untouched():

    @dace.program
    def prog(A: dace.float64[20], B: dace.float64[20], out: dace.int64[1]):
        for i in range(20):
            if i >= 3:
                B[i] = A[i] + 1.0
        out[0] = i

    stree = _tree(prog.to_sdfg())
    assert _nodes(stree, tn.IfScope)
    assert _reduce(stree) == 0
    assert _nodes(stree, tn.IfScope)


def test_stree_same_loop_variable_reused_later_is_fine():
    """A later loop rebinding ``i`` does not count as a read of this loop's final value."""

    @dace.program
    def prog(A: dace.float64[20], B: dace.float64[20]):
        for i in range(20):
            if i >= 3:
                B[i] = A[i] + 1.0
        for i in range(20):
            B[i] = B[i] * 2.0

    stree = _tree(prog.to_sdfg())
    assert _reduce(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    a = np.arange(20, dtype=np.float64)
    b = np.zeros(20)
    _run(stree, A=a, B=b)
    expected = np.zeros(20)
    expected[3:] = (a[3:] + 1.0) * 2.0
    assert np.allclose(b, expected)


def test_stree_data_dependent_guard_untouched():

    @dace.program
    def prog(A: dace.float64[N], B: dace.float64[N]):
        for i in range(N):
            if A[i] > 4.5:
                B[i] = A[i] + 1.0

    stree = _tree(prog.to_sdfg())
    assert _reduce(stree) == 0


# ---------------------------------------------------------------------------------------------------------------------
# Hoisting, tasklet-computed masks and if/else splitting (the stencil pattern)
# ---------------------------------------------------------------------------------------------------------------------

MODE = np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.0])


def _stencil_program():

    @dace.program
    def prog(inp: dace.float64[6, 30, 30], out: dace.float64[6, 30, 30], scale: dace.float64[6], mode: dace.float64[6]):
        for k in range(6):
            for j in range(30):
                for i in range(30):
                    if mode[k] == 0.0:
                        if i >= 2 and i < 30 - 2 and j >= 2 and j < 30 - 2:
                            out[k, j, i] = scale[k] * inp[k, j, i]
                    else:
                        out[k, j, i] = scale[k] * inp[k, j, i] * 2.0

    return prog


def _stencil_reference(inp, scale):
    expected = np.zeros_like(inp)
    for k in range(6):
        if MODE[k] == 0.0:
            expected[k, 2:28, 2:28] = scale[k] * inp[k, 2:28, 2:28]
        else:
            expected[k] = scale[k] * inp[k] * 2.0
    return expected


def _check_stencil(stree):
    inp = np.random.rand(6, 30, 30)
    scale = np.random.rand(6)
    out = np.zeros((6, 30, 30))
    _run(stree, inp=inp, out=out, scale=scale, mode=MODE)
    assert np.allclose(out, _stencil_reference(inp, scale))


def test_stree_stencil_mask_hoisted_and_k_split():
    """``for k: for j: for i: if mode[k] == 0: (if 2 <= i, j < 28: A) else: B`` with constant ``mode``: the mask (a
    copy + tasklet before simplification, a symbol assignment after) is hoisted out of ``i`` and ``j``, the ``k`` loop
    splits into the runs where the mask holds (running ``A`` with ``i``/``j`` reduced to ``2:28``) and where it does
    not (running ``B``), and no conditional remains."""
    sdfg = _stencil_program().to_sdfg()
    sdfg.add_constant('mode', MODE)
    stree = _tree(sdfg)
    assert _reduce(stree) > 0
    assert not _nodes(stree, tn.IfScope) and not _nodes(stree, tn.ElseScope)
    k_loops = [f for f in _nodes(stree, tn.ForScope) if f.loop.loop_variable == 'k']
    # mode == 0 on k in {0, 1}, {3}, {5}; mode != 0 on {2}, {4}: five k loops in iteration order.
    assert [f.loop.init_statement.as_string for f in k_loops] == ['k = 0', 'k = 2', 'k = 3', 'k = 4', 'k = 5']
    inner = {
        f.loop.loop_variable: f.loop.init_statement.as_string
        for f in k_loops[0].preorder_traversal() if isinstance(f, tn.ForScope)
    }
    assert inner['i'] == 'i = 2' and inner['j'] == 'j = 2'
    inner = {
        f.loop.loop_variable: f.loop.init_statement.as_string
        for f in k_loops[1].preorder_traversal() if isinstance(f, tn.ForScope)
    }
    assert inner['i'] == 'i = 0' and inner['j'] == 'j = 0'
    _check_stencil(stree)


def test_stree_stencil_sibling_negated_guards():
    """The same stencil with the else branch written as a sibling ``if not mask`` (as some frontends emit)."""
    sdfg = _stencil_program().to_sdfg()
    sdfg.add_constant('mode', MODE)
    stree = _tree(sdfg)
    from dace.frontend.python import astutils
    for else_scope in _nodes(stree, tn.ElseScope):
        guard = else_scope.parent.children[else_scope.parent.children.index(else_scope) - 1]
        assert isinstance(guard, tn.IfScope)
        negated = tn.IfScope(condition=tn.CodeBlock([astutils.negate_expr(guard.condition.code[0])]),
                             children=else_scope.children)
        else_scope.parent.children[else_scope.parent.children.index(else_scope)] = negated
        negated.parent = else_scope.parent
    tn.validate_children_and_parents_align(stree, root=True)
    assert 'if (not' in stree.as_string()
    assert _reduce(stree) > 0
    assert not _nodes(stree, tn.IfScope)
    _check_stencil(stree)


def test_stree_hoist_only_runtime_mask_stays():
    """A mask over runtime data is not folded, but it is hoisted (nothing in the loops writes ``mode``) and the
    ``i``/``j`` restriction is reduced: ``for k: if mode[k] == 0: for j in 2:28: for i in 2:28: A else: for j: for i: B``."""
    sdfg = _stencil_program().to_sdfg()  # ``mode`` stays a runtime array
    stree = _tree(sdfg)
    assert _reduce(stree) > 0
    guards = _nodes(stree, tn.IfScope)
    assert len(guards) == 1 and 'mode' in guards[0].condition.as_string
    assert isinstance(guards[0].parent, tn.ForScope) and guards[0].parent.loop.loop_variable == 'k'
    inner = {
        f.loop.loop_variable: f.loop.init_statement.as_string
        for f in guards[0].preorder_traversal() if isinstance(f, tn.ForScope)
    }
    assert inner == {'i': 'i = 2', 'j': 'j = 2'}
    _check_stencil(stree)


def test_stree_guard_after_plain_statements_splits_loop():
    """The guard follows other work in the body (``S; t = cst[k] > 0; if t: A``): the loop is partitioned so ``S``
    runs everywhere and ``A`` only where the constant says so, in the original iteration order."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], C: dace.float64[8], cstarr: dace.int32[8]):
        for k in range(8):
            B[k] = A[k] * 3.0
            if cstarr[k] > 0:
                C[k] = B[k] + 1.0

    sdfg = prog.to_sdfg()
    sdfg.add_constant('cstarr', CSTARR)
    stree = _tree(sdfg)
    assert _reduce(stree) == 1
    assert not _nodes(stree, tn.IfScope)
    assert _for_headers(stree) == ['k = 0 ; (k < 3)', 'k = 3 ; (k < 5)', 'k = 5 ; (k < 7)', 'k = 7 ; (k < 8)']
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    c = np.zeros(8)
    _run(stree, A=a, B=b, C=c, cstarr=CSTARR)
    assert np.allclose(b, a * 3.0)
    expected = np.zeros(8)
    expected[[3, 4, 7]] = a[[3, 4, 7]] * 3.0 + 1.0
    assert np.allclose(c, expected)


def test_stree_two_guards_in_one_body():
    """Two independently guarded blocks in one body are decided jointly on the cells of the partition."""

    @dace.program
    def prog(A: dace.float64[8], B: dace.float64[8], C: dace.float64[8], cstarr: dace.int32[8]):
        for k in range(8):
            if cstarr[k] == 0:
                B[k] = A[k] * 3.0
            else:
                B[k] = A[k] * 5.0
            if cstarr[k] > 1:
                C[k] = A[k] + 1.0

    sdfg = prog.to_sdfg()
    sdfg.add_constant('cstarr', CSTARR)
    stree = _tree(sdfg)
    assert _reduce(stree) == 1
    assert not _nodes(stree, tn.IfScope) and not _nodes(stree, tn.ElseScope)
    assert _for_headers(stree) == ['k = 0 ; (k < 3)', 'k = 3 ; (k < 5)', 'k = 5 ; (k < 7)', 'k = 7 ; (k < 8)']
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    c = np.zeros(8)
    _run(stree, A=a, B=b, C=c, cstarr=CSTARR)
    assert np.allclose(b, np.where(CSTARR == 0, a * 3.0, a * 5.0))
    expected = np.zeros(8)
    expected[7] = a[7] + 1.0
    assert np.allclose(c, expected)


def test_stree_no_fission_for_guard_after_statement_with_invariant_condition():
    """``for i: S; if M > 3: A`` must not be turned into ``for i: S; if M > 3: for i: A`` (that would reorder)."""

    @dace.program
    def prog(A: dace.float64[20], B: dace.float64[20], C: dace.float64[20]):
        for i in range(20):
            B[i] = A[i] * 3.0
            if M > 3:
                C[i] = B[i] + 1.0

    stree = _tree(prog.to_sdfg())
    assert _reduce(stree) == 0


def test_stree_sibling_guards_not_paired_when_first_branch_changes_condition():
    """``if c: A`` / ``if not c: B`` are not exclusive when ``A`` changes ``c``: here ``A`` clears ``flag`` at
    ``i == 4``, so both branches run in that iteration. Splitting on ``i >= 4`` must keep ``B`` there."""
    from dace import data
    from dace.properties import CodeBlock
    from dace.sdfg import nodes

    def tasklet(code: str, inputs: dict, output: str) -> tn.TaskletNode:
        node = nodes.Tasklet('t', set(inputs), {'out'}, f'out = {code}')
        return tn.TaskletNode(node=node,
                              in_memlets={
                                  c: dace.Memlet(m)
                                  for c, m in inputs.items()
                              },
                              out_memlets={'out': dace.Memlet(output)})

    first = tn.IfScope(condition=CodeBlock('(i >= 4) and (flag[0] > 0)'),
                       children=[tasklet('-1.0', {}, 'flag[0]'),
                                 tasklet('1.0', {}, 'B[i]')])
    second = tn.IfScope(condition=CodeBlock('not ((i >= 4) and (flag[0] > 0))'),
                        children=[tasklet('b + a', {
                            'a': 'A[i]',
                            'b': 'B[i]'
                        }, 'B[i]')])
    loop = dace.sdfg.state.LoopRegion('loop', 'i < 8', 'i', 'i = 0', 'i = i + 1')
    stree = tn.ScheduleTreeRoot(name='sibling_guards',
                                containers={
                                    'A': data.Array(dace.float64, [8]),
                                    'B': data.Array(dace.float64, [8]),
                                    'flag': data.Array(dace.float64, [1]),
                                },
                                arg_names=['A', 'B', 'flag'],
                                children=[tn.ForScope(loop=loop, children=[first, second])])
    tn.validate_children_and_parents_align(stree, root=True)
    _reduce(stree)
    a = np.arange(8, dtype=np.float64)
    b = np.zeros(8)
    _run(stree, A=a, B=b, flag=np.ones(1))
    expected = a.copy()
    expected[4] += 1.0
    assert np.allclose(b, expected)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-x', '-q']))
