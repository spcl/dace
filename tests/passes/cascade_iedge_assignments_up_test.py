# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for ``CascadeInterstateEdgeAssignmentsUp``.

Each test is a small focused ``@dace.program`` exercising one of the
binding rules / legality predicates from ``CASCADE_UP_DESIGN.md``:

* outer-only invariant single hoist;
* two-loop shared hoist (one move serves both siblings);
* mixed outer + loop-var (refuse, RHS reads the loop variable);
* data-dependent (refuse, RHS reads an array);
* conditional-guarded assignment (L5: refuse inside a ConditionalBlock
  branch);
* transitive chain (``s1 = K + 1; s2 = 2 * s1`` -- both hoist together);
* all-or-nothing (assignment can clear one of two enclosing loops --
  must NOT hoist per the user's binding principle);
* idempotence (a second pass-application is a no-op);
* value preservation (numerical equivalence against numpy);
* manual iedge inside a hand-built SDFG to test the moving directly,
  bypassing the Python frontend's promotion idiosyncrasies.

The pass is invoked standalone here. Pipeline-integration coverage lives
in the ``tests/canonicalize/`` suite (see
``canonicalize_symbol_lifting_test.py`` and the cloudsc-style xfails the
design doc pins).
"""
from typing import List, Tuple

import numpy as np
import pytest

import dace
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import (CascadeInterstateEdgeAssignmentsUp)

N = dace.symbol('N')
K = dace.symbol('K')
M = dace.symbol('M')


def _apply(sdfg: dace.SDFG) -> int:
    """Run the pass once and return the count of moves."""
    return CascadeInterstateEdgeAssignmentsUp().apply_pass(sdfg, {}) or 0


def _all_iedge_assignments(sdfg: dace.SDFG) -> List[Tuple[str, str, str]]:
    """``(containing-region-label, lhs, rhs)`` for every iedge assignment
    anywhere in ``sdfg``."""
    out: List[Tuple[str, str, str]] = []
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            for lhs, rhs in e.data.assignments.items():
                out.append((cfg.label, lhs, str(rhs)))
    return out


def _assignments_inside_loops(sdfg: dace.SDFG) -> List[Tuple[str, str, str]]:
    """Iedge assignments whose containing region is, or sits inside, a
    ``LoopRegion``."""
    out: List[Tuple[str, str, str]] = []
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        # is cfg a LoopRegion, or inside one?
        in_loop = False
        g = cfg
        while g is not None and g is not sdfg:
            if isinstance(g, LoopRegion):
                in_loop = True
                break
            g = getattr(g, 'parent_graph', None)
        if not in_loop:
            continue
        for e in cfg.edges():
            for lhs, rhs in e.data.assignments.items():
                out.append((cfg.label, lhs, str(rhs)))
    return out


# ----------------------------------------------------------------------
# T1. Outer-only invariant single hoist
# ----------------------------------------------------------------------


def test_outer_only_single_hoist_manual():
    """Hand-built SDFG: a single LoopRegion whose body's iedge carries
    ``kp1 = K + 1``. The hoist target is the SDFG root."""
    sdfg = dace.SDFG('outer_only_single_hoist')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('kp1', dace.int64)

    loop = LoopRegion('outer_loop',
                      condition_expr='i < N',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1')
    s_pre = loop.add_state('body_pre', is_start_block=True)
    s_post = loop.add_state('body_post')
    loop.add_edge(s_pre, s_post, InterstateEdge(assignments={'kp1': 'K + 1'}))
    sdfg.add_node(loop, is_start_block=True)

    # Before: assignment inside the loop.
    inside_before = _assignments_inside_loops(sdfg)
    assert ('outer_loop', 'kp1', 'K + 1') in inside_before

    moved = _apply(sdfg)
    assert moved == 1

    # After: assignment at SDFG root, none left in the loop body.
    inside_after = _assignments_inside_loops(sdfg)
    assert ('outer_loop', 'kp1', 'K + 1') not in inside_after
    root_assignments = [(lhs, rhs) for cfg in [sdfg] for e in cfg.edges() for lhs, rhs in e.data.assignments.items()]
    assert ('kp1', 'K + 1') in root_assignments


# ----------------------------------------------------------------------
# T2. Two-loop shared hoist (one move serves both sibling loops)
# ----------------------------------------------------------------------


def _make_loop_with_iedge(name: str, key: str, rhs: str, loop_var: str = 'i') -> LoopRegion:
    loop = LoopRegion(name,
                      condition_expr=f'{loop_var} < N',
                      loop_var=loop_var,
                      initialize_expr=f'{loop_var} = 0',
                      update_expr=f'{loop_var} = {loop_var} + 1')
    s_pre = loop.add_state(f'{name}_body_pre', is_start_block=True)
    s_post = loop.add_state(f'{name}_body_post')
    loop.add_edge(s_pre, s_post, InterstateEdge(assignments={key: rhs}))
    return loop


def test_two_sibling_loops_each_hoist_independently():
    """Two sibling LoopRegions, each carrying ``kp1 = K + 1`` in its body.
    Each is hoisted independently; the binding rule does not require
    cross-sibling sharing -- only that each move clears all enclosing
    loops of its source. Both end up at the SDFG root."""
    sdfg = dace.SDFG('two_sibling_loops')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('kp1', dace.int64)

    l1 = _make_loop_with_iedge('loop_a', 'kp1', 'K + 1')
    l2 = _make_loop_with_iedge('loop_b', 'kp1', 'K + 1')
    sdfg.add_node(l1, is_start_block=True)
    sdfg.add_node(l2)
    sdfg.add_edge(l1, l2, InterstateEdge())

    moved = _apply(sdfg)
    assert moved == 2  # one per sibling

    assert not _assignments_inside_loops(sdfg)


# ----------------------------------------------------------------------
# T3. Mixed outer + loop-var: refuse
# ----------------------------------------------------------------------


def test_mixed_outer_plus_loop_var_refuses():
    """``tmp = K + i`` inside a loop with variable ``i``: the rhs reads
    ``i``, so the move out of that loop is illegal (L1). The pass leaves
    it in place."""
    sdfg = dace.SDFG('mixed_outer_plus_loop_var')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('tmp', dace.int64)

    loop = _make_loop_with_iedge('mix_loop', 'tmp', 'K + i')
    sdfg.add_node(loop, is_start_block=True)

    moved = _apply(sdfg)
    assert moved == 0
    assert ('mix_loop', 'tmp', 'K + i') in _assignments_inside_loops(sdfg)


# ----------------------------------------------------------------------
# T4. Data-dependent (refuse): rhs reads an array
# ----------------------------------------------------------------------


def test_data_dependent_assignment_refuses_or_stays():
    """``tmp = arr[i]`` would require a per-iteration array read; the
    pass does not move it (no symbol on the LHS resolves to an outer-
    invariant value). We pin: the assignment must remain inside the loop.

    The Python frontend lowers ``tmp = arr[i]`` to a tasklet, not an
    iedge assignment (iedges only carry symbolic expressions, not array
    reads). So we hand-build an iedge whose rhs references a symbol
    that is *assigned by another iedge inside the same loop* -- that is
    the practical 'data-dependent' shape the pass needs to refuse."""
    sdfg = dace.SDFG('data_dependent')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('per_iter', dace.int64)
    sdfg.add_symbol('derived', dace.int64)

    loop = LoopRegion('dd_loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    s0 = loop.add_state('s0', is_start_block=True)
    s1 = loop.add_state('s1')
    s2 = loop.add_state('s2')
    loop.add_edge(s0, s1, InterstateEdge(assignments={'per_iter': 'K * i'}))
    loop.add_edge(s1, s2, InterstateEdge(assignments={'derived': 'per_iter + 1'}))
    sdfg.add_node(loop, is_start_block=True)

    moved = _apply(sdfg)
    assert moved == 0  # ``per_iter`` depends on i; ``derived`` depends on per_iter
    inside = _assignments_inside_loops(sdfg)
    assert ('dd_loop', 'per_iter', 'K*i') in inside or ('dd_loop', 'per_iter', 'K * i') in inside
    assert ('dd_loop', 'derived', 'per_iter + 1') in inside


# ----------------------------------------------------------------------
# T5. Conditional-guarded assignment (L5): refuse inside ConditionalBlock
# ----------------------------------------------------------------------


def test_conditional_branch_refuses_l5():
    """``if c: { kp1 = K + 1 }`` inside a loop -- the iedge lives on an
    edge inside a ConditionalBlock branch. L5 refuses the hoist (the
    conservative subset). The assignment stays where it is."""
    sdfg = dace.SDFG('conditional_branch')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('c', dace.int64)
    sdfg.add_symbol('kp1', dace.int64)

    loop = LoopRegion('guard_loop',
                      condition_expr='i < N',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1')

    # The branch is a CFR; inside it, a 2-state body whose iedge carries
    # the assignment.
    branch = ControlFlowRegion('branch_body')
    bs0 = branch.add_state('bs0', is_start_block=True)
    bs1 = branch.add_state('bs1')
    branch.add_edge(bs0, bs1, InterstateEdge(assignments={'kp1': 'K + 1'}))

    cb = ConditionalBlock('guard')
    cb.add_branch(dace.properties.CodeBlock('c > 0'), branch)
    loop.add_node(cb, is_start_block=True)
    sdfg.add_node(loop, is_start_block=True)

    moved = _apply(sdfg)
    assert moved == 0
    # The assignment must still be on the same iedge, inside the branch
    # body inside the loop.
    found = False
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        if cfg.label != 'branch_body':
            continue
        for e in cfg.edges():
            if e.data.assignments.get('kp1') in ('K + 1', 'K+1'):
                found = True
    assert found, 'guarded assignment was unexpectedly moved'


# ----------------------------------------------------------------------
# T6. Cross-NSDFG hoist (L6): v1 refuses, documenting future-work xfail
# ----------------------------------------------------------------------


@pytest.mark.xfail(strict=True,
                   reason=('L6 NSDFG-boundary passthrough is deferred to a v2 of '
                           'CascadeInterstateEdgeAssignmentsUp (see CASCADE_UP_DESIGN.md). '
                           'v1 walks within one SDFG only.'))
def test_cross_nsdfg_hoist_l6():
    """An assignment inside a NestedSDFG, whose rhs references an outer
    symbol -- v2 will hoist past the NSDFG via ``symbol_mapping`` and
    drop the now-shadowed inner declaration. v1 refuses."""
    outer = dace.SDFG('outer')
    outer.add_symbol('K', dace.int64)
    inner = dace.SDFG('inner')
    inner.add_symbol('K', dace.int64)
    inner.add_symbol('kp1', dace.int64)
    loop = _make_loop_with_iedge('inner_loop', 'kp1', 'K + 1')
    inner.add_node(loop, is_start_block=True)
    st = outer.add_state('st', is_start_block=True)
    st.add_nested_sdfg(inner, inputs=set(), outputs=set(), symbol_mapping={'K': 'K'})

    moved = _apply(outer)
    assert moved >= 1  # would-be: 1 (hoisted across the NSDFG boundary)


# ----------------------------------------------------------------------
# T7. Transitive chain: ``s1 = K + 1; s2 = 2 * s1``
# ----------------------------------------------------------------------


def test_transitive_chain_both_hoist():
    """``s1 = K + 1`` then ``s2 = 2 * s1``: both invariants must hoist.
    The two assignments may end up on different iedges at the SDFG root,
    but neither must remain in the loop body."""
    sdfg = dace.SDFG('transitive_chain')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('s1', dace.int64)
    sdfg.add_symbol('s2', dace.int64)

    loop = LoopRegion('chain_loop',
                      condition_expr='i < N',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1')
    s0 = loop.add_state('s0', is_start_block=True)
    s1 = loop.add_state('s1')
    s2 = loop.add_state('s2')
    loop.add_edge(s0, s1, InterstateEdge(assignments={'s1': 'K + 1'}))
    loop.add_edge(s1, s2, InterstateEdge(assignments={'s2': '2 * s1'}))
    sdfg.add_node(loop, is_start_block=True)

    # Run to a fixed point (the second hoist becomes legal only after the
    # first one has cleared ``s1`` out of the loop).
    moved_total = 0
    for _ in range(4):
        n = _apply(sdfg)
        moved_total += n
        if n == 0:
            break
    assert moved_total == 2
    assert not _assignments_inside_loops(sdfg)


# ----------------------------------------------------------------------
# T8. All-or-nothing: hoist one level legal, two levels not -> refuse
# ----------------------------------------------------------------------


def test_all_or_nothing_one_level_only_refuses():
    """Two enclosing loops ``for i: for j: kp1 = K + j + 1``. The inner
    move (out of ``j``) is legal -- but the outer move (out of ``i``) is
    not, because rhs reads ``j`` from the perspective of any scope above
    ``j``. The binding all-or-nothing rule says: do nothing.

    The rhs reads ``j``, so the assignment cannot legally clear the
    ``j``-loop in the first place -- which means it ALSO cannot clear
    the outer ``i``-loop. The pass is a no-op."""
    sdfg = dace.SDFG('all_or_nothing')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('kp1', dace.int64)

    outer = LoopRegion('outer', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    inner = LoopRegion('inner', condition_expr='j < N', loop_var='j', initialize_expr='j = 0', update_expr='j = j + 1')
    s0 = inner.add_state('s0', is_start_block=True)
    s1 = inner.add_state('s1')
    # rhs references the inner loop var -> moving past inner is illegal
    # -> the binding rule forbids any move (since outer is also enclosing).
    inner.add_edge(s0, s1, InterstateEdge(assignments={'kp1': 'K + j + 1'}))
    outer.add_node(inner, is_start_block=True)
    sdfg.add_node(outer, is_start_block=True)

    moved = _apply(sdfg)
    assert moved == 0
    inside = _assignments_inside_loops(sdfg)
    assert ('inner', 'kp1', 'K + j + 1') in inside


def test_all_or_nothing_two_enclosing_loops_invariant_both_hoist():
    """Sanity contrast for T8: when rhs is invariant w.r.t. *both*
    enclosing loops (``kp1 = K + 1``), the move clears both and lands at
    the SDFG root in one go."""
    sdfg = dace.SDFG('all_or_nothing_yes')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('kp1', dace.int64)

    outer = LoopRegion('outer', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    inner = LoopRegion('inner', condition_expr='j < N', loop_var='j', initialize_expr='j = 0', update_expr='j = j + 1')
    s0 = inner.add_state('s0', is_start_block=True)
    s1 = inner.add_state('s1')
    inner.add_edge(s0, s1, InterstateEdge(assignments={'kp1': 'K + 1'}))
    outer.add_node(inner, is_start_block=True)
    sdfg.add_node(outer, is_start_block=True)

    moved = _apply(sdfg)
    assert moved == 1
    assert not _assignments_inside_loops(sdfg)


# ----------------------------------------------------------------------
# T9. Idempotence: a second application is a no-op
# ----------------------------------------------------------------------


def test_idempotent_second_application_noop():
    sdfg = dace.SDFG('idempotent')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('kp1', dace.int64)
    loop = _make_loop_with_iedge('idem_loop', 'kp1', 'K + 1')
    sdfg.add_node(loop, is_start_block=True)
    first = _apply(sdfg)
    second = _apply(sdfg)
    assert first == 1 and second == 0


# ----------------------------------------------------------------------
# T10. Value preservation: end-to-end via the Python frontend
# ----------------------------------------------------------------------


@dace.program
def frontend_kp1(a: dace.float64[N], b: dace.float64[N], K_in: dace.int64):
    """Python-frontend kernel where ``range(0, K_in + 1)`` triggers
    promotion of ``K_in + 1`` to a coined symbol. After
    canonicalize-pipeline rearrangements that bury the assignment inside
    the loop, cascade-up must lift it back to the SDFG root so the
    range-symbol is constant across iterations."""
    for i in range(0, K_in + 1):
        b[i] = a[i] * 2.0


def test_frontend_kp1_value_preserving():
    n, k_in = 10, 6
    a = np.random.rand(n)
    sdfg = frontend_kp1.to_sdfg(simplify=True)
    # Standalone pass (not the full canonicalize) -- the assignment may
    # already be at the right scope from the frontend; either way value
    # preservation is the contract.
    _apply(sdfg)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(a=a, b=out, K_in=np.int64(k_in), N=n)
    exp = np.zeros(n)
    for i in range(0, k_in + 1):
        exp[i] = a[i] * 2.0
    assert np.allclose(out, exp)


# ----------------------------------------------------------------------
# T11. Inside-fine: moving DOWN / sideways is never attempted
# ----------------------------------------------------------------------


def test_does_not_push_assignments_downward():
    """``kp1 = K + 1`` already at the SDFG root: the pass must NOT push
    it into the loop. It is already at its outermost legal scope."""
    sdfg = dace.SDFG('already_at_root')
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('kp1', dace.int64)
    pre = sdfg.add_state('pre', is_start_block=True)
    loop = _make_loop_with_iedge('post_loop', 'unused_in_body_loop_var', 'i')  # placeholder iedge
    loop.edges()[0].data.assignments.clear()  # clear the placeholder iedge content
    sdfg.add_node(loop)
    sdfg.add_edge(pre, loop, InterstateEdge(assignments={'kp1': 'K + 1'}))

    before_root = [(lhs, rhs) for e in sdfg.edges() for lhs, rhs in e.data.assignments.items()]
    assert ('kp1', 'K + 1') in before_root

    moved = _apply(sdfg)
    assert moved == 0
    after_root = [(lhs, rhs) for e in sdfg.edges() for lhs, rhs in e.data.assignments.items()]
    assert ('kp1', 'K + 1') in after_root
    assert not _assignments_inside_loops(sdfg)


# ----------------------------------------------------------------------
# T12. ICON pattern: per-i beg/end + inner maps reading [beg:end]
# ----------------------------------------------------------------------


def test_icon_pattern_per_i_beg_end_is_noop():
    """ICON-shape kernel: an outer ``i`` loop carries ``beg = foo(i);
    end = bar(i)`` as iedge assignments, then multiple inner loops whose
    ranges read ``[beg, end)``. Both ``beg`` and ``end`` depend on ``i``
    so neither can be hoisted past the outer ``i`` loop (L1 fails); the
    inner loops in turn cannot be moved DOWN past anything (that is a
    different pass anyway, and would itself need the symmetric check:
    ``range(beg, end)`` reads beg/end, so the inner ranges pin beg/end
    where they are). The pass must be a hard no-op AND the SDFG must
    stay valid + value-preserving.

    This is a critical safety contract: cascading invariants up is only
    sound when the moving symbol is truly invariant on every level we
    cross. The pass refuses every other move, including this one.
    """
    sdfg = dace.SDFG('icon_pattern')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('beg', dace.int64)
    sdfg.add_symbol('end', dace.int64)

    outer = LoopRegion('outer_i',
                       condition_expr='i < N',
                       loop_var='i',
                       initialize_expr='i = 0',
                       update_expr='i = i + 1')

    # Inner loops over [beg, end) -- modeled as LoopRegions with explicit
    # beg/end as range symbols. Three of them.
    setup_state = outer.add_state('setup', is_start_block=True)
    inner_a = LoopRegion('inner_a',
                         condition_expr='ja < end',
                         loop_var='ja',
                         initialize_expr='ja = beg',
                         update_expr='ja = ja + 1')
    inner_a.add_state('body_a', is_start_block=True)
    inner_b = LoopRegion('inner_b',
                         condition_expr='jb < end',
                         loop_var='jb',
                         initialize_expr='jb = beg',
                         update_expr='jb = jb + 1')
    inner_b.add_state('body_b', is_start_block=True)
    inner_c = LoopRegion('inner_c',
                         condition_expr='jc < end',
                         loop_var='jc',
                         initialize_expr='jc = beg',
                         update_expr='jc = jc + 1')
    inner_c.add_state('body_c', is_start_block=True)

    outer.add_node(inner_a)
    outer.add_node(inner_b)
    outer.add_node(inner_c)
    # setup -> inner_a, with the per-i beg/end assignments
    outer.add_edge(setup_state, inner_a, InterstateEdge(assignments={'beg': '2 * i', 'end': '2 * i + 8'}))
    outer.add_edge(inner_a, inner_b, InterstateEdge())
    outer.add_edge(inner_b, inner_c, InterstateEdge())

    sdfg.add_node(outer, is_start_block=True)

    moved = _apply(sdfg)
    assert moved == 0, ('per-i beg/end assignments depend on the outer loop variable; cascade-up must '
                        'refuse to move them past the outer loop')

    # The iedge stays exactly where it was: still inside the outer LoopRegion,
    # on the setup -> inner_a edge.
    found_beg, found_end = False, False
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        if cfg.label != 'outer_i':
            continue
        for e in cfg.edges():
            if 'beg' in e.data.assignments and '2*i' in str(e.data.assignments['beg']).replace(' ', ''):
                found_beg = True
            if 'end' in e.data.assignments and '2*i+8' in str(e.data.assignments['end']).replace(' ', ''):
                found_end = True
    assert found_beg and found_end, 'beg/end were moved away from the outer loop body'

    # SDFG remains valid.
    sdfg.validate()


@dace.program
def icon_pattern_frontend(arr: dace.float64[N, M], out: dace.float64[N]):
    """End-to-end Python-frontend ICON-shape: per-``i`` slice bounds
    feed two inner reductions whose ranges depend on ``i``. The promoted
    bound symbols (``i_times_2``, ``i_times_2_plus_4`` per the frontend
    heuristic) are introduced inside the outer ``i`` loop body and read
    by the inner ranges; cascade-up must NOT move them out (the rhs of
    each reads ``i``)."""
    for i in range(0, N):
        s1 = 0.0
        for j in range(2 * i, 2 * i + 4):
            s1 += arr[i, j]
        s2 = 0.0
        for j in range(2 * i, 2 * i + 4):
            s2 += arr[i, j] * 2.0
        out[i] = s1 + s2


def test_icon_pattern_frontend_value_preserving():
    n = 6
    m = 2 * n + 4  # comfortable upper bound: i in [0, n), j in [2*i, 2*i+4) < 2*n+4 = m
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((n, m)).astype(np.float64)
    sdfg = icon_pattern_frontend.to_sdfg(simplify=True)
    _apply(sdfg)
    sdfg.validate()
    out = np.zeros(n)
    sdfg(arr=arr, out=out, N=n, M=m)
    exp = np.zeros(n)
    for i in range(0, n):
        s1 = 0.0
        for j in range(2 * i, 2 * i + 4):
            s1 += arr[i, j]
        s2 = 0.0
        for j in range(2 * i, 2 * i + 4):
            s2 += arr[i, j] * 2.0
        exp[i] = s1 + s2
    assert np.allclose(out, exp)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
